from __future__ import annotations

from concurrent.futures import Future, InvalidStateError
from dataclasses import dataclass
import queue
import threading
import time
from typing import Literal

import torch

from ._scheduler import Scheduler


class AdmissionError(RuntimeError):
    pass


@dataclass(frozen=True)
class ModelSpec:
    model: torch.nn.Module
    kind: Literal["dense", "sequence"]
    width: int
    outputs: int
    max_length: int = 256
    vocabulary: int = 4096


@dataclass
class Record:
    tensors: tuple[torch.Tensor, ...]
    future: Future
    generation: ModelSpec
    samples: int
    bucket: int
    admitted: float
    deadline: float


class Handle:
    def __init__(self, runtime, request_id, future):
        self.runtime, self.request_id, self.future = runtime, request_id, future

    def cancel(self):
        return self.runtime.cancel(self.request_id)

    def result(self, timeout=None):
        return self.future.result(timeout)


class Slot:
    """A lease retained through the final D2H event, including cancelled work."""
    def __init__(self, spec, bucket, capacity, device):
        shapes = [(capacity, spec.width)] if spec.kind == "dense" else [(capacity, bucket), (capacity,)]
        dtype = torch.float32 if spec.kind == "dense" else torch.int64
        pinned = device.type == "cuda"
        self.signature = (spec.kind, spec.width, spec.outputs, bucket)
        self.host = [torch.empty(s, dtype=dtype, pin_memory=pinned) for s in shapes]
        self.device = [torch.empty_like(x, device=device) for x in self.host] if pinned else self.host
        self.output = torch.empty((capacity, spec.outputs), dtype=torch.float32, pin_memory=pinned)
        # Staging only: excludes weights, intermediate activations and allocator caches.
        self.bytes = sum(x.numel() * x.element_size() for x in self.host) * (2 if pinned else 1)
        self.bytes += self.output.numel() * self.output.element_size()
        self.events = [torch.cuda.Event(enable_timing=True) for _ in range(6)] if pinned else []


class Runtime:
    """Bounded multi-request batching for dense and variable-length models.

    Submitted CPU tensors must remain unchanged until completion. Request memory
    reservations estimate staging plus returned output, not total CUDA memory.
    Model versions are explicit; replacement of a registered version is rejected.
    """
    def __init__(self, *, device="cpu", max_batch=64, max_requests=1024,
                 max_request_bytes=256 << 20, staging_bytes=128 << 20,
                 max_inflight=2, delay_ms=2., policy="deadline", pipeline=True, shape_buckets=True):
        self.shape_buckets = shape_buckets
        self.backend_error = None
        self.quarantined = []
        self.device = torch.device(device)
        if self.device.type not in ("cpu", "cuda"):
            raise ValueError("only CPU and CUDA are supported")
        if staging_bytes <= 0:
            raise ValueError("staging_bytes must be positive")
        self.max_batch, self.staging_bytes = max_batch, staging_bytes
        self.max_inflight = max_inflight if pipeline and self.device.type == "cuda" else 1
        self.native = Scheduler(max_batch, max_requests, max_request_bytes,
                                self.max_inflight, delay_ms, policy)
        self.models: dict[str, ModelSpec] = {}
        self.records: dict[int, Record] = {}
        self.lock = threading.RLock()
        self.idle_slots: list[Slot] = []
        self.slot_count, self.pool_bytes, self.peak_pool_bytes = 0, 0, 0
        self.closed = False
        self.pending = queue.Queue()
        self.metrics = dict(completed=0, failed=0, cancelled=0, deadline_misses=0,
                            batches=0, samples=0, padded_tokens=0, useful_tokens=0,
                            pack_ms=0., h2d_ms=0., compute_ms=0., d2h_ms=0.)
        self.timelines = []
        if self.device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA requested but unavailable")
            with torch.cuda.device(self.device):
                self.copy_stream = torch.cuda.Stream(device=self.device)
                self.compute_stream = torch.cuda.Stream(device=self.device)
                self.output_stream = torch.cuda.Stream(device=self.device)
                self.origin = torch.cuda.Event(enable_timing=True)
                self.origin.record()
                self.origin.synchronize()
        self.worker = threading.Thread(target=self._dispatch, name="batch-dispatch", daemon=True)
        self.completer = threading.Thread(target=self._complete, name="event-completion", daemon=True)
        self.completer.start()
        self.worker.start()

    def register(self, version: str, spec: ModelSpec):
        if not version or ":" in version or spec.kind not in ("dense", "sequence"):
            raise ValueError("version must be nonempty without ':' and kind must be supported")
        if min(spec.width, spec.outputs, spec.max_length, spec.vocabulary) <= 0:
            raise ValueError("model dimensions must be positive")
        with self.lock:
            if self.closed or version in self.models:
                raise ValueError("closed runtime or duplicate model version")
            spec.model.eval().to(self.device)
            if self.device.type == "cuda": torch.cuda.synchronize(self.device)
            self.models[version] = spec

    def submit(self, version: str, *inputs: torch.Tensor, timeout_ms=1000.) -> Handle:
        with self.lock:
            if self.closed or self.backend_error: raise AdmissionError("runtime is draining or backend failed")
            spec = self.models[version]
            if not inputs or any(x.device.type != "cpu" or x.requires_grad for x in inputs):
                raise ValueError("inputs must be CPU tensors without gradients")
            x = inputs[0]
            if x.ndim != 2 or not 0 < x.shape[0] <= self.max_batch:
                raise ValueError("expected [samples, features/sequence] within max_batch")
            n = x.shape[0]
            if spec.kind == "dense":
                if len(inputs) != 1 or x.dtype != torch.float32 or x.shape[1] != spec.width:
                    raise ValueError("dense requires float32 [samples, width]")
                bucket = spec.width
            else:
                if (len(inputs) != 2 or x.dtype != torch.int64 or inputs[1].dtype != torch.int64
                    or inputs[1].shape != (n,) or not 0 < x.shape[1] <= spec.max_length):
                    raise ValueError("sequence requires int64 tokens [B,L] and lengths [B]")
                if (inputs[1].min().item() < 1 or inputs[1].max().item() > x.shape[1]
                    or x.min().item() < 0 or x.max().item() >= spec.vocabulary):
                    raise ValueError("invalid sequence length or token ID")
                bucket = (min(1 << (x.shape[1] - 1).bit_length(), spec.max_length)
                          if self.shape_buckets else spec.max_length)
            bytes_per = (spec.width * 4 if spec.kind == "dense" else bucket * 8 + 8)
            reservation = n * (bytes_per * 2 + spec.outputs * 4)
            future = Future()
            try:
                rid = self.native.offer(f"{version}:{bucket}", n, reservation, timeout_ms)
            except RuntimeError as exc:
                raise AdmissionError(str(exc)) from exc
            now = time.perf_counter()
            self.records[rid] = Record(inputs, future, spec, n, bucket, now, now + timeout_ms / 1000)
            return Handle(self, rid, future)

    def cancel(self, rid):
        with self.lock:
            record = self.records.get(rid)
            if record is None or record.future.done(): return False
            cancelled = record.future.cancel()
            if cancelled:
                if self.native.cancel(rid) == 1: self.records.pop(rid)
                self.metrics["cancelled"] += 1
            return cancelled

    def _slot(self, spec, bucket):
        signature = (spec.kind, spec.width, spec.outputs, bucket)
        with self.lock:
            for i, slot in enumerate(self.idle_slots):
                if slot.signature == signature: return self.idle_slots.pop(i)
            # Capacity is reserved at full batch size, avoiding allocator growth
            # for every partially filled batch. Evict only completed idle slots.
            estimate = self.max_batch * ((spec.width * 4 if spec.kind == "dense" else bucket * 8 + 8)
                                        * (2 if self.device.type == "cuda" else 1) + spec.outputs * 4)
            if estimate > self.staging_bytes:
                raise MemoryError("one staging slot exceeds configured staging budget")
            while self.idle_slots and (self.slot_count >= self.max_inflight
                                      or self.pool_bytes + estimate > self.staging_bytes):
                old = self.idle_slots.pop(0)
                self.slot_count -= 1
                self.pool_bytes -= old.bytes
            if self.slot_count >= self.max_inflight or self.pool_bytes + estimate > self.staging_bytes:
                raise MemoryError("staging budget exhausted by in-flight work")
            slot = Slot(spec, bucket, self.max_batch, self.device)
            self.slot_count += 1
            self.pool_bytes += slot.bytes
            self.peak_pool_bytes = max(self.peak_pool_bytes, self.pool_bytes)
            return slot

    @staticmethod
    def _set(future, *, value=None, error=None):
        try:
            if error is None: future.set_result(value)
            else: future.set_exception(error)
        except InvalidStateError:
            # Cancellation races completion. The backend still retains storage.
            pass

    def _dispatch(self):
        try:
            while True:
                batch = self.native.take()
                with self.lock:
                    for rid in batch.expired:
                        record = self.records.pop(rid)
                        self._set(record.future, error=TimeoutError("expired before execution"))
                        self.metrics["deadline_misses"] += 1
                if batch.stopped: break
                if not batch.ids: continue
                start = time.perf_counter()
                slot = None
                try:
                    with self.lock: records = [self.records[rid] for rid in batch.ids]
                    if self.backend_error: raise RuntimeError(self.backend_error)
                    spec, bucket = records[0].generation, records[0].bucket
                    slot = self._slot(spec, bucket)
                    offsets, pos = [], 0
                    for tensor in slot.host: tensor[:batch.samples].zero_()
                    for record in records:
                        end = pos + record.samples
                        slot.host[0][pos:end, :record.tensors[0].shape[1]].copy_(record.tensors[0])
                        if spec.kind == "sequence": slot.host[1][pos:end].copy_(record.tensors[1])
                        offsets.append((pos, end))
                        pos = end
                    packed = time.perf_counter()
                    with torch.inference_mode():
                        if self.device.type == "cuda":
                            with torch.cuda.device(self.device):
                                self.copy_stream.wait_stream(torch.cuda.current_stream(self.device))
                                with torch.cuda.stream(self.copy_stream):
                                    slot.events[0].record()
                                    for host, dev in zip(slot.host, slot.device):
                                        dev[:pos].copy_(host[:pos], non_blocking=True)
                                        dev.record_stream(self.copy_stream)
                                    slot.events[1].record()
                                with torch.cuda.stream(self.compute_stream):
                                    self.compute_stream.wait_event(slot.events[1])
                                    slot.events[2].record()
                                    output = spec.model(*(dev[:pos] for dev in slot.device))
                                    if output.shape != (pos, spec.outputs) or output.dtype != torch.float32:
                                        raise ValueError("model output must be float32 [samples, outputs]")
                                    slot.events[3].record()
                                with torch.cuda.stream(self.output_stream):
                                    self.output_stream.wait_event(slot.events[3])
                                    slot.events[4].record()
                                    slot.output[:pos].copy_(output, non_blocking=True)
                                    output.record_stream(self.output_stream)
                                    slot.events[5].record()
                        else:
                            output = spec.model(*(host[:pos] for host in slot.host))
                            if output.shape != (pos, spec.outputs) or output.dtype != torch.float32:
                                raise ValueError("model output must be float32 [samples, outputs]")
                            slot.output[:pos].copy_(output)
                    self.pending.put((batch, records, slot, offsets, start, packed, None, (time.perf_counter() - packed) * 1000))
                except Exception as exc:
                    # A partially launched batch may still own CUDA buffers.
                    if self.device.type == "cuda":
                        try: torch.cuda.synchronize(self.device)
                        except Exception as sync_error: self._poison(sync_error)
                    self.pending.put((batch, None, slot, None, start, time.perf_counter(), exc, 0.))
        finally:
            self.pending.put(None)

    def _complete(self):
        while True:
            item = self.pending.get()
            if item is None: return
            batch, records, slot, offsets, start, packed, error, cpu_ms = item
            stage = {}
            if error is None:
                try:
                    if self.device.type == "cuda":
                        # Polling this event never blocks the dispatcher. FIFO is
                        # safe because output copies share one ordered stream.
                        while not slot.events[5].query(): time.sleep(.0001)
                        stage = dict(h2d_ms=slot.events[0].elapsed_time(slot.events[1]),
                                     compute_ms=slot.events[2].elapsed_time(slot.events[3]),
                                     d2h_ms=slot.events[4].elapsed_time(slot.events[5]),
                                     h2d_start_ms=self.origin.elapsed_time(slot.events[0]),
                                     h2d_end_ms=self.origin.elapsed_time(slot.events[1]),
                                     compute_start_ms=self.origin.elapsed_time(slot.events[2]),
                                     compute_end_ms=self.origin.elapsed_time(slot.events[3]))
                    else: stage = dict(compute_ms=cpu_ms)
                    values = [slot.output[a:b].clone() for a, b in offsets]
                except Exception as exc:
                    error = exc
                    if self.device.type == "cuda":
                        try: torch.cuda.synchronize(self.device)
                        except Exception as sync_error: self._poison(sync_error)
            completed = time.perf_counter()
            with self.lock:
                for i, rid in enumerate(batch.ids):
                    record = self.records.pop(rid)
                    if record.future.cancelled(): continue
                    if error is not None:
                        self._set(record.future, error=error)
                        self.metrics["failed"] += 1
                    elif completed > record.deadline:
                        self._set(record.future, error=TimeoutError("deadline exceeded during execution"))
                        self.metrics["deadline_misses"] += 1
                    else:
                        self._set(record.future, value=values[i])
                        self.metrics["completed"] += 1
                self.metrics["batches"] += 1
                self.metrics["samples"] += batch.samples
                self.metrics["pack_ms"] += (packed - start) * 1000
                for name, value in stage.items():
                    if name in self.metrics: self.metrics[name] += value
                if records and records[0].generation.kind == "sequence":
                    self.metrics["padded_tokens"] += batch.samples * records[0].bucket
                    self.metrics["useful_tokens"] += sum(r.tensors[1].sum().item() for r in records)
                if slot is not None:
                    if self.backend_error: self.quarantined.append(slot)
                    else: self.idle_slots.append(slot)
                self.native.finish(batch.ids, (completed - start) * 1000)
                if len(self.timelines) < 10000:
                    self.timelines.append(dict(start=start, packed=packed, end=completed,
                                               samples=batch.samples, **stage))

    def _poison(self, error):
        # A device-context failure requires process restart. Keep uncertain
        # storage quarantined and terminally fail remaining accepted requests.
        with self.lock:
            self.backend_error = str(error)
            self.closed = True
            self.native.close()

    def stats(self):
        with self.lock:
            return self.native.stats() | self.metrics | dict(
                staging_bytes=self.pool_bytes, peak_staging_bytes=self.peak_pool_bytes,
                backend_failed=self.backend_error is not None,
                ready=bool(self.models) and not self.closed and self.worker.is_alive() and self.completer.is_alive())

    def shutdown(self, wait=True):
        with self.lock:
            self.closed = True
            self.native.close()
        if wait:
            if threading.current_thread() in (self.worker, self.completer):
                raise RuntimeError("blocking shutdown cannot run in a runtime callback")
            self.worker.join()
            self.completer.join()
            with self.lock:
                self.idle_slots.clear()
                self.pool_bytes, self.slot_count = 0, 0

    def __enter__(self): return self
    def __exit__(self, *_): self.shutdown()
