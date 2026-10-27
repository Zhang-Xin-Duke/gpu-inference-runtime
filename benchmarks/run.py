"""Open-loop local harness. Raw results include offered-load and load-generator lag."""
import argparse
from concurrent.futures import CancelledError
import json
import platform
from pathlib import Path
import statistics
import subprocess
import sys
import threading
import time

import numpy as np
import torch
from gpu_runtime.models import DenseModel, SequenceModel
from gpu_runtime.runtime import Runtime, ModelSpec, AdmissionError


def hardware(device):
    info = dict(platform=platform.platform(), python=platform.python_version(),
                torch=torch.__version__, cuda=torch.version.cuda, device=device,
                torch_threads=torch.get_num_threads(),
                float32_matmul_precision=torch.get_float32_matmul_precision(),
                matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                measurement_clock=dict(name='perf_counter',
                    implementation=time.get_clock_info('perf_counter').implementation,
                    resolution_s=time.get_clock_info('perf_counter').resolution))
    if device == "cuda":
        prop = torch.cuda.get_device_properties(0)
        info.update(gpu=prop.name, gpu_bytes=prop.total_memory,
                    capability=list(torch.cuda.get_device_capability(0)))
        try:
            info["nvidia_smi"] = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True).strip()
        except (OSError, subprocess.SubprocessError): pass
    return info


def run(args, kind, mode):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    max_batch = args.samples if mode == "unbatched" else 64
    runtime = Runtime(device=args.device, max_batch=max_batch, max_requests=args.max_requests,
                      delay_ms=0 if mode == "unbatched" else args.delay_ms,
                      pipeline=mode == "pipeline", policy=args.policy,
                      shape_buckets=not args.no_buckets)
    model = (DenseModel(args.features, args.hidden, 32) if kind == "dense"
             else SequenceModel(4096, args.embedding, 32))
    spec = ModelSpec(model, kind, args.features if kind == "dense" else args.embedding, 32)
    runtime.register("v1", spec)
    lengths = np.random.choice([5, 15, 30, 60, 120, 240], args.requests)
    inputs = [(torch.randn(args.samples, args.features),) if kind == "dense" else (
        torch.randint(0, 4096, (args.samples, int(length))), torch.full((args.samples,), int(length))) for length in lengths]
    # Warm model and adapter, excluding startup and first-use allocation.
    for _ in range(20): runtime.submit("v1", *inputs[0], timeout_ms=10000).result(15)
    start_stats = runtime.stats()
    runtime.timelines.clear()
    durations, statuses, lag = [], [], []
    lock = threading.Lock()
    handles = []
    intervals = np.random.exponential(1 / args.rps, args.requests)
    targets = np.cumsum(intervals)
    if args.burst:
        # Tenfold arrivals in the first half of each 200 ms interval.
        targets = np.array([int(x / .2) * .2 + (x % .2) / 10 for x in targets])
    begin = time.perf_counter()
    def done(future, target):
        stamp = time.perf_counter()
        try:
            future.result()
            status = "ok" if (stamp - target) * 1000 <= args.deadline_ms else "late_completion"
        except TimeoutError: status = "expired"
        except CancelledError: status = "cancelled"
        except Exception: status = "failed"
        with lock:
            durations.append((stamp - target) * 1000)
            statuses.append(status)
    try:
        for inputs_i, offset in zip(inputs, targets):
            target = begin + float(offset)
            remaining = target - time.perf_counter()
            if remaining > 0: time.sleep(remaining)
            offered = time.perf_counter()
            lag.append(max(0., (offered - target) * 1000))
            try:
                remaining_ms = args.deadline_ms - (offered - target) * 1000
                if remaining_ms <= 0:
                    statuses.append("generator_late")
                    continue
                handle = runtime.submit("v1", *inputs_i, timeout_ms=remaining_ms)
                handle.future.add_done_callback(lambda f, t=target: done(f, t))
                handles.append(handle)
            except AdmissionError: statuses.append("rejected")
        runtime.shutdown()
    finally:
        runtime.shutdown()
    elapsed = time.perf_counter() - begin
    stats = runtime.stats()
    timeline = runtime.timelines.copy()
    overlap = sum(max(0., min(a["h2d_end_ms"], b["compute_end_ms"])
                      - max(a["h2d_start_ms"], b["compute_start_ms"]))
                  for i, a in enumerate(timeline) if "h2d_start_ms" in a
                  for j, b in enumerate(timeline) if i != j and "compute_start_ms" in b)
    result = dict(kind=kind, mode=mode, elapsed_s=elapsed, requests=args.requests,
                  offered_rps=args.rps, completed=statuses.count("ok"),
                  statuses={x: statuses.count(x) for x in sorted(set(statuses))},
                  goodput_rps=statuses.count("ok") / elapsed,
                  p50_ms=float(np.percentile(durations, 50)) if durations else None,
                  p95_ms=float(np.percentile(durations, 95)) if durations else None,
                  p99_ms=float(np.percentile(durations, 99)) if durations else None,
                  generator_lag_p99_ms=float(np.percentile(lag, 99)),
                  measured_cross_batch_h2d_compute_overlap_ms=overlap,
                  metrics={k: stats[k] - start_stats[k] for k in (
                      "accepted", "rejected", "expired", "completed", "failed", "cancelled",
                      "deadline_misses", "batches", "samples", "padded_tokens", "useful_tokens",
                      "pack_ms", "h2d_ms", "compute_ms", "d2h_ms")},
                  peak_staging_bytes=stats["peak_staging_bytes"],
                  peak_reserved_bytes=stats["peak_reserved_bytes"],
                  timeline=timeline)
    # Correctness is separately measured outside the timed arrival stream.
    with torch.inference_mode(): reference = model(*(x.to(args.device) for x in inputs[0])).cpu()
    with Runtime(device=args.device, max_batch=max(16, args.samples)) as check:
        check.register("v1", spec)
        actual = check.submit("v1", *inputs[0], timeout_ms=10000).result(15)
        torch.testing.assert_close(actual, reference, atol=1e-5, rtol=1e-4)
    result["correctness"] = "passed (rtol=1e-4, atol=1e-5)"
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--requests", type=int, default=1000)
    p.add_argument("--rps", type=float, default=500)
    p.add_argument("--deadline-ms", type=float, default=100)
    p.add_argument("--delay-ms", type=float, default=2)
    p.add_argument("--max-requests", type=int, default=256)
    p.add_argument("--policy", choices=["age", "deadline"], default="deadline")
    p.add_argument("--no-buckets", action="store_true")
    p.add_argument("--burst", action="store_true")
    p.add_argument("--features", type=int, default=256)
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--embedding", type=int, default=64)
    p.add_argument("--samples", type=int, default=1)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--kind", choices=["dense", "sequence", "all"], default="all")
    p.add_argument("--modes", nargs="+", choices=["unbatched", "sync", "pipeline"],
                   default=["unbatched", "sync", "pipeline"])
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.requests < 1 or args.rps <= 0 or not 1 <= args.samples <= 64: p.error("requests and rps must be positive")
    torch.set_num_threads(args.threads)
    kinds = ["dense", "sequence"] if args.kind == "all" else [args.kind]
    result = dict(hardware=hardware(args.device), command=sys.argv,
                  config={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, runs=[])
    for kind in kinds:
        for mode in args.modes:
            row = run(args, kind, mode)
            result["runs"].append(row)
            print(json.dumps({k: v for k, v in row.items() if k not in ("timeline", "metrics")}))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__": main()
