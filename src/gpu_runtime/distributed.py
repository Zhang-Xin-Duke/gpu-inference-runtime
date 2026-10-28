"""Request-level data parallel inference over identical remote model replicas."""
from concurrent.futures import Future, ThreadPoolExecutor, InvalidStateError
from collections import Counter
from dataclasses import dataclass, field
import http.client
import json
import math
import threading
import time
from urllib.parse import urlsplit

from . import wire
from .runtime import AdmissionError


@dataclass
class Node:
    url: str
    outstanding: int = 0
    active: int = 0
    requests: int = 0
    completed: int = 0
    failures: int = 0
    stale_connections: int = 0
    retired_idle_connections: int = 0
    cooldown_until: float = 0.
    connection_epoch: int = 0
    idle_connections: list = field(default_factory=list, repr=False)


class Router:
    def __init__(self, endpoints, *, max_pending=256, concurrency=64, policy="least_work",
                 retry=True, cooldown_s=2., keepalive_s=4.):
        if not endpoints or len(set(endpoints)) != len(endpoints): raise ValueError("unique endpoints required")
        if min(max_pending, concurrency) <= 0 or policy not in ("least_work", "round_robin"):
            raise ValueError("invalid router limits or policy")
        if not math.isfinite(keepalive_s) or keepalive_s < 0:
            raise ValueError("keepalive_s must be finite and nonnegative")
        self.nodes = [Node(url.rstrip("/")) for url in endpoints]
        self.lock = threading.Lock()
        self.connections = set()
        self.per_node_limit = 16
        self.policy, self.retry, self.cooldown_s = policy, retry, cooldown_s
        self.keepalive_s = keepalive_s
        self.attempt_errors = Counter()
        self.routing_unavailable = 0
        self.cursor = 0
        self.closed = False
        self.accepted = self.rejected = self.finished = self.active = 0
        self.permits = threading.BoundedSemaphore(max_pending)
        self.metadata = [self._get(node, "/metadata") for node in self.nodes]
        if any(m != self.metadata[0] for m in self.metadata[1:]):
            raise ValueError("replicas must have identical model metadata and weight fingerprints")
        self.model_version = self.metadata[0]["model_version"]
        self.hardware = [self._get(node,"/hardware") for node in self.nodes]
        self.pool = ThreadPoolExecutor(min(concurrency, 32, len(self.nodes)*self.per_node_limit), thread_name_prefix="replica-rpc")

    @staticmethod
    def _connection(node, timeout):
        parsed = urlsplit(node.url)
        if parsed.scheme not in ("http", "https") or parsed.path not in ("", "/"):
            raise ValueError("endpoint must be an HTTP(S) origin")
        cls = http.client.HTTPSConnection if parsed.scheme == "https" else http.client.HTTPConnection
        return cls(parsed.hostname, parsed.port, timeout=timeout)

    def _get(self, node, path):
        conn = self._connection(node, 3)
        try:
            conn.request("GET", path)
            response = conn.getresponse()
            body = response.read()
            if response.status != 200: raise ConnectionError(f"replica returned {response.status}")
            return json.loads(body)
        finally: conn.close()

    def _choose(self, excluded, samples):
        with self.lock:
            now = time.perf_counter()
            candidates = [i for i,n in enumerate(self.nodes) if i not in excluded and n.cooldown_until <= now and n.active < self.per_node_limit]
            if not candidates:
                self.routing_unavailable += 1
                raise ConnectionError("no available replicas")
            if self.policy == "round_robin":
                selected = min(candidates, key=lambda i: (i-self.cursor) % len(self.nodes))
            else:
                selected = min(candidates, key=lambda i: (self.nodes[i].outstanding,
                                                          (i-self.cursor) % len(self.nodes)))
            self.cursor = (selected+1) % len(self.nodes)
            self.nodes[selected].outstanding += samples
            self.nodes[selected].active += 1
            self.nodes[selected].requests += 1
            return selected

    def submit(self, *inputs, timeout_ms=1000):
        if not inputs or inputs[0].ndim != 2 or inputs[0].shape[0] <= 0:
            raise ValueError("expected tensors with a nonempty sample dimension")
        if not 0 < timeout_ms <= 10000: raise ValueError("timeout_ms must be in (0,10000]")
        deadline = time.perf_counter() + timeout_ms / 1000
        with self.lock:
            if self.closed or not self.permits.acquire(blocking=False):
                self.rejected += 1
                raise AdmissionError("router is draining or full")
            self.accepted += 1
            self.active += 1
        future = Future()
        def execute():
            if future.cancelled():
                self._release()
                return
            try:
                value = self._execute(inputs, deadline)
            except BaseException as exc:
                try: future.set_exception(exc)
                except InvalidStateError: pass
            else:
                try: future.set_result(value)
                except InvalidStateError: pass
        try:
            self.pool.submit(execute)
            return future
        except BaseException:
            self._release()
            raise

    def _release(self):
        with self.lock:
            self.active -= 1
            self.finished += 1
            self.permits.release()

    def _execute(self, inputs, deadline):
        samples, excluded = inputs[0].shape[0], set()
        last_error = None
        try:
            for _ in range(2 if self.retry else 1):
                remaining = deadline-time.perf_counter()
                if remaining <= 0: raise TimeoutError("router queue deadline exceeded")
                selected = self._choose(excluded, samples)
                node = self.nodes[selected]
                excluded.add(selected)
                conn = None
                reusable = False
                reused = False
                try:
                    with self.lock:
                        epoch = node.connection_epoch
                        while node.idle_connections:
                            candidate, returned = node.idle_connections.pop()
                            if time.perf_counter() - returned < self.keepalive_s:
                                conn = candidate
                                reused = True
                                break
                            candidate.close()
                            self.connections.discard(candidate)
                            node.retired_idle_connections += 1
                    if conn is None:
                        conn = self._connection(node, remaining)
                        with self.lock: self.connections.add(conn)
                    conn.timeout = remaining
                    if conn.sock is not None: conn.sock.settimeout(remaining)
                    payload = wire.encode(inputs, model_version=self.model_version,
                                          timeout_ms=min(10000., remaining*1000))
                    conn.request("POST", "/infer-binary", payload,
                                 {"Content-Type": "application/octet-stream"})
                    response = conn.getresponse()
                    body = response.read(wire.MAX_BODY+1)
                    if response.status == 504: raise TimeoutError("replica execution deadline exceeded")
                    if response.status == 503: raise ConnectionError("replica overloaded")
                    if response.status != 200: raise ValueError(f"replica rejected request: HTTP {response.status}")
                    result, _ = wire.decode(body)
                    if len(result) != 1 or result[0].shape != (samples, self.metadata[0]["outputs"]):
                        raise ValueError("invalid replica output schema")
                    if time.perf_counter() > deadline: raise TimeoutError("RPC exceeded deadline")
                    with self.lock: node.completed += 1
                    reusable = True
                    return result[0]
                except (OSError, http.client.HTTPException, TimeoutError) as exc:
                    last_error = exc
                    stale = reused and isinstance(exc, (http.client.RemoteDisconnected,
                                                          BrokenPipeError, ConnectionResetError))
                    with self.lock:
                        node.failures += 1
                        self.attempt_errors[type(exc).__name__] += 1
                        # Idle sockets are exclusively owned by this pool.
                        # Active leases close on return if their epoch changed.
                        node.connection_epoch += 1
                        for idle, _ in node.idle_connections:
                            idle.close()
                            self.connections.discard(idle)
                        node.idle_connections.clear()
                        if stale:
                            # A server may retire an idle socket while remaining
                            # healthy. Invalidate stale leases without parking the
                            # entire replica. A single-worker router may reconnect
                            # within its existing two-attempt/deadline budget.
                            node.stale_connections += 1
                            if len(self.nodes) == 1: excluded.discard(selected)
                        else:
                            node.cooldown_until = time.perf_counter() + (min(.05,self.cooldown_s) if isinstance(exc,TimeoutError) else self.cooldown_s)
                finally:
                    with self.lock:
                        if conn is not None:
                            if reusable and epoch == node.connection_epoch and not self.closed:
                                node.idle_connections.append((conn, time.perf_counter()))
                            else:
                                conn.close()
                                self.connections.discard(conn)
                        node.outstanding -= samples
                        node.active -= 1
            raise last_error or ConnectionError("no replica completed request")
        finally: self._release()

    def stats(self):
        with self.lock:
            return dict(accepted=self.accepted, rejected=self.rejected, finished=self.finished,
                        active=self.active, open_connections=len(self.connections),
                        attempt_errors=dict(self.attempt_errors), routing_unavailable=self.routing_unavailable,
                        nodes=[dict(endpoint=n.url, outstanding=n.outstanding, active=n.active,
                        requests=n.requests, completed=n.completed, failures=n.failures,
                        stale_connections=n.stale_connections,
                        retired_idle_connections=n.retired_idle_connections,
                        idle_connections=len(n.idle_connections),
                        cooling_down=n.cooldown_until > time.perf_counter()) for n in self.nodes])

    def shutdown(self):
        with self.lock: self.closed = True
        self.pool.shutdown(wait=True)
        with self.lock:
            for conn in self.connections: conn.close()
            self.connections.clear()
            for node in self.nodes: node.idle_connections.clear()

    def __enter__(self): return self
    def __exit__(self,*_): self.shutdown()
