"""Request-level data parallel inference over identical remote model replicas."""
from concurrent.futures import Future, ThreadPoolExecutor, InvalidStateError
from dataclasses import dataclass
import http.client
import json
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
    cooldown_until: float = 0.


class Router:
    def __init__(self, endpoints, *, max_pending=256, concurrency=64, policy="least_work",
                 retry=True, cooldown_s=2.):
        if not endpoints or len(set(endpoints)) != len(endpoints): raise ValueError("unique endpoints required")
        if min(max_pending, concurrency) <= 0 or policy not in ("least_work", "round_robin"):
            raise ValueError("invalid router limits or policy")
        self.nodes = [Node(url.rstrip("/")) for url in endpoints]
        self.lock = threading.Lock()
        self.local = threading.local()
        self.connections = set()
        self.per_node_limit = 16
        self.policy, self.retry, self.cooldown_s = policy, retry, cooldown_s
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
            now = time.monotonic()
            candidates = [i for i,n in enumerate(self.nodes) if i not in excluded and n.cooldown_until <= now and n.active < self.per_node_limit]
            if not candidates: raise ConnectionError("no available replicas")
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
        deadline = time.monotonic() + timeout_ms / 1000
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
                remaining = deadline-time.monotonic()
                if remaining <= 0: raise TimeoutError("router queue deadline exceeded")
                selected = self._choose(excluded, samples)
                node = self.nodes[selected]
                excluded.add(selected)
                try:
                    cache = getattr(self.local, "connections", None)
                    if cache is None: cache = self.local.connections = {}
                    conn = cache.get(selected)
                    if conn is None:
                        conn = cache[selected] = self._connection(node, remaining)
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
                    if time.monotonic() > deadline: raise TimeoutError("RPC exceeded deadline")
                    with self.lock: node.completed += 1
                    return result[0]
                except (OSError, http.client.HTTPException, TimeoutError) as exc:
                    last_error = exc
                    conn = getattr(self.local, "connections", {}).pop(selected, None)
                    if conn is not None:
                        conn.close()
                        with self.lock: self.connections.discard(conn)
                    with self.lock:
                        node.failures += 1
                        node.cooldown_until = time.monotonic() + (min(.05,self.cooldown_s) if isinstance(exc,TimeoutError) else self.cooldown_s)
                finally:
                    with self.lock:
                        node.outstanding -= samples
                        node.active -= 1
            raise last_error or ConnectionError("no replica completed request")
        finally: self._release()

    def stats(self):
        with self.lock:
            return dict(accepted=self.accepted, rejected=self.rejected, finished=self.finished,
                        active=self.active, nodes=[dict(endpoint=n.url, outstanding=n.outstanding, active=n.active,
                        requests=n.requests, completed=n.completed, failures=n.failures,
                        cooling_down=n.cooldown_until > time.monotonic()) for n in self.nodes])

    def shutdown(self):
        with self.lock: self.closed = True
        self.pool.shutdown(wait=True)
        for conn in self.connections: conn.close()

    def __enter__(self): return self
    def __exit__(self,*_): self.shutdown()
