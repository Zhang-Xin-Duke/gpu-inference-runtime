"""A small bounded HTTP process contract for cluster-manager deployment."""
import argparse
import hashlib
import json
import signal
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch

from .models import DenseModel, SequenceModel
from . import wire
from .runtime import AdmissionError, ModelSpec, Runtime


class Server(ThreadingHTTPServer):
    daemon_threads = False
    block_on_close = True
    request_queue_size = 32

    def __init__(self, address, runtime):
        self.runtime = runtime
        self.model_version = next(iter(runtime.models))
        spec = runtime.models[self.model_version]
        digest = hashlib.sha256()
        for name, tensor in sorted(spec.model.state_dict().items()):
            digest.update(name.encode())
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        self.metadata = dict(model_version=self.model_version, fingerprint=digest.hexdigest(),
                             kind=spec.kind, width=spec.width, outputs=spec.outputs,
                             max_length=spec.max_length, vocabulary=spec.vocabulary)
        self.hardware = dict(device=str(runtime.device), torch=torch.__version__,
                             cuda=torch.version.cuda,
                             gpu=torch.cuda.get_device_name(runtime.device) if runtime.device.type=="cuda" else None)
        self.permits = threading.BoundedSemaphore(32)
        super().__init__(address, Handler)

    def process_request(self, request, address):
        if not self.permits.acquire(blocking=False):
            try: request.sendall(b"HTTP/1.1 503 Service Unavailable\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
            finally: self.shutdown_request(request)
            return
        try: super().process_request(request, address)
        except BaseException:
            self.permits.release()
            raise

    def process_request_thread(self, request, address):
        try: super().process_request_thread(request, address)
        finally: self.permits.release()


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    def setup(self):
        super().setup()
        self.connection.settimeout(5)
        self.connection.setsockopt(socket.IPPROTO_TCP,socket.TCP_NODELAY,1)

    def reply(self, status, value):
        payload = json.dumps(value).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        try: self.wfile.write(payload)
        except (BrokenPipeError, ConnectionResetError): pass

    def reply_binary(self, value):
        payload = wire.encode((value,))
        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        try: self.wfile.write(payload)
        except (BrokenPipeError, ConnectionResetError): pass

    def log_message(self, *_):
        pass

    def do_GET(self):
        r = self.server.runtime
        if self.path == "/healthz":
            alive = r.worker.is_alive() and r.completer.is_alive()
            self.reply(200 if alive else 503, {"alive": alive})
        elif self.path == "/readyz":
            ready = r.stats()["ready"]
            self.reply(200 if ready else 503, {"ready": ready})
        elif self.path == "/hardware": self.reply(200, self.server.hardware)
        elif self.path == "/metadata": self.reply(200, self.server.metadata)
        elif self.path == "/metrics": self.reply(200, r.stats())
        else: self.reply(404, {"error": "unknown endpoint"})

    def do_POST(self):
        binary = self.path == "/infer-binary"
        if self.path not in ("/infer", "/infer-binary"):
            self.close_connection = True
            return self.reply(404, {"error": "unknown endpoint"})
        handle = None
        try:
            length = int(self.headers.get("Content-Length", "0"))
            limit = wire.MAX_BODY if binary else 1 << 20
            if length <= 0 or length > limit:
                self.close_connection = True
                return self.reply(413, {"error": "body size limit exceeded"})
            body = self.rfile.read(length)
            if len(body) != length: raise ValueError("truncated body")
            if binary:
                inputs, data = wire.decode(body)
                if data.get("model_version") != self.server.model_version:
                    raise ValueError("model version mismatch")
            else:
                data = json.loads(body)
                inputs = (torch.tensor(data["x"], dtype=torch.float32),)
            timeout_ms = float(data.get("timeout_ms", 1000))
            if not 0 < timeout_ms <= 10000:
                raise ValueError("timeout_ms must be between 0 and 10000")
            handle = self.server.runtime.submit(self.server.model_version, *inputs, timeout_ms=timeout_ms)
            result = handle.result(timeout_ms / 1000 + 1)
            if binary: self.reply_binary(result)
            else: self.reply(200, {"output": result.tolist()})
        except AdmissionError as exc: self.reply(503, {"error": str(exc)})
        except TimeoutError:
            if handle is not None: handle.cancel()
            self.reply(504, {"error": "deadline exceeded"})
        except (ValueError, TypeError, KeyError, RuntimeError) as exc:
            self.reply(400, {"error": str(exc)})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--device", default="cpu")
    p.add_argument("--kind", choices=["dense", "sequence"], default="dense")
    p.add_argument("--features", type=int, default=256)
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--embedding", type=int, default=64)
    p.add_argument("--max-batch", type=int, default=64)
    p.add_argument("--max-inflight", type=int, default=2)
    p.add_argument("--delay-ms", type=float, default=2.)
    args = p.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(7)
    runtime = Runtime(device=args.device, max_batch=args.max_batch,
                      max_inflight=args.max_inflight, delay_ms=args.delay_ms)
    model = (DenseModel(args.features, args.hidden, 32) if args.kind == "dense"
             else SequenceModel(4096, args.embedding, 32))
    width = args.features if args.kind == "dense" else args.embedding
    runtime.register(args.kind + "-v1", ModelSpec(model, args.kind, width, 32))
    server = Server((args.host, args.port), runtime)
    def drain(*_):
        runtime.shutdown(wait=False)
        threading.Thread(target=server.shutdown, daemon=True).start()
    signal.signal(signal.SIGTERM, drain)
    signal.signal(signal.SIGINT, drain)
    try: server.serve_forever(poll_interval=.1)
    finally:
        runtime.shutdown()
        server.server_close()


if __name__ == "__main__": main()
