"""A small bounded HTTP process contract for cluster-manager deployment."""
import argparse
import json
import signal
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch

from .models import DenseModel
from .runtime import AdmissionError, ModelSpec, Runtime


class Server(ThreadingHTTPServer):
    daemon_threads = False
    block_on_close = True
    request_queue_size = 32

    def __init__(self, address, runtime):
        self.runtime = runtime
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
    def setup(self):
        super().setup()
        self.connection.settimeout(5)

    def reply(self, status, value):
        payload = json.dumps(value).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
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
        elif self.path == "/metrics": self.reply(200, r.stats())
        else: self.reply(404, {"error": "unknown endpoint"})

    def do_POST(self):
        if self.path != "/infer": return self.reply(404, {"error": "unknown endpoint"})
        handle = None
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 1 << 20:
                return self.reply(413, {"error": "body must be 1..1048576 bytes"})
            data = json.loads(self.rfile.read(length))
            x = torch.tensor(data["x"], dtype=torch.float32)
            timeout_ms = float(data.get("timeout_ms", 1000))
            if not 0 < timeout_ms <= 10000:
                raise ValueError("timeout_ms must be between 0 and 10000")
            handle = self.server.runtime.submit("dense-v1", x, timeout_ms=timeout_ms)
            result = handle.result(timeout_ms / 1000 + 1)
            self.reply(200, {"output": result.tolist()})
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
    args = p.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(7)
    runtime = Runtime(device=args.device)
    runtime.register("dense-v1", ModelSpec(DenseModel(), "dense", 256, 32))
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
