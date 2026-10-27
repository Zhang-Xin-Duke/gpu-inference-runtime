import json
import os
import signal
import socket
import subprocess
import sys
import time
import pytest
from urllib.request import Request, urlopen


@pytest.mark.skipif(sys.platform == 'win32', reason='Windows SIGTERM terminates the process without invoking Python signal handlers')
def test_http_and_sigterm():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    proc = subprocess.Popen([sys.executable, "-m", "gpu_runtime.service", "--port", str(port)])
    base = f"http://127.0.0.1:{port}"
    try:
        for _ in range(150):
            try:
                with urlopen(base + "/readyz", timeout=.1) as response:
                    assert json.load(response)["ready"]
                    break
            except (OSError, TimeoutError): time.sleep(.1)
        else: raise AssertionError("service did not become ready")
        payload = json.dumps({"x": [[1.] * 256]}).encode()
        with urlopen(Request(base + "/infer", data=payload, headers={"Content-Type": "application/json"})) as response:
            output = json.load(response)["output"]
            assert len(output) == 1 and len(output[0]) == 32
        proc.send_signal(signal.SIGTERM)
        assert proc.wait(timeout=10) == 0
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
