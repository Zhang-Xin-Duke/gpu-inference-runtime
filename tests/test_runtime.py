import time
import threading
import pytest
import torch
from gpu_runtime.models import DenseModel, SequenceModel
from gpu_runtime.runtime import Runtime, ModelSpec, AdmissionError


@pytest.fixture(autouse=True)
def seed():
    torch.manual_seed(7)
    torch.set_num_threads(1)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.cuda)])
@pytest.mark.parametrize("kind", ["dense", "sequence"])
def test_batch_matches_direct_model_and_versions(device, kind):
    if device == "cuda" and not torch.cuda.is_available(): pytest.skip("CUDA unavailable")
    model = DenseModel(16, 32, 4) if kind == "dense" else SequenceModel(100, 16, 4)
    with Runtime(device=device, max_batch=16, delay_ms=10) as r:
        spec = ModelSpec(model, kind, 16, 4, vocabulary=100)
        r.register("v1", spec)
        r.register("v2", ModelSpec(DenseModel(16, 32, 4), "dense", 16, 4))
        items, handles = [], []
        for n, length in [(2, 3), (1, 4), (3, 7), (1, 3), (2, 8)]:
            inputs = (torch.randn(n, 16),) if kind == "dense" else (
                torch.randint(0, 100, (n, length)), torch.full((n,), length))
            items.append(inputs)
            handles.append(r.submit("v1", *inputs, timeout_ms=5000))
        other = r.submit("v2", torch.randn(1, 16), timeout_ms=5000)
        for inputs, handle in zip(items, handles):
            with torch.inference_mode(): expected = model(*(x.to(device) for x in inputs)).cpu()
            torch.testing.assert_close(handle.result(10), expected, rtol=1e-4, atol=1e-5)
        other.result(10)
        assert r.stats()["reserved_bytes"] == 0
        assert r.stats()["completed"] == 6
        # Values are owned CPU results; future slot reuse cannot overwrite them.
        before = handles[0].result().clone()
        r.submit("v1", *items[0], timeout_ms=5000).result(10)
        torch.testing.assert_close(handles[0].result(), before)
    assert r.stats()["staging_bytes"] == 0


def test_cancel_queued_and_drain():
    r = Runtime(max_batch=16, delay_ms=10000)
    r.register("v1", ModelSpec(DenseModel(8, 16, 2), "dense", 8, 2))
    a = r.submit("v1", torch.ones(1, 8), timeout_ms=30000)
    b = r.submit("v1", torch.ones(1, 8), timeout_ms=30000)
    assert a.cancel()
    r.shutdown()
    assert a.future.cancelled()
    assert b.result().shape == (1, 2)
    assert r.stats()["reserved_bytes"] == 0
    with pytest.raises(AdmissionError): r.submit("v1", torch.ones(1, 8))


def test_failure_releases_permits_and_next_batch_runs():
    class Broken(torch.nn.Module):
        def forward(self, x): raise ValueError("injected failure")
    with Runtime(max_batch=4, delay_ms=0) as r:
        r.register("broken", ModelSpec(Broken(), "dense", 8, 2))
        r.register("good", ModelSpec(DenseModel(8, 16, 2), "dense", 8, 2))
        with pytest.raises(ValueError, match="injected"):
            r.submit("broken", torch.ones(1, 8)).result(3)
        assert r.submit("good", torch.ones(1, 8)).result(3).shape == (1, 2)
        assert r.stats()["reserved_bytes"] == 0


def test_running_cancel_retains_capacity_until_backend_done():
    entered, release = threading.Event(), threading.Event()
    class Slow(torch.nn.Module):
        def forward(self, x):
            entered.set()
            assert release.wait(3)
            return x[:, :2]
    r = Runtime(max_batch=4, max_requests=1, delay_ms=0)
    r.register("v1", ModelSpec(Slow(), "dense", 8, 2))
    handle = r.submit("v1", torch.ones(1, 8))
    assert entered.wait(3)
    assert handle.cancel()
    with pytest.raises(AdmissionError): r.submit("v1", torch.ones(1, 8))
    assert r.stats()["reserved_bytes"] > 0
    release.set()
    r.shutdown()
    assert r.stats()["reserved_bytes"] == 0


def test_staging_budget_failure_is_terminal():
    with Runtime(max_batch=8, staging_bytes=8, delay_ms=0) as r:
        r.register("v1", ModelSpec(DenseModel(8, 16, 2), "dense", 8, 2))
        with pytest.raises(MemoryError): r.submit("v1", torch.ones(1, 8)).result(3)
        assert r.stats()["reserved_bytes"] == 0


def test_validation_and_duplicate_generation():
    with Runtime() as r:
        spec = ModelSpec(SequenceModel(100, 16, 4), "sequence", 16, 4, vocabulary=100)
        r.register("v1", spec)
        with pytest.raises(ValueError): r.register("v1", spec)
        with pytest.raises(ValueError): r.submit("v1", torch.ones(1, 4, dtype=torch.int64), torch.tensor([0]))
        assert r.stats()["accepted"] == 0


def test_queued_deadline_and_cancel_metrics():
    entered, release = threading.Event(), threading.Event()
    class Slow(torch.nn.Module):
        def forward(self, x):
            entered.set()
            assert release.wait(3)
            return x[:, :2]
    r = Runtime(max_batch=4, delay_ms=0)
    r.register("v1", ModelSpec(Slow(), "dense", 8, 2))
    a = r.submit("v1", torch.ones(1, 8), timeout_ms=3000)
    assert entered.wait(3)
    b = r.submit("v1", torch.ones(1, 8), timeout_ms=5)
    assert a.cancel() and not a.cancel()
    time.sleep(.02)
    release.set()
    with pytest.raises(TimeoutError): b.result(3)
    r.shutdown()
    assert r.stats()["cancelled"] == 1 and r.stats()["reserved_bytes"] == 0


def test_concurrent_submit_order_and_no_permit_leak():
    from concurrent.futures import ThreadPoolExecutor
    with Runtime(max_batch=32, max_requests=256, delay_ms=1) as r:
        model = DenseModel(8, 16, 2)
        r.register("v1", ModelSpec(model, "dense", 8, 2))
        inputs = [torch.full((i % 3 + 1, 8), float(i)) for i in range(100)]
        with ThreadPoolExecutor(8) as pool:
            handles = list(pool.map(lambda x: r.submit("v1", x, timeout_ms=5000), inputs))
        for x, h in zip(inputs, handles):
            with torch.inference_mode(): expected = model(x)
            torch.testing.assert_close(h.result(5), expected)
        assert r.stats()["completed"] == 100
        assert r.stats()["reserved_bytes"] == 0


def test_shape_bucket_reduces_padding_for_same_inputs():
    measurements = []
    for buckets in (False, True):
        with Runtime(max_batch=8, delay_ms=5, shape_buckets=buckets) as r:
            r.register("v1", ModelSpec(SequenceModel(100, 16, 4), "sequence", 16, 4,
                                      max_length=128, vocabulary=100))
            handles = [r.submit("v1", torch.ones(1, 5, dtype=torch.int64), torch.tensor([5]))
                       for _ in range(4)]
            for handle in handles: handle.result(3)
            measurements.append(r.stats())
    assert measurements[0]["useful_tokens"] == measurements[1]["useful_tokens"] == 20
    assert measurements[0]["padded_tokens"] == 512
    assert measurements[1]["padded_tokens"] == 32


@pytest.mark.cuda
def test_cuda_slot_reuse_and_shape_switch():
    if not torch.cuda.is_available(): pytest.skip("CUDA unavailable")
    with Runtime(device="cuda", max_batch=8, max_inflight=2, delay_ms=1) as r:
        model = SequenceModel(100, 16, 4)
        r.register("v1", ModelSpec(model, "sequence", 16, 4, vocabulary=100))
        old = None
        for length in [3, 7, 15, 31, 63, 127] * 2:
            x, sizes = torch.randint(0, 100, (1, length)), torch.tensor([length])
            handles = [r.submit("v1", x, sizes, timeout_ms=10000) for _ in range(8)]
            values = [h.result(15) for h in handles]
            with torch.inference_mode(): expected = model(x.cuda(), sizes.cuda()).cpu()
            for value in values: torch.testing.assert_close(value, expected, rtol=1e-4, atol=1e-5)
            if old is not None: torch.testing.assert_close(old[0], old[1])
            old = values[0], values[0].clone()
        assert r.stats()["reserved_bytes"] == 0
        assert r.slot_count <= 2
