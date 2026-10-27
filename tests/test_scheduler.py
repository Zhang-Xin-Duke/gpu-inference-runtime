from concurrent.futures import ThreadPoolExecutor
import time
import pytest
from gpu_runtime._scheduler import Scheduler


def scheduler(**kw):
    return Scheduler(**(dict(max_samples=4, max_requests=4, max_bytes=400,
                             max_inflight=2, delay_ms=0, policy="deadline") | kw))


def test_admission_accounts_running_until_completion():
    q = scheduler(max_requests=1)
    a = q.offer("v1", 1, 200, 1000)
    batch = q.take()
    assert batch.ids == [a]
    assert q.cancel(a) == 2
    with pytest.raises(RuntimeError): q.offer("v1", 1, 100, 1000)
    q.finish(batch.ids, 1)
    assert q.stats()["reserved_bytes"] == 0
    q.offer("v1", 1, 100, 1000)
    with pytest.raises(RuntimeError): q.finish(batch.ids, 1)


def test_compatibility_and_sample_limit():
    q = scheduler(max_requests=10)
    a = q.offer("v1:32", 2, 10, 1000)
    b = q.offer("v1:64", 1, 10, 1000)
    c = q.offer("v1:32", 2, 10, 1000)
    batch = q.take()
    assert batch.ids == [a, c] and batch.samples == 4
    q.finish(batch.ids, 1)
    assert q.take().ids == [b]


def test_queued_cancellation_releases_once():
    q = scheduler()
    a = q.offer("v1", 1, 10, 1000)
    assert q.cancel(a) == 1 and q.cancel(a) == 0
    assert q.stats()["reserved_bytes"] == 0


def test_expiration_without_execution_slot():
    q = scheduler(max_inflight=1)
    a = q.offer("v1", 1, 10, 1000)
    q.take()
    b = q.offer("v1", 1, 20, 5)
    time.sleep(.015)
    expired = q.take()
    assert expired.expired == [b] and expired.ids == []
    assert q.stats()["reserved_bytes"] == 10
    q.finish([a], 1)


def test_gil_is_released_and_close_drains():
    q = scheduler(delay_ms=10000)
    a = q.offer("v1", 1, 10, 20000)
    with ThreadPoolExecutor(1) as pool:
        pending = pool.submit(q.take)
        q.close()
        batch = pending.result(timeout=1)
    assert batch.ids == [a]
    q.finish([a], 1)
    assert q.take().stopped
    with pytest.raises(RuntimeError): q.offer("v1", 1, 10, 1000)


def test_deadline_policy_flushes_before_age_policy():
    q = scheduler(delay_ms=500)
    q.offer("v1", 1, 10, 25)
    start = time.monotonic()
    assert q.take().ids
    assert time.monotonic() - start < .2


@pytest.mark.parametrize("samples,bytes_", [(0, 1), (5, 1), (1, 401)])
def test_invalid_capacity(samples, bytes_):
    with pytest.raises(RuntimeError): scheduler().offer("v1", samples, bytes_, 1000)


def test_partial_completion_cannot_release_a_batch():
    q = scheduler()
    a = q.offer("v1", 1, 10, 1000)
    b = q.offer("v1", 1, 10, 1000)
    batch = q.take()
    with pytest.raises(RuntimeError): q.finish([a], 1)
    assert q.stats()["reserved_bytes"] == 20
    q.finish(batch.ids, 1)
    assert q.stats()["active_batches"] == 0


def test_nonfinite_or_overflowing_timeout_is_rejected():
    for timeout in (float("inf"), float("nan"), 1e100):
        with pytest.raises(ValueError): scheduler().offer("v1", 1, 10, timeout)
