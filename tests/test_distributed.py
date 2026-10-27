import threading
import time
import pytest
import torch
from gpu_runtime import ModelSpec, Runtime
from gpu_runtime.distributed import Router
from gpu_runtime.models import DenseModel
from gpu_runtime.service import Server
from gpu_runtime import wire


@pytest.fixture
def replicas():
    items=[]
    for _ in range(2):
        torch.manual_seed(7)
        r=Runtime(delay_ms=0)
        r.register("dense-v1",ModelSpec(DenseModel(8,16,2),"dense",8,2))
        server=Server(("127.0.0.1",0),r)
        thread=threading.Thread(target=server.serve_forever,daemon=True)
        thread.start()
        items.append((r,server,thread))
    yield items
    for r,s,t in items:
        s.shutdown();r.shutdown();s.server_close();t.join()


def test_wire_roundtrip_and_malformed():
    inputs=(torch.randn(2,8),torch.tensor([2,3]))
    body=wire.encode(inputs,model_version="test",timeout_ms=10)
    decoded,header=wire.decode(body)
    for a,b in zip(inputs,decoded):torch.testing.assert_close(a,b)
    assert header["model_version"]=="test"
    for bad in (b"",body[:3],body[:-1],body+b"extra",b"\xff\xff\xff\xff{}"):
        with pytest.raises(ValueError):wire.decode(bad)


def test_two_replicas_result_parity_and_accounting(replicas):
    urls=[f"http://127.0.0.1:{s.server_port}" for _,s,_ in replicas]
    x=torch.randn(2,8)
    with Router(urls,concurrency=8) as router:
        futures=[router.submit(x,timeout_ms=5000) for _ in range(40)]
        with torch.inference_mode(): expected=replicas[0][0].models["dense-v1"].model(x)
        for f in futures:torch.testing.assert_close(f.result(10),expected)
        stats=router.stats()
        assert stats["active"]==0 and stats["finished"]==40
        assert all(n["completed"]>0 and n["outstanding"]==0 for n in stats["nodes"])


def test_connection_failure_retries_another_replica(replicas):
    urls=[f"http://127.0.0.1:{s.server_port}" for _,s,_ in replicas]
    router=Router(urls,concurrency=1)
    # Close existing listening socket before the first inference RPC.
    replicas[0][1].shutdown();replicas[0][1].server_close()
    try:
        result=router.submit(torch.ones(1,8),timeout_ms=2000).result(3)
        assert result.shape==(1,2)
        assert router.stats()["nodes"][0]["failures"]==1
        assert router.stats()["nodes"][1]["completed"]==1
    finally:router.shutdown()


def test_fingerprint_mismatch(replicas):
    replicas[1][1].metadata["fingerprint"]="different"
    urls=[f"http://127.0.0.1:{s.server_port}" for _,s,_ in replicas]
    with pytest.raises(ValueError,match="fingerprints"):Router(urls)


def test_cancelled_router_requests_release_all_permits(replicas):
    urls=[f"http://127.0.0.1:{s.server_port}" for _,s,_ in replicas]
    router=Router(urls,concurrency=1,max_pending=64)
    futures=[router.submit(torch.ones(1,8),timeout_ms=5000) for _ in range(50)]
    cancelled=sum(f.cancel() for f in futures)
    router.shutdown()
    stats=router.stats()
    assert cancelled>0
    assert stats["active"]==0 and stats["accepted"]==stats["finished"]==50


def test_binary_protocol_bounds_and_identity(replicas):
    import http.client
    server=replicas[0][1]
    conn=http.client.HTTPConnection('127.0.0.1',server.server_port)
    try:
        body=wire.encode((torch.ones(1,8),),model_version='wrong',timeout_ms=100)
        conn.request('POST','/infer-binary',body,{'Content-Type':'application/octet-stream'})
        response=conn.getresponse();response.read()
        assert response.status==400
        conn.request('POST','/infer-binary',b'bad')
        response=conn.getresponse();response.read()
        assert response.status==400
    finally:conn.close()
