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
        result=router.submit(torch.ones(1,8),timeout_ms=5000).result(6)
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


def test_restart_invalidates_other_rpc_threads_stale_sockets(monkeypatch):
    import http.client
    generation=[0]
    warming=[True]
    barrier=threading.Barrier(2)
    body=wire.encode((torch.ones(1,2),))
    class Response:
        status=200
        def read(self,*_):return body
    class Connection:
        sock=None
        def __init__(self):self.generation=generation[0]
        def request(self,*_):pass
        def close(self):pass
        def getresponse(self):
            if warming[0]:barrier.wait(timeout=5)
            if self.generation!=generation[0]:
                raise http.client.RemoteDisconnected('worker restarted')
            return Response()
    monkeypatch.setattr(Router,'_get',lambda *_:dict(model_version='v1',outputs=2))
    monkeypatch.setattr(Router,'_connection',staticmethod(lambda *_:Connection()))
    router=Router(['http://replica.test'],concurrency=2,retry=False,cooldown_s=0)
    release_a,release_b=threading.Event(),threading.Event()
    entered_a,entered_b=threading.Event(),threading.Event()
    def block(entered,release):
        entered.set()
        assert release.wait(timeout=10)
    try:
        warm=[router.submit(torch.ones(1,8),timeout_ms=5000) for _ in range(2)]
        for future in warm:future.result(6)
        warming[0]=False
        blocked_a=router.pool.submit(block,entered_a,release_a)
        assert entered_a.wait(5)
        generation[0]+=1
        # One thread discovers the restart while the other still caches its
        # old connection. That second thread must reconnect on its next call.
        with pytest.raises(http.client.RemoteDisconnected):
            router.submit(torch.ones(1,8),timeout_ms=5000).result(6)
        blocked_b=router.pool.submit(block,entered_b,release_b)
        assert entered_b.wait(5)
        release_a.set();blocked_a.result(5)
        result=router.submit(torch.ones(1,8),timeout_ms=5000).result(6)
        torch.testing.assert_close(result,torch.ones(1,2))
        assert router.stats()['nodes'][0]['failures']==1
    finally:
        release_a.set();release_b.set();router.shutdown()


def test_replica_connections_stay_bounded_when_rpc_threads_change_nodes(monkeypatch):
    # A thread-local cache can leave every RPC thread holding an idle socket
    # to every replica, consuming the service's bounded connection permits.
    barrier=threading.Barrier(4)
    lock=threading.Lock()
    live={'http://one.test':0,'http://two.test':0}
    peak=dict(live)
    body=wire.encode((torch.ones(1,2),))
    class Response:
        status=200
        def read(self,*_):return body
    class Connection:
        sock=None
        def __init__(self,url):
            self.url=url;self.closed=False
            with lock:
                live[url]+=1;peak[url]=max(peak[url],live[url])
        def request(self,*_):pass
        def getresponse(self):barrier.wait(timeout=5);return Response()
        def close(self):
            with lock:
                if not self.closed:live[self.url]-=1;self.closed=True
    monkeypatch.setattr(Router,'_get',lambda *_:dict(model_version='v1',outputs=2))
    monkeypatch.setattr(Router,'_connection',staticmethod(lambda node,*_:Connection(node.url)))
    router=Router(list(live),concurrency=4,retry=False)
    router.per_node_limit=2
    phase=[0];threads={}
    def choose(excluded,samples):
        # Keep two requests per replica, but deliberately rotate thread
        # ownership across replicas between waves.
        with router.lock:
            tid=threading.get_ident()
            if tid not in threads:threads[tid]=len(threads)
            selected=(threads[tid]+phase[0])%2
            node=router.nodes[selected]
            assert node.active<2
            node.active+=1;node.outstanding+=samples;node.requests+=1
            return selected
    monkeypatch.setattr(router,'_choose',choose)
    try:
        for wave in range(3):
            phase[0]=wave
            futures=[router.submit(torch.ones(1,8),timeout_ms=5000) for _ in range(4)]
            for future in futures:future.result(6)
        assert all(value<=2 for value in peak.values()),peak
    finally:router.shutdown()
    assert all(value==0 for value in live.values())


def test_idle_peer_close_reconnects_without_cooling_healthy_replica(monkeypatch):
    import http.client
    body=wire.encode((torch.ones(1,2),))
    created=[]
    class Response:
        status=200
        def read(self,*_):return body
    class Connection:
        sock=None
        def __init__(self):self.calls=0;created.append(self)
        def request(self,*_):self.calls+=1
        def getresponse(self):
            if self.calls>1:raise http.client.RemoteDisconnected('idle peer closed')
            return Response()
        def close(self):pass
    monkeypatch.setattr(Router,'_get',lambda *_:dict(model_version='v1',outputs=2))
    monkeypatch.setattr(Router,'_connection',staticmethod(lambda *_:Connection()))
    with Router(['http://healthy.test'],concurrency=1,cooldown_s=2) as router:
        for _ in range(3):
            assert router.submit(torch.ones(1,8),timeout_ms=1000).result(2).shape==(1,2)
        stats=router.stats()
        assert not stats['nodes'][0]['cooling_down']
        assert stats['nodes'][0]['stale_connections']==2
        assert stats['active']==0 and stats['finished']==3


def test_idle_pool_retires_sockets_before_server_idle_timeout(replicas):
    url=f'http://127.0.0.1:{replicas[0][1].server_port}'
    with Router([url],concurrency=1,keepalive_s=.01) as router:
        router.submit(torch.ones(1,8),timeout_ms=1000).result(2)
        time.sleep(.03)
        router.submit(torch.ones(1,8),timeout_ms=1000).result(2)
        stats=router.stats()
        assert stats['nodes'][0]['retired_idle_connections']==1
        assert stats['nodes'][0]['failures']==0


def test_real_server_idle_close_does_not_disable_healthy_worker(replicas,monkeypatch):
    from gpu_runtime.service import Handler
    original=Handler.setup
    def setup(handler):
        original(handler)
        handler.connection.settimeout(.05)
    monkeypatch.setattr(Handler,'setup',setup)
    url=f'http://127.0.0.1:{replicas[0][1].server_port}'
    with Router([url],concurrency=1,keepalive_s=4,cooldown_s=2) as router:
        router.submit(torch.ones(1,8),timeout_ms=1000).result(2)
        time.sleep(.1)
        assert router.submit(torch.ones(1,8),timeout_ms=1000).result(2).shape==(1,2)
        node=router.stats()['nodes'][0]
        assert node['stale_connections']==1 and not node['cooling_down']
