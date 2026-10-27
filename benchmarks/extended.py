"""Duration-based open-loop benchmark with a separate load-generator process."""
import argparse
import bisect
from collections import Counter
from concurrent.futures import CancelledError
import hashlib
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import queue
if os.name != 'nt':
    import resource
import threading
import time

import numpy as np
import torch
from gpu_runtime import Runtime, ModelSpec, AdmissionError
from gpu_runtime.models import DenseModel, SequenceModel
from gpu_runtime.distributed import Router
from run import hardware


class Histogram:
    # Conservative upper-bin quantiles, at most 1% relative bin width after 1 us.
    edges = [0.] + [0.001 * 1.01**i for i in range(1810)] + [float("inf")]
    def __init__(self): self.counts = [0] * len(self.edges); self.n = 0; self.total = 0.
    def add(self, value):
        value=max(0.,value)
        self.counts[min(bisect.bisect_left(self.edges,value),len(self.counts)-1)]+=1
        self.n+=1; self.total+=value
    def summary(self):
        result=dict(count=self.n, mean_ms=self.total/self.n if self.n else None)
        for p in (50,95,99):
            needed=math.ceil(self.n*p/100)
            count=0; value=None
            for edge,n in zip(self.edges,self.counts):
                count+=n
                if self.n and count>=needed: value=edge;break
            result[f"p{p}_ms"]=value
        return result


def catalog(args):
    torch.manual_seed(args["seed"])
    distributions={"short":[4,8,12,16],"mixed":[5,15,30,60,120,240],
                   "bimodal":[8,8,8,8,256],"long":[192,224,256]}
    items=[]
    for i in range(64):
        if args["kind"]=="dense":items.append((torch.randn(args["samples"],args["features"]),))
        else:
            length=distributions[args["lengths"]][i%len(distributions[args["lengths"]])]
            items.append((torch.randint(0,4096,(args["samples"],length)),torch.full((args["samples"],),length)))
    return items


def rss_mb():
    if os.name == 'nt':
        import ctypes
        from ctypes import wintypes
        class Counters(ctypes.Structure):
            _fields_ = [('cb', wintypes.DWORD), ('PageFaultCount', wintypes.DWORD)] + [
                (name, ctypes.c_size_t) for name in (
                    'PeakWorkingSetSize', 'WorkingSetSize', 'QuotaPeakPagedPoolUsage',
                    'QuotaPagedPoolUsage', 'QuotaPeakNonPagedPoolUsage',
                    'QuotaNonPagedPoolUsage', 'PagefileUsage', 'PeakPagefileUsage')]
        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        psapi = ctypes.WinDLL('psapi', use_last_error=True)
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            raise ctypes.WinError(ctypes.get_last_error())
        return counters.WorkingSetSize / (1 << 20)
    path=Path('/proc/self/statm')
    if path.exists():return int(path.read_text().split()[1])*os.sysconf('SC_PAGE_SIZE')/(1<<20)
    value=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value/(1<<20) if os.uname().sysname=='Darwin' else value/1024


def backend(args, work, control):
    torch.set_num_threads(args["threads"])
    inputs=catalog(args)
    torch.manual_seed(args["model_seed"])
    model=None
    if args["endpoints"]:
        runtime=Router(args["endpoints"],max_pending=args["max_requests"],
                       concurrency=args["rpc_concurrency"],policy=args["routing"],retry=not args["no_retry"])
        def submit(x,timeout):return runtime.submit(*x,timeout_ms=timeout)
    else:
        max_batch=args["samples"] if args["mode"]=='unbatched' else args["max_batch"]
        runtime=Runtime(device=args["device"],max_batch=max_batch,max_requests=args["max_requests"],
                        delay_ms=0 if args["mode"]=='unbatched' else args["delay_ms"],
                        max_inflight=args["slots"],pipeline=args["mode"]=='pipeline',
                        shape_buckets=not args["no_buckets"],policy=args["policy"])
        model=(DenseModel(args["features"],args["hidden"],32) if args["kind"]=='dense'
               else SequenceModel(4096,args["embedding"],32))
        runtime.register('v1',ModelSpec(model,args["kind"],args["features"] if args["kind"]=='dense' else args["embedding"],32))
        def submit(x,timeout):return runtime.submit('v1',*x,timeout_ms=timeout).future
    try:
        for i in range(16):submit(inputs[i%len(inputs)],10000).result(15)
        references={}
        for i in (0,1,2,3):references[i]=submit(inputs[i],10000).result(15)
        if args["endpoints"]:
            for endpoint in args["endpoints"]:
                with Router([endpoint],concurrency=1) as probe:
                    for i,expected in references.items():
                        torch.testing.assert_close(probe.submit(*inputs[i],timeout_ms=10000).result(15),expected,rtol=1e-4,atol=1e-5)
        else:
            with torch.inference_mode():
                for i,expected in references.items():
                    direct=model(*(x.to(args["device"]) for x in inputs[i])).cpu()
                    torch.testing.assert_close(expected,direct,rtol=1e-4,atol=1e-5)
        before=runtime.stats()
        control.send(dict(ready=True, hardware=hardware(args["device"]),
                          metadata=runtime.metadata if args["endpoints"] else None,
                          replicas=runtime.hardware if args["endpoints"] else None))
        tag,begin=work.get(timeout=30)
        assert tag=='start'
        counts=Counter(); latency=Histogram(); transport=Histogram(); admission=Histogram()
        lock=threading.Lock(); stop=threading.Event(); observations=[]; received=0
        sampled=[]; interval=[Histogram()]
        def observe():
            while not stop.is_set():
                row=dict(t=time.perf_counter()-begin,rss_mb=rss_mb(),backend=runtime.stats())
                with lock:
                    row.update(outcomes=dict(counts),interval_latency=interval[0].summary())
                    interval[0]=Histogram()
                if not args["endpoints"] and args["device"]=='cuda':
                    row.update(cuda_allocated_bytes=torch.cuda.memory_allocated(),
                               cuda_reserved_bytes=torch.cuda.memory_reserved())
                observations.append(row)
                stop.wait(args["sample_interval"])
        monitor=threading.Thread(target=observe,daemon=True);monitor.start()
        def done(future,arrival,index):
            stamp=time.perf_counter()
            try:
                value=future.result()
                status='ok' if stamp-arrival<=args["deadline_ms"]/1000 else 'late_completion'
                if index in references:
                    with lock:
                        if len(sampled)<32:sampled.append((index,value))
            except CancelledError:status='cancelled'
            except TimeoutError:status='expired'
            except Exception:status='failed'
            with lock:
                counts[status]+=1
                latency.add((stamp-arrival)*1000)
                interval[0].add((stamp-arrival)*1000)
        while True:
            item=work.get(timeout=max(30,args["duration"]+10))
            if item is None:break
            rid,index,arrival,sent=item;received+=1
            now=time.perf_counter()
            with lock:transport.add((now-sent)*1000)
            remaining=args["deadline_ms"]-(now-arrival)*1000
            if remaining<=0:
                with lock:counts['transport_expired']+=1
                continue
            start=time.perf_counter()
            try:
                future=submit(inputs[index],remaining)
                with lock:admission.add((time.perf_counter()-start)*1000)
                future.add_done_callback(lambda f,a=arrival,i=index:done(f,a,i))
            except AdmissionError:
                with lock:counts['rejected']+=1
            except TimeoutError:
                with lock:counts['expired_at_admission']+=1
        drain_start=time.perf_counter()
        runtime.shutdown()
        drained=time.perf_counter()
        stop.set();monitor.join(timeout=10)
        after=runtime.stats()
        assert sum(counts.values())==received,(counts,received)
        if args["endpoints"]:assert after['active']==0 and after['finished']==after['accepted']
        else:assert after['reserved_bytes']==after['queued']==after['active_requests']==0
        for i,value in sampled:torch.testing.assert_close(value,references[i],rtol=1e-4,atol=1e-5)
        control.send(dict(received=received,statuses=dict(counts),latency=latency.summary(),
                          transport_delay=transport.summary(),admission_time=admission.summary(),
                          goodput_rps=counts['ok']/args['duration'],drain_s=drained-drain_start,
                          before=before,after=after,observations=observations,
                          correctness='direct/replica parity and sampled outputs passed',
                          accounting='all emitted requests terminal; reservations released',
                          sampled_output_checks=len(sampled)))
    except BaseException as exc:
        try:control.send(dict(error=repr(exc)))
        except Exception:pass
        runtime.shutdown()
        raise


def execute(args):
    ctx=mp.get_context('spawn')
    work=ctx.Queue(maxsize=args.ipc_capacity)
    parent,child=ctx.Pipe(duplex=False)
    process=ctx.Process(target=backend,args=(vars(args),work,child))
    process.start();child.close()
    if not parent.poll(180):process.kill();raise TimeoutError('backend startup timeout')
    ready=parent.recv()
    if not ready.get('ready'):raise RuntimeError(ready)
    begin=time.perf_counter()+.1
    work.put(('start',begin))
    if args.control_file:
        Path(args.control_file).write_text(json.dumps(dict(begin=begin,wall_time=time.time())))
    rng=np.random.default_rng(args.seed)
    target=0.;rid=0;counts=Counter();lag=Histogram()
    try:
        while True:
            target+=float(rng.exponential(1/args.rps))
            if target>=args.duration:break
            offset=(int(target)+target%1/10) if args.burst else target
            arrival=begin+offset
            remaining=arrival-time.perf_counter()
            if remaining>0:time.sleep(remaining)
            sent=time.perf_counter();lag.add((sent-arrival)*1000)
            rid+=1
            if sent-arrival>args.deadline_ms/1000:counts['generator_stale']+=1;continue
            try:work.put_nowait((rid,rid%64,arrival,sent));counts['emitted']+=1
            except queue.Full:counts['ipc_full']+=1
        remaining=begin+args.duration-time.perf_counter()
        if remaining>0:time.sleep(remaining)
        work.put(None,timeout=30)
        if not parent.poll(args.deadline_ms/1000+60):raise TimeoutError('backend drain timeout')
        result=parent.recv()
        process.join(timeout=20)
        if process.exitcode!=0 or 'error' in result:raise RuntimeError(result)
        assert result['received']==counts['emitted']
        assert rid==counts['emitted']+counts['ipc_full']+counts['generator_stale']
        return dict(config=vars(args),ready=ready,offered=rid,producer=dict(counts),
                    generator_lag=lag.summary(),**result)
    finally:
        if process.is_alive():process.kill();process.join()
        work.cancel_join_thread();work.close();parent.close()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    p.add_argument('--endpoints',nargs='*',default=[])
    p.add_argument('--duration',type=float,default=20)
    p.add_argument('--rps',type=float,default=1000)
    p.add_argument('--deadline-ms',type=float,default=100)
    p.add_argument('--kind',choices=['dense','sequence'],default='dense')
    p.add_argument('--mode',choices=['unbatched','sync','pipeline'],default='pipeline')
    p.add_argument('--max-batch',type=int,default=64)
    p.add_argument('--slots',type=int,default=2)
    p.add_argument('--delay-ms',type=float,default=2)
    p.add_argument('--max-requests',type=int,default=1024)
    p.add_argument('--ipc-capacity',type=int,default=4096)
    p.add_argument('--features',type=int,default=256)
    p.add_argument('--hidden',type=int,default=512)
    p.add_argument('--embedding',type=int,default=64)
    p.add_argument('--samples',type=int,default=1)
    p.add_argument('--lengths',choices=['short','mixed','bimodal','long'],default='mixed')
    p.add_argument('--no-buckets',action='store_true')
    p.add_argument('--policy',choices=['age','deadline'],default='deadline')
    p.add_argument('--burst',action='store_true')
    p.add_argument('--routing',choices=['least_work','round_robin'],default='least_work')
    p.add_argument('--no-retry',action='store_true')
    p.add_argument('--rpc-concurrency',type=int,default=32)
    p.add_argument('--sample-interval',type=float,default=1)
    p.add_argument('--seed',type=int,default=7)
    p.add_argument('--model-seed',type=int,default=7)
    p.add_argument('--threads',type=int,default=1)
    p.add_argument('--control-file')
    p.add_argument('--output',required=True)
    args=p.parse_args()
    if args.duration<=0 or args.rps<=0 or not 0<args.deadline_ms<=10000:p.error('invalid time/rate limits')
    result=execute(args)
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(output=str(output),offered=result['offered'],goodput_rps=result['goodput_rps'],
                         statuses=result['statuses'],latency=result['latency'],generator_lag=result['generator_lag'])),flush=True)


if __name__=='__main__':main()
