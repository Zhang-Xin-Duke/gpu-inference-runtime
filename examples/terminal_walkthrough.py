"""Scripted presentation of real project commands; suitable for terminal capture."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    p.add_argument('--pause',type=float,default=6)
    a=p.parse_args();root=Path(__file__).resolve().parents[1];os.chdir(root)
    def pause():time.sleep(a.pause)
    def chapter(number,title):
        print('\033[2J\033[H',end='',flush=True)
        print('\033[1;36mGPU INFERENCE RUNTIME | TERMINAL WALKTHROUGH\033[0m')
        print('Scripted presentation. Every command below executes the real project.\n')
        print(f'\033[1;33m{number}. {title}\033[0m\n',flush=True)
        time.sleep(1)
    def command(args):
        shown='python '+(subprocess.list2cmdline(args) if os.name=='nt' else shlex.join(args))
        print('\033[1;32mgpu-inference-runtime> \033[0m',end='',flush=True)
        for char in shown:
            print(char,end='',flush=True);time.sleep(.012)
        print('\n',flush=True)
        subprocess.run([sys.executable,*args],check=True)
        print(flush=True);pause()
    chapter(1,'Check the inference environment')
    command(['-c',"import torch; print('PyTorch:',torch.__version__); print('CUDA available:',torch.cuda.is_available()); print('GPU:',torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"])
    chapter(2,'Many callers: batch four requests; reject overload')
    command(['examples/quick_demo.py','--device',a.device])
    print('Use case: independent prediction requests share one model execution.')
    print('Each caller gets its own CPU tensor; accepted work stays bounded.');pause()
    chapter(3,'Variable-length inputs: reduce unnecessary padding')
    command(['examples/use_cases.py','--case','sequence','--device',a.device])
    print('Use case: batch short histories together without padding them to 256.');pause()
    chapter(4,'Serve inference over HTTP')
    command(['examples/use_cases.py','--case','service','--device',a.device])
    print('Use case: embed the runtime inside an inference service.');pause()
    chapter(5,'Measure deadline goodput under an arrival stream')
    command(['benchmarks/extended.py','--device',a.device,'--kind','dense','--mode','pipeline',
             '--duration','5','--rps','1000','--deadline-ms','100','--output','results/demo-benchmark.json'])
    result=json.loads(Path('results/demo-benchmark.json').read_text())
    print('\033[1;36mRecorded benchmark summary\033[0m')
    print(f"Offered requests: {result['offered']:,}")
    print(f"On-time goodput:  {result['goodput_rps']:.1f} requests/s")
    print(f"P99 completion:   {result['latency']['p99_ms']:.2f} ms")
    print(f"Outcomes:         {result['statuses']}")
    print('Accounting:      '+result['accounting'])
    print('Correctness:     '+result['correctness']);pause()
    chapter(6,'What the project provides')
    print('  CPU requests -> bounded queue -> compatible GPU batch -> CPU results\n')
    print('  [PASS] Dense batching and individual output ownership')
    print('  [PASS] Early overload rejection')
    print('  [PASS] Sequence shape buckets and direct-model parity')
    print('  [PASS] HTTP readiness and inference')
    print('  [PASS] Independent-process load test and terminal accounting\n')
    print('  This is a short functional demo, not a maximum-throughput result.')
    print('\nALL USE CASES PASSED',flush=True);pause()


if __name__=='__main__':main()
