"""Matched local-transport replica tests with recorded worker stage counters.

Start identical GPU services before running. This script never rents GPUs.
"""
import argparse
import json
import os
import signal
from pathlib import Path
import subprocess
import sys
import time
import urllib.request


def snapshot(endpoints):
    result=[]
    for endpoint in endpoints:
        try:
            with urllib.request.urlopen(endpoint+'/metrics',timeout=2) as response:
                result.append(json.load(response))
        except OSError as error:result.append({'error':type(error).__name__})
    return dict(t=time.perf_counter(),workers=result,
                load_average=os.getloadavg() if hasattr(os,'getloadavg') else None)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--endpoints',nargs='+',required=True)
    p.add_argument('--hidden',type=int,default=32768)
    p.add_argument('--samples',type=int,default=32)
    p.add_argument('--rates',nargs='+',type=float,default=[200,600,1000])
    p.add_argument('--seeds',nargs='+',type=int,default=[7,17])
    p.add_argument('--duration',type=float,default=8)
    p.add_argument('--output-dir',type=Path,required=True)
    args=p.parse_args();args.output_dir.mkdir(parents=True,exist_ok=True)
    for seed in args.seeds:
        counts=list(range(1,len(args.endpoints)+1))
        if args.seeds.index(seed)%2:counts.reverse()
        for rate in args.rates:
            for count in counts:
                name=f'replicas-{count}-rps-{rate:g}-seed-{seed}'
                endpoints=args.endpoints[:count]
                output=args.output_dir/(name+'.json')
                control=args.output_dir/(name+'.control.json')
                command=[sys.executable,str(Path(__file__).with_name('extended.py')),
                         '--endpoints',*endpoints,'--features','256','--hidden',str(args.hidden),
                         '--samples',str(args.samples),'--max-batch','128',
                         '--duration',str(args.duration),'--rps',str(rate),'--seed',str(seed),
                         '--deadline-ms','1000','--max-requests','256',
                         '--control-file',str(control),'--output',str(output)]
                print('START',name,flush=True)
                proc=subprocess.Popen(command,start_new_session=os.name!='nt')
                observations=[];started=time.monotonic()
                try:
                    while proc.poll() is None:
                        if time.monotonic()-started>args.duration+120:
                            raise TimeoutError('cell exceeded bounded run and drain time')
                        if control.exists():observations.append(snapshot(endpoints))
                        time.sleep(.5)
                    if proc.returncode:raise RuntimeError(f'{name} failed')
                finally:
                    if proc.poll() is None:
                        if os.name=='nt':
                            subprocess.run(['taskkill','/PID',str(proc.pid),'/T','/F'],
                                           stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
                        else:os.killpg(proc.pid,signal.SIGKILL)
                        proc.wait(timeout=10)
                (args.output_dir/(name+'.workers.json')).write_text(json.dumps(observations,indent=2)+'\n')


if __name__=='__main__':main()
