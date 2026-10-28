"""Load, overload/recovery, stability and replica sweeps on an existing GPU host.

This script never provisions resources. It records commands and partial results.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.request

import torch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--cell-seconds', type=float, default=20)
    p.add_argument('--soak-seconds', type=float, default=300)
    p.add_argument('--seeds', nargs='+', type=int, default=[7,17])
    p.add_argument('--skip-micro', action='store_true')
    p.add_argument('--replicas-only', action='store_true')
    p.add_argument('--phase-seconds', nargs=3, type=float, default=[30,30,60])
    p.add_argument('--device', choices=['cpu','cuda'], default='cuda')
    p.add_argument('--hidden', type=int, default=32768)
    p.add_argument('--feature-rates', nargs='+', type=float, default=[150,450,900,1800])
    p.add_argument('--replica-rates', nargs='+', type=float, default=[200,400,600,1000])
    args = p.parse_args()
    if min(args.cell_seconds,args.soak_seconds) <= 0: p.error('positive durations required')
    out = args.output_dir.resolve(); out.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[1]
    commands=[]; services=[]; telemetry=None
    def save():
        (out/'commands.json').write_text(json.dumps(commands,indent=2)+'\n')
    def run(name, command, env=None, timeout=240):
        target=out/(name+'.json'); target.parent.mkdir(parents=True,exist_ok=True)
        cmd=[sys.executable,*command,'--output',str(target)]
        item=dict(name=name,argv=cmd,cuda_visible_devices=(env or os.environ).get('CUDA_VISIBLE_DEVICES'),start=time.time())
        commands.append(item);save()
        print('START',name,flush=True)
        with target.with_suffix('.log').open('w') as log:
            result=subprocess.run(cmd,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=timeout)
        item.update(returncode=result.returncode,end=time.time());save()
        if result.returncode:raise RuntimeError(f'{name} failed; inspect its log')
        print('DONE',name,flush=True)
    env0=dict(os.environ,CUDA_VISIBLE_DEVICES='0') if args.device=='cuda' else None
    feature=['benchmarks/extended.py','--device',args.device,'--kind','features',
             '--feature-widths','128','128','128','128','128','128','128','128',
             '--shared-features','1','0','1','0','1','0','1','0',
             '--feature-packing','cuda','--samples','32','--max-batch','128',
             '--hidden',str(args.hidden),'--deadline-ms','1000','--max-requests','128',
             '--sample-interval','1','--threads','1']
    manifest=dict(torch=torch.__version__,cuda=torch.version.cuda,cpu_count=os.cpu_count(),
                  gpu_count=torch.cuda.device_count(),config=vars(args)|{'output_dir':str(out)},
                  sources={str(f.relative_to(root)):hashlib.sha256(f.read_bytes()).hexdigest()
                           for folder in ('src','cpp','benchmarks') for f in (root/folder).rglob('*')
                           if f.is_file() and f.suffix in ('.py','.cpp','.cu')})
    (out/'source-and-environment.json').write_text(json.dumps(manifest,indent=2)+'\n')
    try:
        if args.device=='cuda':
            telemetry=subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.used,power.draw,clocks.sm,clocks.mem',
                                        '--format=csv','-l','1'],stdout=(out/'gpu-telemetry.csv').open('w'),stderr=(out/'gpu-telemetry.log').open('w'))
        if not args.skip_micro and not args.replicas_only:
            for seed in args.seeds:
                run(f'micro/seed-{seed}',['benchmarks/feature_pack.py','--suite','extended',
                     '--iterations','20','--rounds','3','--seed',str(seed)],env0,timeout=360)
        # Reverse CPU/fused order on the second seed, preserving arrivals.
        for seed in ([] if args.replicas_only else args.seeds):
            for rate in args.feature_rates:
                order=('cpu','cuda') if args.seeds.index(seed)%2==0 else ('cuda','cpu')
                for backend in order:
                    command=feature.copy();command[command.index('--feature-packing')+1]=backend
                    run(f'feature-load/{backend}-rps-{rate:g}-seed-{seed}',
                        command+['--duration',str(args.cell_seconds),'--rps',str(rate),'--seed',str(seed)],env0)
        for backend in (() if args.replicas_only else ('cpu','cuda')):
            command=feature.copy();command[command.index('--feature-packing')+1]=backend
            phase_args=[f'{duration:g}:{rate}' for duration,rate in zip(args.phase_seconds,(150,1800,150))]
            run(f'feature-recovery/{backend}',command+['--phases',*phase_args,'--seed','47'],env0,timeout=sum(args.phase_seconds)+180)
        if not args.replicas_only:
            run('feature-soak/cuda',feature+['--duration',str(args.soak_seconds),'--rps','150','--seed','53'],
                env0,timeout=args.soak_seconds+180)
        if args.device=='cuda' and torch.cuda.device_count() >= 2:
            endpoints=[]
            for gpu in (0,1):
                port=19380+gpu
                command=[sys.executable,'-m','gpu_runtime.service','--device','cuda',
                         '--port',str(port),'--hidden',str(args.hidden),'--max-batch','128']
                proc=subprocess.Popen(command,cwd=root,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu)),
                                      stdout=(out/f'worker-{gpu}.log').open('w'),stderr=subprocess.STDOUT)
                services.append(proc);endpoints.append(f'http://127.0.0.1:{port}')
            for endpoint in endpoints:
                for _ in range(120):
                    try:
                        with urllib.request.urlopen(endpoint+'/readyz',timeout=2) as response:
                            if json.load(response)['ready']:break
                    except OSError:pass
                    time.sleep(.5)
                else:raise TimeoutError('replica readiness')
            for seed in args.seeds:
                for rate in args.replica_rates:
                    counts=(1,2) if args.seeds.index(seed)%2==0 else (2,1)
                    for count in counts:
                        name=f'replica-load/n-{count}-rps-{rate:g}-seed-{seed}'
                        run(name,['benchmarks/extended.py','--endpoints',*endpoints[:count],
                            '--duration',str(args.cell_seconds),'--rps',str(rate),'--seed',str(seed),
                            '--features','256','--hidden',str(args.hidden),'--samples','32',
                            '--max-batch','128','--deadline-ms','1000','--max-requests','256',
                            '--control-file',str(out/'current-cell.json')])
                        snapshots=[]
                        for endpoint in endpoints:
                            with urllib.request.urlopen(endpoint+'/metrics',timeout=3) as response:
                                snapshots.append(json.load(response))
                        (out/(name+'.workers.json')).write_text(json.dumps(snapshots,indent=2)+'\n')
        (out/'complete.json').write_text(json.dumps(dict(completed=time.time(),cells=len(commands)))+'\n')
    finally:
        for proc in services:
            proc.terminate()
            try:proc.wait(timeout=20)
            except subprocess.TimeoutExpired:proc.kill();proc.wait()
        if telemetry:
            telemetry.terminate();telemetry.wait(timeout=10)
        save()


if __name__=='__main__':main()
