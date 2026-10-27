"""Large FP32 inference workloads for a 32 GB GPU; no precision or clock changes."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',required=True,type=Path)
    p.add_argument('--duration',type=float,default=15)
    a=p.parse_args()
    if a.duration<=0:p.error('duration must be positive')
    root=Path(__file__).resolve().parents[1]
    a.output_dir=a.output_dir.resolve();a.output_dir.mkdir(parents=True,exist_ok=True)
    plan=[]
    for mode in ['sync','pipeline']:
        for seed in [7,17,27]:
            name=f'compute-{mode}-{seed}'
            cmd=[sys.executable,'benchmarks/extended.py','--device','cuda','--kind','dense',
                 '--duration',str(a.duration),'--features','32768','--hidden','32768',
                 '--samples','128','--max-batch','128','--mode',mode,
                 '--slots','2' if mode=='pipeline' else '1','--rps','1000',
                 '--seed',str(seed),'--deadline-ms','100','--output',str(a.output_dir/(name+'.json'))]
            plan.append(dict(name=name,command=cmd))
    (a.output_dir/'compute-plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    for cell in plan:
        print('START',cell['name'],flush=True)
        subprocess.run(cell['command'],cwd=root,check=True,timeout=a.duration+240)
    subprocess.run([sys.executable,'benchmarks/profile_cuda.py','--features','32768',
                    '--hidden','32768','--samples','128','--max-batch','128','--requests','6',
                    '--output',str(a.output_dir/'compute-trace.json')],cwd=root,check=True,timeout=240)


if __name__=='__main__':main()
