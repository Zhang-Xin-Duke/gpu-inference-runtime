"""Longer single-GPU sweeps; run one shard per GPU host."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

p=argparse.ArgumentParser()
p.add_argument('--kind',choices=['dense','sequence'],required=True)
p.add_argument('--output-dir',required=True,type=Path)
p.add_argument('--cell-seconds',type=float,default=15)
p.add_argument('--soak-seconds',type=float,default=300)
p.add_argument('--device',default='cuda')
a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=True)
plan=[]
for mode in ['unbatched','sync','pipeline']:
    for rps in [100,500,1000,2000,4000,8000]:
        plan.append((f'load-{mode}-{rps}',a.cell_seconds,['--mode',mode,'--rps',str(rps)]))
for size in [8,16,32,64]:
    for slots in [1,2,4]:
        plan.append((f'batch-{size}-slots-{slots}',10,['--max-batch',str(size),'--slots',str(slots),'--rps','1000']))
if a.kind=='sequence':
    for distribution in ['short','mixed','bimodal','long']:
        for buckets in [True,False]:
            plan.append((f'lengths-{distribution}-buckets-{buckets}',a.cell_seconds,
                         ['--lengths',distribution,'--rps','1000']+([] if buckets else ['--no-buckets'])))
else:
    for features,hidden in [(256,512),(2048,2048),(4096,2048)]:
        for samples in [1,4,8]:
            plan.append((f'model-{features}-{hidden}-samples-{samples}',a.cell_seconds,
                         ['--features',str(features),'--hidden',str(hidden),'--samples',str(samples),'--rps','1000']))
plan.append(('soak',a.soak_seconds,['--rps','1000','--max-requests','1024']))
(a.output_dir/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
for name,duration,extra in plan:
    output=a.output_dir/(name+'.json')
    if output.exists():continue
    cmd=[sys.executable,'benchmarks/extended.py','--device',a.device,'--kind',a.kind,
         '--duration',str(duration),'--deadline-ms','100','--output',str(output),*extra]
    print('START',name,flush=True)
    started=time.monotonic()
    subprocess.run(cmd,check=True,timeout=duration+240)
    print('END',name,'wall_s',round(time.monotonic()-started,2),flush=True)
files=sorted(p for p in a.output_dir.rglob('*') if p.is_file() and p.name!='SHA256SUMS')
(a.output_dir/'SHA256SUMS').write_text(''.join(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+p.name+'\n' for p in files))
