"""Run a recorded replica failure/recovery experiment using trusted operator argv files."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--endpoints',nargs='+',required=True)
    p.add_argument('--stop-argv',type=Path,required=True)
    p.add_argument('--restart-argv',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--duration',type=float,default=60)
    p.add_argument('--rps',type=float,default=300)
    args=p.parse_args()
    events=[]
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        control=Path(tmp)/'start.json'
        proc=subprocess.Popen([sys.executable,'benchmarks/extended.py','--endpoints',*args.endpoints,
            '--duration',str(args.duration),'--rps',str(args.rps),'--deadline-ms','1000',
            '--max-requests','256','--control-file',str(control),'--output',str(args.output)])
        try:
            while not control.exists():
                if proc.poll() is not None:raise RuntimeError('benchmark ended before ready')
                time.sleep(.1)
            begin=json.loads(control.read_text())['begin']
            for at,path,label in [(args.duration/3,args.stop_argv,'stop_replica'),
                                   (args.duration*2/3,args.restart_argv,'restart_replica')]:
                wait=begin+at-time.monotonic()
                if wait>0:time.sleep(wait)
                command=json.loads(path.read_text())
                if not isinstance(command,list) or not all(isinstance(x,str) for x in command):raise ValueError('expected argv list')
                stamp=time.monotonic()
                run=subprocess.run(command,capture_output=True,text=True,timeout=30)
                events.append(dict(t=stamp-begin,event=label,returncode=run.returncode,
                                   command_duration_s=time.monotonic()-stamp))
                if run.returncode:raise RuntimeError(label+' failed')
            if proc.wait(timeout=args.duration+60):raise RuntimeError('benchmark failed')
        finally:
            if proc.poll() is None:proc.kill();proc.wait()
    data=json.loads(args.output.read_text())
    data['events']=events
    data['config'].pop('control_file',None)
    args.output.write_text(json.dumps(data,indent=2)+'\n')


if __name__=='__main__':main()
