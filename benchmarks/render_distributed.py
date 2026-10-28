"""Render recorded replica telemetry; never synthesize runtime measurements."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    args=p.parse_args()
    data=json.loads(args.input.read_text())
    rows=[r for r in data['observations'] if r['t']>=0]
    workers=len(rows[0]['backend'].get('nodes',[])) if rows else 0
    if len(rows)<3 or workers<2:
        raise ValueError('recorded observations for at least two replicas required')
    if any(len(r['backend'].get('nodes',[]))!=workers for r in rows):
        raise ValueError('replica count must remain constant across observations')
    t=np.array([r['t'] for r in rows])
    ok=np.array([r['outcomes'].get('ok',0) for r in rows])
    rates=np.r_[0,np.diff(ok)/np.maximum(.001,np.diff(t))]
    p99=np.array([r['interval_latency']['p99_ms'] or 0 for r in rows])
    inflight=np.array([[n['outstanding'] for n in r['backend']['nodes']] for r in rows])
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,
                         'axes.spines.top':False,'axes.spines.right':False})
    colors=[plt.get_cmap('tab10')(i%10) for i in range(workers)]
    fig=plt.figure(figsize=(max(10,workers*3.6),7.1),facecolor='#f7f9fb')
    columns=workers*2
    gs=fig.add_gridspec(3,columns,height_ratios=[.85,1,1],hspace=.5,wspace=.28,
                        left=.08,right=.96,bottom=.17,top=.86)
    cards=[fig.add_subplot(gs[0,2*i:2*i+2]) for i in range(workers)]
    throughput=fig.add_subplot(gs[1,:]);latency=fig.add_subplot(gs[2,:workers]);queue=fig.add_subplot(gs[2,workers:])
    left=latency.get_position();right=queue.get_position()
    latency.set_position([left.x0,left.y0,left.width-.025,left.height])
    queue.set_position([right.x0+.025,right.y0,right.width-.025,right.height])
    def draw(frame):
        end=frame+1;row=rows[frame]
        fig.suptitle(data.get('title','Recorded distributed inference run'),x=.08,ha='left',weight='bold',fontsize=16)
        for i,ax in enumerate(cards):
            ax.clear();ax.set_axis_off();n=row['backend']['nodes'][i]
            baseline=data['before']['nodes'][i]
            gpu=data['ready']['replicas'][i].get('gpu') or 'CPU'
            state='Cooldown' if n['cooling_down'] else 'Eligible for routing'
            ax.text(0,.95,f'GPU worker {i+1} · {gpu}',weight='bold',color=colors[i],fontsize=12)
            ax.text(0,.57,f"{n['completed']-baseline['completed']:,} completed · {n['outstanding']} samples in flight")
            ax.text(0,.18,f"{state} · {n['failures']-baseline['failures']} failed RPC attempts",fontsize=9)
        for ax in (throughput,latency,queue):
            ax.clear();ax.set_xlim(0,max(t));ax.grid(alpha=.18);ax.set_axisbelow(True)
            for event in data.get('events',[]):
                ax.axvline(event['t'],color='#b64a45' if event['event']=='stop_replica' else '#168575',ls='--',lw=1)
            ax.axvline(t[frame],color='#8a99aa',lw=.7)
            ax.set_xlabel('Recorded time (seconds)')
        throughput.plot(t[:end],rates[:end],color='#3267aa',lw=2)
        throughput.set_ylim(0,max(10,float(rates.max())*1.15));throughput.set_ylabel('On-time completions / s')
        throughput.set_title(f"t = {t[frame]:.1f}s · offered {data['config']['rps']:g} requests/s · deadline {data['config']['deadline_ms']:g} ms",loc='left',fontsize=10)
        latency.plot(t[:end],p99[:end],color='#926336',lw=2)
        latency.set_ylim(0,max(10,float(p99.max())*1.1));latency.set_ylabel('Window P99 latency (ms)')
        for i,color in enumerate(colors):queue.plot(t[:end],inflight[:end,i],label=f'Worker {i+1}',color=color,lw=2)
        queue.set_ylim(0,max(4,float(inflight.max())*1.1));queue.set_ylabel('Outstanding samples');queue.legend(frameon=False,fontsize=8)
    caption=data.get('topology_caption','Includes router, network and host overhead; RPC activity is not GPU utilization.')
    fig.text(.08,.035,'Playback of measured telemetry. Dashed lines: worker stop / restart request.\n'+caption,fontsize=9,color='#536477')
    frames=list(range(0,len(rows),max(1,len(rows)//60)))
    animation=FuncAnimation(fig,draw,frames=frames,interval=120,repeat=True)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    animation.save(args.output,writer=PillowWriter(fps=8),dpi=100)
    draw(len(rows)-1)
    fig.savefig(args.output.with_suffix('.png'),dpi=130)
    plt.close(fig)


if __name__=='__main__':main()
