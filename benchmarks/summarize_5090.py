"""Validate and plot saved RTX 5090 engineering measurements; never reruns inference."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def trace_summary(path):
    events = json.loads(path.read_text())['traceEvents']
    kernels = [e for e in events if e.get('cat') == 'kernel']
    copies = [e for e in events if e.get('cat') == 'gpu_memcpy']
    # Kernels share one compute stream in this runtime, so these intersections
    # do not double-count overlapping kernels. Report sum, not wall-time speedup.
    overlaps = [max(0., min(c['ts']+c['dur'], k['ts']+k['dur'])-max(c['ts'], k['ts']))
                for c in copies for k in kernels
                if c.get('args', {}).get('stream') != k.get('args', {}).get('stream')]
    return dict(kernels=len(kernels), copies=len(copies),
                streams=sorted({e['args']['stream'] for e in kernels+copies}),
                copied_bytes=sum(e['args'].get('bytes', 0) for e in copies),
                copy_kernel_intersection_us=sum(overlaps),
                positive_intersections=sum(o > 0 for o in overlaps))


def plot_trace(path):
    events=json.loads(path.read_text())['traceEvents']
    gpu=[e for e in events if e.get('cat') in ('kernel','gpu_memcpy')]
    if not gpu:return
    start=min(e['ts'] for e in gpu)
    fig,ax=plt.subplots(figsize=(12,3),layout='constrained')
    lanes={'H2D':(2,'#bd7800'),'Compute':(1,'#2463a6'),'D2H':(0,'#64748b')}
    for e in gpu:
        lane='Compute' if e['cat']=='kernel' else ('H2D' if 'HtoD' in e['name'] else 'D2H')
        y,color=lanes[lane]
        ax.broken_barh([((e['ts']-start)/1000,e['dur']/1000)],(y-.3,.6),facecolors=color)
        if lane=='D2H':ax.plot([(e['ts']-start)/1000],[y],marker='|',markersize=12,color=color)
    ax.set(yticks=[0,1,2],yticklabels=['D2H','Compute','H2D'],xlabel='Profiler time from first GPU event (ms)',
           title='RTX 5090: measured GPU copies and kernels',ylim=(-.6,2.6))
    ax.grid(axis='x',alpha=.2)
    fig.supxlabel('D2H ticks mark copy starts; true copy durations can be below one pixel.',fontsize=9)
    fig.savefig(path.with_suffix('.png'),dpi=160)
    fig.savefig(path.with_suffix('.svg'))
    plt.close(fig)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    a=p.parse_args(); rows=[]
    for f in sorted(a.root.rglob('*.json')):
        if 'trace' in f.name: continue
        d=json.loads(f.read_text())
        if not isinstance(d,dict) or 'offered' not in d or 'config' not in d: continue
        if d['config'].get('endpoints'):continue
        assert d['offered']==sum(d['producer'].get(k,0) for k in ('emitted','ipc_full','generator_stale')), f
        assert d['received']==d['producer']['emitted']==sum(d['statuses'].values()), f
        assert d['statuses'].get('failed',0)==0, f
        assert d['accounting'].startswith('all emitted requests terminal'), f
        assert d['sampled_output_checks']>0, f
        c=d['config']; after=d['after']
        if not c['endpoints']:
            assert after['reserved_bytes']==after['queued']==after['active_requests']==0, f
        rows.append(dict(source=f.relative_to(a.root).as_posix(), kind=c['kind'],mode=c['mode'],
                         seed=c['seed'],duration_s=c['duration'],rps=c['rps'],samples=c['samples'],
                         features=c['features'],hidden=c['hidden'],max_batch=c['max_batch'],slots=c['slots'],
                         offered=d['offered'],on_time=d['statuses'].get('ok',0),
                         goodput_rps=d['goodput_rps'],p99_ms=d['latency']['p99_ms'],
                         generator_lag_p99_ms=d['generator_lag']['p99_ms'],
                         failed=d['statuses'].get('failed',0),
                         peak_observed_cuda_bytes=max((o.get('cuda_allocated_bytes',0) for o in d['observations']),default=0),
                         first_rss_mb=d['observations'][0]['rss_mb'],last_rss_mb=d['observations'][-1]['rss_mb']))
    if not rows: raise ValueError('no extended results')
    with (a.root/'summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    colors={'unbatched':'#64748b','sync':'#bd7800','pipeline':'#2463a6'}
    styles={'unbatched':('o',':'),'sync':('s','--'),'pipeline':('^','-')}
    fig,axs=plt.subplots(1,2,figsize=(12,4.8),sharey=True,layout='constrained')
    for ax,kind in zip(axs,['dense','sequence']):
        for mode in colors:
            group=sorted((r for r in rows if r['source'].startswith(kind+'/load-'+mode+'-')),key=lambda r:r['rps'])
            if group:
                marker,ls=styles[mode]
                ax.plot([r['rps'] for r in group],[r['goodput_rps'] for r in group],label=mode,
                        color=colors[mode],marker=marker,linestyle=ls,linewidth=1.8)
        ax.plot([0,8000],[0,8000],color='#aaa',linewidth=1,linestyle='--',label='offered rate')
        ax.set(title=kind.capitalize(),xlabel='Offered requests / second',xlim=(0,8200),ylim=(0,8400))
        ax.grid(axis='y',alpha=.2)
    axs[0].set_ylabel('On-time requests / second')
    axs[1].legend(loc='upper left')
    fig.suptitle('RTX 5090: deadline goodput across offered load',fontsize=16)
    fig.supxlabel('100 ms deadline · 10 s per cell · seed 7 · independent arrival process · Windows WDDM',fontsize=10)
    fig.savefig(a.root/'load-sweep.png',dpi=160);fig.savefig(a.root/'load-sweep.svg')
    plt.close(fig)
    traces={f.name:trace_summary(f) for f in a.root.glob('*trace.json')}
    for f in a.root.glob('*trace.json'):plot_trace(f)
    (a.root/'trace-summary.json').write_text(json.dumps(traces,indent=2)+'\n')
    print(json.dumps(dict(extended_cells=len(rows),offered=sum(r['offered'] for r in rows),
                         model_failures=sum(r['failed'] for r in rows),traces=traces),indent=2))


if __name__=='__main__':main()
