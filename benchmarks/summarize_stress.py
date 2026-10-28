"""Validate and summarize stress_suite.py artifacts without rerunning workloads."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
from statistics import median

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    args=p.parse_args();root=args.root
    groups=defaultdict(list);matched=defaultdict(list)
    repaired=root.parent/'repaired'
    sources=[('feature-load',root/'feature-load'),
             ('replica-load',(repaired if (repaired/'complete.json').exists() else root)/'replica-load')]
    if (repaired/'complete.json').exists():
        sources.append(('replica-before-fix',root/'replica-load'))
    for folder,directory in sources:
        for path in sorted(directory.glob('*.json')):
            if path.name.endswith('.workers.json'):continue
            d=json.loads(path.read_text());c=d['config']
            assert d['offered']==sum(d['producer'].values()),path
            assert sum(d['statuses'].values())==d['received']==d['producer']['emitted'],path
            assert d['accounting']=='all emitted requests terminal; reservations released',path
            if c['endpoints']:
                key=len(c['endpoints'])
                assert d['after']['active']==0 and d['after']['accepted']==d['after']['finished'],path
            else:
                key=c['feature_packing']
                assert d['after']['reserved_bytes']==d['after']['queued']==d['after']['active_requests']==0,path
            groups[folder,key,c['rps']].append(d)
            matched[folder,c['rps'],c['seed']].append(d['offered'])
    assert all(len(v)==2 and len(set(v))==1 for v in matched.values()),'unmatched arrivals'
    rows=[]
    for (category,variant,rate),runs in sorted(groups.items(),key=lambda x:str(x[0])):
        rows.append(dict(category=category,variant=variant,offered_rps=rate,runs=len(runs),
            median_goodput_rps=median(d['goodput_rps'] for d in runs),
            min_goodput_rps=min(d['goodput_rps'] for d in runs),max_goodput_rps=max(d['goodput_rps'] for d in runs),
            median_p95_ms=median(d['latency']['p95_ms'] or 0 for d in runs),
            median_p99_ms=median(d['latency']['p99_ms'] or 0 for d in runs),
            generator_p99_ms=max(d['generator_lag']['p99_ms'] or 0 for d in runs),
            ipc_full=sum(d['producer'].get('ipc_full',0) for d in runs),
            generator_stale=sum(d['producer'].get('generator_stale',0) for d in runs),
            offered=sum(d['offered'] for d in runs),
            statuses={status:sum(d['statuses'].get(status,0) for d in runs)
                      for status in set().union(*(d['statuses'] for d in runs))}))
    recovery={}
    for backend in ('cpu','cuda'):
        d=json.loads((root/f'feature-recovery/{backend}.json').read_text())
        assert len(d['phases'])==3
        assert sum(sum(p['statuses'].values()) for p in d['phases'])==d['received']
        recovery[backend]=dict(phases=d['phases'],phase_offered=d['phase_offered'],
                               after=d['after'],generator_lag=d['generator_lag'])
    soak=json.loads((root/'feature-soak/cuda.json').read_text())
    obs=soak['observations']
    def memory(key):
        values=[o[key] for o in obs if key in o]
        return dict(first_60_samples_median=median(values[:min(60,len(values))]),
                    last_60_samples_median=median(values[-min(60,len(values)):]),peak=max(values)) if values else None
    summary=dict(load=rows,recovery=recovery,
        soak=dict(duration=soak['config']['duration'],offered=soak['offered'],statuses=soak['statuses'],
                  goodput_rps=soak['goodput_rps'],latency=soak['latency'],
                  rss_mb=memory('rss_mb'),cuda_allocated_bytes=memory('cuda_allocated_bytes'),
                  cuda_reserved_bytes=memory('cuda_reserved_bytes'),after=soak['after']))
    supplement=root.parent/'supplement'
    if (supplement/'complete.json').exists():
        supplemental=[]
        for path in sorted(supplement.glob('*.json')):
            if not path.stem.startswith(('knee-','batch-')):continue
            d=json.loads(path.read_text())
            assert sum(d['statuses'].values())==d['received']==d['producer']['emitted']
            assert d['after']['reserved_bytes']==d['after']['queued']==d['after']['active_requests']==0
            supplemental.append(dict(name=path.stem,phase_offered=d['phase_offered'],
                                     phases=d['phases'],generator_lag=d['generator_lag'],after=d['after']))
        assert len(supplemental)==10
        summary['supplemental']=supplemental
    (root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,3,figsize=(16,8),layout='constrained')
    colors={'cpu':'#63758c','cuda':'#168575',1:'#3267aa',2:'#c07a24'}
    for category,variants,ax in [('feature-load',('cpu','cuda'),axes[0,0]),('replica-load',(1,2),axes[0,1])]:
        for variant in variants:
            selected=sorted((r for r in rows if r['category']==category and r['variant']==variant),key=lambda r:r['offered_rps'])
            x=[r['offered_rps'] for r in selected];y=[r['median_goodput_rps'] for r in selected]
            label=(('CPU packing' if variant=='cpu' else 'Fused GPU packing')
                   if category=='feature-load' else f'{variant} GPU replica'+('s' if variant!=1 else ''))
            ax.plot(x,y,'o-',color=colors[variant],label=label)
            ax.fill_between(x,[r['min_goodput_rps'] for r in selected],[r['max_goodput_rps'] for r in selected],color=colors[variant],alpha=.12)
        ax.set(xlabel='Offered requests / s',ylabel='On-time completions / s')
        ax.legend(frameon=False)
    axes[0,0].set_title('Feature packing under open-loop load')
    axes[0,1].set_title('Same-host replica scaling')
    for variant in (1,2):
        selected=sorted((r for r in rows if r['category']=='replica-load' and r['variant']==variant),key=lambda r:r['offered_rps'])
        axes[0,2].plot([r['offered_rps'] for r in selected],[r['median_p99_ms'] for r in selected],'o-',color=colors[variant],label=f'{variant} GPU')
    axes[0,2].axhline(1000,ls='--',color='#ad4a47',label='1,000 ms deadline')
    axes[0,2].set(title='Admitted terminal P99',xlabel='Offered requests / s',ylabel='Milliseconds')
    axes[0,2].legend(frameon=False)
    d=json.loads((root/'feature-recovery/cuda.json').read_text());o=d['observations']
    phases=d['config']['phases'];boundaries=[0]
    for seconds,_ in phases:boundaries.append(boundaries[-1]+seconds)
    axes[1,0].step(boundaries,[r for _,r in phases]+[phases[-1][1]],where='post',color='#ad4a47',label='Offered rate')
    previous=None;times=[];rates=[]
    for point in o:
        if previous:
            dt=point['t']-previous['t']
            if dt>0:
                times.append(point['t']);rates.append((point['outcomes'].get('ok',0)-previous['outcomes'].get('ok',0))/dt)
        previous=point
    axes[1,0].plot(times,rates,color=colors['cuda'],label='On-time completion rate')
    axes[1,0].set(title='Continuous overload and recovery',xlabel='Seconds',ylabel='Requests / s')
    axes[1,0].legend(frameon=False)
    axes[1,1].plot([o['t'] for o in obs],[o['rss_mb'] for o in obs],label='Process RSS',color='#63758c')
    if obs and 'cuda_allocated_bytes' in obs[0]:
        axes[1,1].plot([o['t'] for o in obs],[o['cuda_allocated_bytes']/(1<<20) for o in obs],label='Torch CUDA allocated',color=colors['cuda'])
    axes[1,1].set(title='Five-minute steady-load memory',xlabel='Seconds',ylabel='MiB')
    axes[1,1].legend(frameon=False)
    for backend in ('cpu','cuda'):
        selected=sorted((r for r in rows if r['category']=='feature-load' and r['variant']==backend),key=lambda r:r['offered_rps'])
        axes[1,2].plot([r['offered_rps'] for r in selected],
            [100*r['statuses'].get('rejected',0)/r['offered'] for r in selected],'o-',
            label='CPU packing' if backend=='cpu' else 'Fused GPU packing',color=colors[backend])
    axes[1,2].set(title='Admission rejection under load',xlabel='Offered requests / s',ylabel='Rejected / offered (%)')
    axes[1,2].legend(frameon=False)
    for ax in axes.flat:ax.grid(alpha=.2);ax.set_axisbelow(True);ax.set_ylim(bottom=0)
    fig.suptitle('L40 evaluation: open-loop load, overload recovery and stability\nLoad sweeps: two arrival seeds per configuration; shading shows seed range',fontsize=14)
    fig.savefig(root/'stress-overview.svg');fig.savefig(root/'stress-overview.png',dpi=160);plt.close(fig)
    manifest={str(p.relative_to(root.parent)):hashlib.sha256(p.read_bytes()).hexdigest() for p in root.parent.rglob('*')
              if p.is_file() and p.name!='checksums.json'}
    (root/'checksums.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
