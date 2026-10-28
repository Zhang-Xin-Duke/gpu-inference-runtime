"""Summarize a completed matched replica-count sweep from recorded JSON."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
from statistics import median

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def summarize(root, title):
    groups=defaultdict(list)
    matched=defaultdict(list)
    for path in sorted((root/'sweep').glob('replicas-*.json')):
        data=json.loads(path.read_text())
        if not isinstance(data,dict) or 'config' not in data:continue
        if data.get('run_status')=='interrupted':continue
        config=data['config']
        count=len(config['endpoints'])
        assert sum(data['statuses'].values())==data['received'],path
        assert data['after']['active']==0,path
        assert data['after']['accepted']==data['after']['finished'],path
        assert data['correctness']=='direct/replica parity and sampled outputs passed',path
        matched[config['rps'],config['seed']].append(data['offered'])
        groups[count,config['rps']].append(data)
    if not groups:raise ValueError('no recorded sweep cells')
    if any(len(set(values))!=1 for values in matched.values()):
        raise ValueError('matched replica counts must use identical offered arrivals')
    rows=[]
    for (count,rate),runs in sorted(groups.items()):
        rows.append(dict(replicas=count,offered_rps=rate,runs=len(runs),
            offered=sum(d['offered'] for d in runs),
            on_time=sum(d['statuses'].get('ok',0) for d in runs),
            median_goodput_rps=median(d['goodput_rps'] for d in runs),
            median_p99_ms=median(d['latency']['p99_ms'] for d in runs),
            median_generator_p99_ms=median(d['generator_lag']['p99_ms'] for d in runs)))
    with (root/'summary.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (root/'summary.json').write_text(json.dumps(rows,indent=2)+'\n')
    fig,axes=plt.subplots(1,2,figsize=(11,4.2),layout='constrained')
    colors=['#3267aa','#168575','#b57924']
    rates=sorted({r['offered_rps'] for r in rows})
    for i,count in enumerate(sorted({r['replicas'] for r in rows})):
        selected=[r for r in rows if r['replicas']==count]
        x=[r['offered_rps'] for r in selected]
        for ax,metric in zip(axes,['median_goodput_rps','median_p99_ms']):
            label=f'{count} GPU replica'+('s' if count!=1 else '')
            if len(rates)==1:
                value=selected[0][metric]
                ax.bar(label,value,color=colors[i%len(colors)],width=.65)
                ax.annotate(f'{value:.1f}',(label,value),ha='center',va='bottom')
                ax.set_xlabel(f'Offered load: {rates[0]:g} requests / s')
            else:
                ax.plot(x,[r[metric] for r in selected],marker='o',color=colors[i%len(colors)],label=label)
                ax.set_xlabel('Offered requests / s');ax.legend(frameon=False)
            ax.set_ylim(bottom=0);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    for ax,metric in zip(axes,['median_goodput_rps','median_p99_ms']):
        ax.set_ylim(0,max(r[metric] for r in rows)*1.12)
    axes[0].set_ylabel('On-time completions / s');axes[1].set_ylabel('P99 latency (ms)')
    fig.suptitle(title)
    fig.savefig(root/'replica-sweep.png',dpi=150);fig.savefig(root/'replica-sweep.svg');plt.close(fig)
    print(json.dumps(rows,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--title',default='Recorded GPU replica comparison')
    args=parser.parse_args()
    summarize(args.root,args.title)
