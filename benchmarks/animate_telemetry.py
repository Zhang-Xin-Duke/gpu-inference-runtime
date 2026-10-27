"""Render a time-compressed animation from recorded nvidia-smi observations."""
import argparse
import csv
from datetime import datetime
from io import BytesIO
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--frames',type=int,default=90)
    a=p.parse_args()
    with a.input.open() as f:
        rows=[{k.strip():v.strip() for k,v in r.items()} for r in csv.DictReader(f)]
    if len(rows)<2 or a.frames<2:p.error('at least two observations and frames required')
    stamps=[datetime.strptime(r['timestamp'],'%Y/%m/%d %H:%M:%S.%f') for r in rows]
    times=[(s-stamps[0]).total_seconds()/60 for s in stamps]
    util=[float(r['utilization_gpu_percent']) for r in rows]
    power=[float(r['power_W']) for r in rows]
    temp=[float(r['temperature_C']) for r in rows]
    mem=[float(r['memory_used_MiB'])/1024 for r in rows]
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axs=plt.subplots(2,1,figsize=(8,4.8),sharex=True,layout='constrained')
    fig.suptitle('RTX 5090 · recorded evaluation telemetry',fontsize=15)
    lines=[axs[0].plot([],[],color='#2463a6',linewidth=1.5)[0],
           axs[1].plot([],[],color='#bd7800',linewidth=1.5)[0]]
    axs[0].set(ylabel='GPU utilization (%)',ylim=(0,105))
    axs[1].set(ylabel='GPU power (W)',xlabel='Elapsed minutes (compressed playback)',
               ylim=(0,max(600,max(power)*1.05)),xlim=(0,times[-1]))
    for ax in axs:ax.grid(axis='y',alpha=.2)
    note=axs[0].text(.01,.92,'',transform=axs[0].transAxes,va='top',
                     bbox=dict(facecolor='white',edgecolor='none',alpha=.9))
    images=[]
    for index in np.linspace(1,len(rows),min(a.frames,len(rows)),dtype=int):
        lines[0].set_data(times[:index],util[:index]);lines[1].set_data(times[:index],power[:index])
        note.set_text(f'{times[index-1]:.1f} min | {temp[index-1]:.0f} °C | {mem[index-1]:.1f} GiB used')
        b=BytesIO();fig.savefig(b,format='png',dpi=100);b.seek(0)
        images.append(Image.open(b).convert('P',palette=Image.Palette.ADAPTIVE))
    a.output.parent.mkdir(parents=True,exist_ok=True)
    images[0].save(a.output,save_all=True,append_images=images[1:],duration=150,loop=0)
    fig.savefig(a.output.with_suffix('.png'),dpi=160)
    plt.close(fig)


if __name__=='__main__':main()
