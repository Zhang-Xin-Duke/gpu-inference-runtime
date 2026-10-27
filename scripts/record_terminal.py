"""Capture the real walkthrough through Windows ConPTY and encode its replay.

Optional recording dependencies: pywinpty, pyte, Pillow and an FFmpeg executable.
The cast retains original output/timing; the MP4 renders that session, not a desktop.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def capture(path,device):
    from winpty import Backend, PtyProcess
    root=Path(__file__).resolve().parents[1]
    env=os.environ.copy();env['PYTHONPATH']=str(root/'src');env['PYTHONUNBUFFERED']='1'
    path.parent.mkdir(parents=True,exist_ok=True)
    header=dict(version=2,width=100,height=28,timestamp=int(time.time()),
                title='GPU Inference Runtime - real terminal walkthrough',
                command=f'python examples/terminal_walkthrough.py --device {device}',
                capture='Windows ConPTY; scripted presentation with real child-process output')
    started=time.perf_counter();transcript=[]
    process=PtyProcess.spawn([sys.executable,'examples/terminal_walkthrough.py','--device',device],
                             cwd=str(root),env=env,dimensions=(28,100),backend=Backend.ConPTY)
    with path.open('w',encoding='utf-8') as f:
        f.write(json.dumps(header)+'\n')
        while True:
            try:text=process.read(8192)
            except EOFError:break
            if text:
                transcript.append(text)
                f.write(json.dumps([round(time.perf_counter()-started,4),'o',text])+'\n');f.flush()
            if not process.isalive():
                try:
                    tail=process.read(8192)
                    if tail:transcript.append(tail);f.write(json.dumps([round(time.perf_counter()-started,4),'o',tail])+'\n')
                except EOFError:pass
                break
    if 'ALL USE CASES PASSED' not in ''.join(transcript):
        raise RuntimeError('walkthrough failed; inspect the captured session before publishing')
    print('Captured real terminal session:',path,flush=True)


def render(cast,video,poster,ffmpeg):
    import pyte
    from PIL import Image, ImageDraw, ImageFont
    rows=[json.loads(line) for line in cast.read_text(encoding='utf-8').splitlines()]
    header,events=rows[0],rows[1:]
    duration=events[-1][0]+5
    screen=pyte.Screen(header['width'],header['height']);stream=pyte.Stream(screen)
    base=Path('C:/Windows/Fonts')
    normal=ImageFont.truetype(str(base/'consola.ttf'),22)
    bold=ImageFont.truetype(str(base/'consolab.ttf'),22)
    chrome=ImageFont.truetype(str(base/'consola.ttf'),17)
    cw=normal.getlength('M');line_height=25
    colors=dict(default='#dce7f2',black='#0d1725',red='#ff8b86',green='#a4dda3',
                brown='#f4cd78',yellow='#f4cd78',blue='#8bbcff',magenta='#d8a9ef',
                cyan='#75dbe1',white='#edf4fc',brightblack='#8d9eb0',brightwhite='#ffffff')
    def color(value,default):
        if value=='default':return default
        return colors.get(value,'#'+value if len(value)==6 else default)
    def frame(t):
        image=Image.new('RGB',(1280,800),'#0d1725');d=ImageDraw.Draw(image)
        d.rectangle((0,0,1280,36),fill='#19283a')
        d.text((20,8),'GPU Inference Runtime | real terminal session',font=chrome,fill='#c9d8e8')
        d.text((1080,8),'ConPTY 100x28',font=chrome,fill='#9eb2c9')
        for y in range(header['height']):
            x=0
            while x<header['width']:
                char=screen.buffer[y][x];style=(char.fg,char.bg,char.bold,char.reverse)
                end=x+1
                while end<header['width']:
                    c=screen.buffer[y][end]
                    if (c.fg,c.bg,c.bold,c.reverse)!=style:break
                    end+=1
                text=''.join(screen.buffer[y][i].data for i in range(x,end))
                fg=color(char.fg,'#dce7f2');bg=color(char.bg,'#0d1725')
                if char.reverse:fg,bg=bg,fg
                px,py=30+x*cw,48+y*line_height
                if bg!='#0d1725':d.rectangle((px,py,30+end*cw,py+line_height),fill=bg)
                d.text((px,py),text,font=bold if char.bold else normal,fill=fg)
                x=end
        if not getattr(screen.cursor,'hidden',False) and t%1<.5:
            px,py=30+screen.cursor.x*cw,48+screen.cursor.y*line_height+23
            d.rectangle((px,py,px+cw-1,py+1),fill='#dce7f2')
        d.text((20,777),'Original session timing | real commands and output',font=chrome,fill='#8fa4bb')
        clock=lambda s:f'{int(s)//60:02}:{int(s)%60:02}'
        d.text((1100,777),f'{clock(t)} / {clock(duration)}',font=chrome,fill='#8fa4bb')
        return image
    video.parent.mkdir(parents=True,exist_ok=True)
    encoder=subprocess.Popen([ffmpeg,'-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24',
        '-s','1280x800','-r','12','-i','-','-an','-c:v','libx264','-preset','veryfast',
        '-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(video)],stdin=subprocess.PIPE)
    index=0;poster_saved=False
    try:
        for n in range(int(duration*12)+1):
            t=n/12
            while index<len(events) and events[index][0]<=t:
                stream.feed(events[index][2]);index+=1
            image=frame(t)
            if not poster_saved and 'Actual model batches' in '\n'.join(screen.display):
                poster.parent.mkdir(parents=True,exist_ok=True);image.save(poster);poster_saved=True
            encoder.stdin.write(image.tobytes())
        encoder.stdin.close()
        if encoder.wait(timeout=120):raise RuntimeError('FFmpeg encoding failed')
    finally:
        if encoder.poll() is None:encoder.kill();encoder.wait()
    if not poster_saved:frame(duration).save(poster)
    print(json.dumps(dict(video=str(video),duration_s=round(duration,2),events=len(events),bytes=video.stat().st_size)),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    p.add_argument('--cast',type=Path,default=Path('docs/demo/terminal-walkthrough.cast'))
    p.add_argument('--video',type=Path,default=Path('docs/demo/terminal-walkthrough.mp4'))
    p.add_argument('--poster',type=Path,default=Path('docs/demo/terminal-walkthrough.png'))
    p.add_argument('--ffmpeg',default='ffmpeg')
    p.add_argument('--replay',action='store_true',help='encode an existing cast without rerunning workloads')
    a=p.parse_args()
    if not a.replay:capture(a.cast,a.device)
    render(a.cast,a.video,a.poster,a.ffmpeg)


if __name__=='__main__':main()
