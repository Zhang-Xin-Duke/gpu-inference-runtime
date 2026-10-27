"""Record a bounded request burst and optionally render its explanatory GIF."""
import argparse
import json
from pathlib import Path

import torch
from gpu_runtime import AdmissionError, ModelSpec, Runtime
from gpu_runtime.models import DenseModel


def record(device):
    torch.set_num_threads(1)
    torch.manual_seed(7)
    class ObservedModel(DenseModel):
        def __init__(self):
            super().__init__(256,512,32)
            self.forward_shapes=[]
        def forward(self,x):
            self.forward_shapes.append(list(x.shape))
            return super().forward(x)
    model=ObservedModel()
    inputs=[torch.randn(1,256) for _ in range(4)]
    with Runtime(device=device,max_batch=4,max_requests=4,delay_ms=1000) as runtime:
        runtime.register('dense-v1',ModelSpec(model,'dense',256,32))
        # Gate this controlled burst until all admission decisions are captured.
        # The runtime's native scheduler and real model forward remain unchanged.
        with runtime.lock:
            handles=[runtime.submit('dense-v1',x,timeout_ms=10000) for x in inputs]
            try:
                runtime.submit('dense-v1',torch.ones(1,256),timeout_ms=10000)
            except AdmissionError:
                rejected=True
            else:raise AssertionError('the fifth request must exceed the admission limit')
        results=[h.result(15) for h in handles]
        with torch.inference_mode():
            expected=DenseModel.forward(model,torch.cat(inputs).to(device)).cpu()
        torch.testing.assert_close(torch.cat(results),expected,rtol=1e-4,atol=1e-5)
        assert model.forward_shapes==[[4,256]],model.forward_shapes
        stats=runtime.stats()
        assert stats['reserved_bytes']==stats['active_requests']==stats['queued']==0
        return dict(device=device,gpu=torch.cuda.get_device_name(0) if device=='cuda' else None,
                    torch=torch.__version__,seed=7,max_requests=4,max_batch=4,
                    inputs=[dict(label=label,shape=list(x.shape)) for label,x in zip('ABCD',inputs)],
                    forward_shapes=model.forward_shapes,fifth_request_rejected=rejected,
                    outputs=[dict(label=label,shape=list(x.shape),device=str(x.device),
                                  first_three_values=x[0,:3].tolist()) for label,x in zip('ABCD',results)],
                    correctness='direct-model parity passed (rtol=1e-4, atol=1e-5)',
                    stats=stats,timeline=runtime.timelines,
                    playback='Explanatory stages from this recorded run; animation timing is schematic, not a performance trace.')


def render(data,path):
    from PIL import Image, ImageDraw, ImageFont
    def font(size,bold=False):
        candidates=[Path('C:/Windows/Fonts')/('seguisb.ttf' if bold else 'segoeui.ttf'),
                    Path('/usr/share/fonts/truetype/dejavu')/('DejaVuSans-Bold.ttf' if bold else 'DejaVuSans.ttf')]
        for p in candidates:
            if p.exists():return ImageFont.truetype(str(p),size)
        return ImageFont.load_default(size=size)
    blue='#2463a6';ink='#17283b';muted='#526579';line='#cbd5df';red='#a13c39'
    title,body,small=font(31,True),font(20),font(17)
    heading=font(21,True)
    phases=['Four callers submit independent requests',
            'Admit four requests; reject the fifth at the limit',
            'Pack compatible inputs into one GPU batch',
            'Run one real model forward on the GPU',
            'Split the output into four owned CPU results']
    frames=[]
    for phase in range(5):
        for step in range(12):
            image=Image.new('RGB',(1120,630),'#f5f8fc');d=ImageDraw.Draw(image)
            d.text((32,22),'GPU Inference Runtime',font=title,fill=ink)
            d.text((32,70),'4 independent requests  →  1 GPU forward  →  4 individual results',font=body,fill=muted)
            d.rounded_rectangle((1000,24,1088,60),radius=15,fill=blue)
            d.text((1019,28),f'{phase+1} / 5',font=body,fill='white')
            xs=[32,306,580,854]
            titles=['CPU requests','Bounded queue','GPU execution','CPU results']
            active=[0,1,2,2,3][phase]
            for index,(x,label) in enumerate(zip(xs,titles)):
                d.rounded_rectangle((x,142,x+232,478),radius=16,fill='white',outline=blue if index==active else line,width=3 if index==active else 1)
                d.text((x+16,159),label,font=heading,fill=ink)
                d.line((x+16,197,x+216,197),fill=line,width=1)
            for index,label in enumerate('ABCD'):
                y=216+55*index
                d.rounded_rectangle((48,y,248,y+42),radius=8,fill='#e8f0fa')
                d.text((60,y+8),f'{label}   [1, 256]',font=body,fill=blue)
                if phase>=1:
                    d.rounded_rectangle((322,y,522,y+42),radius=8,fill='#e8f0fa')
                    d.text((336,y+8),f'{label}   accepted',font=body,fill=blue)
                if phase>=4:
                    d.rounded_rectangle((870,y,1070,y+42),radius=8,fill='#e8f0fa')
                    d.text((882,y+8),f'{label}   [1, 32]',font=body,fill=blue)
            d.text((49,447),'E   one extra request',font=small,fill=muted)
            d.text((323,447),'Limit: 4 requests',font=small,fill=muted)
            if phase>=2:
                d.rounded_rectangle((601,219,791,335),radius=10,fill='#e8f0fa',outline=blue,width=2)
                for i,label in enumerate('ABCD'):
                    y=224+i*26
                    d.text((615,y),label,font=small,fill=blue)
                    d.line((645,y+12,774,y+12),fill='#7ea8d0',width=3)
                d.text((610,351),'input: [4, 256]',font=body,fill=ink)
                d.text((610,385),'DenseModel',font=heading,fill=blue)
                if phase>=3:d.text((610,421),'output: [4, 32]',font=body,fill=ink)
            else:
                d.text((602,271),'Wait for compatible',font=small,fill=muted)
                d.text((602,299),'requests to batch',font=small,fill=muted)
            for index in range(3):
                x=xs[index]+240;y=310
                color=blue if phase>index else line
                d.line((x,y,x+24,y),fill=color,width=3)
                d.polygon([(x+24,y),(x+16,y-6),(x+16,y+6)],fill=color)
                if phase>index:
                    dot=x+int(step/11*22)
                    d.ellipse((dot-4,y-4,dot+4,y+4),fill=blue)
            if phase>=1:
                d.rounded_rectangle((306,491,538,528),radius=8,fill='#faeae8')
                d.text((320,497),'E → rejected: queue full',font=small,fill=red)
            d.text((32,546),phases[phase],font=heading,fill=ink)
            device=data['gpu'] or 'CPU'
            d.text((32,585),f'Recorded on {device} · playback slowed; stage timing is schematic.',font=small,fill=muted)
            frames.append(image.convert('P',palette=Image.Palette.ADAPTIVE))
    path.parent.mkdir(parents=True,exist_ok=True)
    frames[0].save(path,save_all=True,append_images=frames[1:],duration=150,loop=0)
    frames[-1].convert('RGB').save(path.with_suffix('.png'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    p.add_argument('--output',type=Path,default=Path('results/demo.json'))
    p.add_argument('--render',type=Path,help='optional GIF; requires Pillow')
    a=p.parse_args();data=record(a.device)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(data,indent=2)+'\n')
    if a.render:render(data,a.render)
    print(f"{data['device']}: 4 requests -> {data['forward_shapes']} model forward -> 4 CPU results; fifth rejected; parity passed")


if __name__=='__main__':main()
