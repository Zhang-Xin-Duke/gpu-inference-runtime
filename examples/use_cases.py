"""Small runnable sequence-batching and HTTP inference examples."""
import argparse
import json
import threading
from urllib.request import Request, urlopen

import torch
from gpu_runtime import ModelSpec, Runtime
from gpu_runtime.models import DenseModel, SequenceModel
from gpu_runtime.service import Server


def sequence(device):
    class ObservedSequence(SequenceModel):
        def __init__(self):
            super().__init__(4096,64,32)
            self.batches=[]
        def forward(self,tokens,lengths):
            self.batches.append(list(tokens.shape))
            return super().forward(tokens,lengths)
    model=ObservedSequence()
    inputs=[(torch.randint(0,4096,(1,n)),torch.tensor([n])) for n in (5,7,31)]
    with Runtime(device=device,max_batch=4,delay_ms=100) as runtime:
        runtime.register('sequence-v1',ModelSpec(model,'sequence',64,32))
        with runtime.lock:
            handles=[runtime.submit('sequence-v1',*x,timeout_ms=10000) for x in inputs]
        for label,handle,x in zip('ABC',handles,inputs):
            result=handle.result(15)
            with torch.inference_mode():
                reference=SequenceModel.forward(model,*(t.to(device) for t in x)).cpu()
            torch.testing.assert_close(result,reference,rtol=1e-4,atol=1e-5)
            print(f'Request {label}: length {x[0].shape[1]:2} -> CPU result {list(result.shape)}')
        assert sorted(model.batches)==[[1,32],[2,8]],model.batches
        print('Actual model batches:',model.batches)
        print('Lengths 5 and 7 share bucket 8; length 31 uses bucket 32.')
        print('PASS: all three results match direct model execution.')


def service(device):
    with Runtime(device=device) as runtime:
        runtime.register('dense-v1',ModelSpec(DenseModel(256,512,32),'dense',256,32))
        server=Server(('127.0.0.1',0),runtime)
        thread=threading.Thread(target=server.serve_forever,daemon=True)
        thread.start()
        base=f'http://127.0.0.1:{server.server_port}'
        try:
            with urlopen(base+'/readyz',timeout=5) as response:
                ready=json.load(response)
                assert ready['ready']
                print('GET  /readyz ->',response.status,ready)
            payload=json.dumps(dict(x=[[1.]*256],timeout_ms=10000)).encode()
            with urlopen(Request(base+'/infer',data=payload,headers={'Content-Type':'application/json'}),timeout=15) as response:
                output=torch.tensor(json.load(response)['output'])
                assert list(output.shape)==[1,32]
                print('POST /infer  ->',response.status,'output shape',list(output.shape))
            with torch.inference_mode():
                expected=runtime.models['dense-v1'].model(torch.ones(1,256,device=device)).cpu()
            torch.testing.assert_close(output,expected,rtol=1e-4,atol=1e-5)
            print('PASS: HTTP result matches direct execution; local server closed.')
        finally:
            server.shutdown();server.server_close();thread.join()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case',choices=['sequence','service'],required=True)
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    a=p.parse_args();torch.set_num_threads(1);torch.manual_seed(7)
    (sequence if a.case=='sequence' else service)(a.device)


if __name__=='__main__':main()
