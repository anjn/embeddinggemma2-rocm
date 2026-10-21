"""Isolated GPU kernel screening worker; checkpoints every individual candidate."""
from __future__ import annotations
import ctypes as C,json,hashlib,subprocess,statistics
from pathlib import Path
import numpy as np
from ..preparation import write_json


def run(job):
    import torch
    torch.set_num_threads(4);torch.manual_seed(812)
    family=job['family'];rows=job['vision_batch']*job['patches']
    out=Path(job['output']);root=out.parent
    records=json.loads(out.read_text()) if out.exists() else []
    source=Path(__file__).parent/'kernels'/dict(gelu='gelu_layout.cpp',qk='qk_rope_packed.cpp',norm='norm_patch.cpp',preprocess='preprocess.cpp')[family]
    library=root/(family+'.so')
    if not library.exists():
        subprocess.run(['/opt/rocm/bin/hipcc','-O3','-shared','-fPIC','--offload-arch='+job['arch'],
                        '-DROWS='+str(rows),str(source),'-o',str(library)],check=True,timeout=300)
    dll=C.CDLL(str(library.resolve()))
    stream=C.c_void_p(torch.cuda.current_stream().cuda_stream)
    def pointer(t):return C.c_void_p(t.data_ptr())
    if family=='gelu':
        x=torch.randn(rows,6144,device='cuda',dtype=torch.bfloat16)
        y=torch.empty(rows,3072,device='cuda',dtype=torch.bfloat16)
        reference=(torch.nn.functional.gelu(x[:,:3072].float(),approximate='tanh')*x[:,3072:].float()).bfloat16()
        fn=dll.test_gelu_layout;fn.argtypes=[C.c_void_p]*3+[C.c_int]*3+[C.c_void_p]
        def launch(c):return fn(pointer(x),pointer(x[:,3072:]),pointer(y),c['vector'],0,c['block'],stream)
        outputs=[y];refs=[reference]
    elif family=='qk':
        x=torch.randn(rows,2304,device='cuda',dtype=torch.bfloat16)
        w=torch.randn(64,device='cuda',dtype=torch.float32)*.1+1
        angle=torch.randn(rows,64,device='cuda');co=angle.cos().bfloat16();si=angle.sin().bfloat16()
        y=torch.empty(rows,12,64,device='cuda',dtype=torch.bfloat16)
        v=x[:,:768].reshape(rows,12,64).float()
        v=(v/torch.sqrt((v*v).mean(-1,keepdim=True)+1e-6)*w).bfloat16()
        paired=v.reshape(rows,12,2,32)
        rotated=torch.cat((-paired[...,16:],paired[...,:16]),-1).reshape_as(v)
        reference=(v*co[:,None,:]+rotated*si[:,None,:]).bfloat16()
        plain=dll.test_layout_block;plain.argtypes=[C.c_void_p]*5+[C.c_int]*2+[C.c_void_p]
        packed=dll.test_packed;packed.argtypes=[C.c_void_p]*5+[C.c_int,C.c_void_p]
        def launch(c):
            args=[pointer(t) for t in (x,w,co,si,y)]
            return packed(*args,c['block'],stream) if c['packed'] else plain(*args,c['lanes'],c['block'],stream)
        outputs=[y];refs=[reference]
    elif family=='norm':
        x=torch.randn(rows,768,device='cuda',dtype=torch.bfloat16)
        residual=torch.randn_like(x);w=torch.randn(768,device='cuda')*.1+1
        y=torch.empty_like(x);z=torch.empty_like(x)
        f=x.float();first=((f*torch.rsqrt((f*f).mean(-1,keepdim=True)+1e-6)*w).bfloat16()+residual).bfloat16()
        f=first.float();second=(f*torch.rsqrt((f*f).mean(-1,keepdim=True)+1e-6)).bfloat16()
        fn=dll.run_pair_layout;fn.argtypes=[C.c_void_p]*5+[C.c_int]*2+[C.c_void_p]
        def launch(c):return fn(*[pointer(t) for t in (x,w,residual,y,z)],c['lanes'],c['block'],stream)
        outputs=[y,z];refs=[first,second]
    else:
        gh,gw=job['grid'];b=job['vision_batch'];w,h=job['input_size']
        x=torch.randint(0,256,(b,h,w,3),device='cuda',dtype=torch.uint8)
        y=torch.empty(b,gh*gw,768,device='cuda')
        resized=torch.nn.functional.interpolate(x.permute(0,3,1,2).float(),size=(gh*16,gw*16),mode='bilinear',align_corners=False)/255
        reference=resized.permute(0,2,3,1).reshape(b,gh,16,gw,16,3).permute(0,1,3,2,4,5).reshape_as(y)
        fn=dll.eg2_patchify_config;fn.argtypes=[C.c_void_p]*2+[C.c_int]*5+[C.c_void_p]+[C.c_int]*2
        def launch(c):return fn(pointer(x),pointer(y),b,h,w,gh,gw,stream,c['block'],c['vector'])
        outputs=[y];refs=[reference]
    done={json.dumps(r['config'],sort_keys=True) for r in records}
    for config in job['candidates']:
        if json.dumps(config,sort_keys=True) in done:continue
        record=dict(config=config,status='running')
        records.append(record);write_json(out,records)
        try:
            for _ in range(3):
                if launch(config):raise RuntimeError('HIP kernel launch failed')
            torch.cuda.synchronize()
            for actual,expected in zip(outputs,refs):
                if not torch.isfinite(actual).all():raise ValueError('Non-finite kernel output')
                torch.testing.assert_close(actual,expected,atol=5e-5 if family=='preprocess' else .02,rtol=1e-4 if family=='preprocess' else .02)
            seconds=[]
            for _ in range(20):
                start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                start.record();launch(config);end.record();end.synchronize()
                seconds.append(start.elapsed_time(end)/1000)
            record.update(status='ok',seconds=seconds,median_seconds=statistics.median(seconds),numeric_pass=True)
        except (RuntimeError,AssertionError,ValueError) as e:
            record.update(status='failed',error=str(e))
        write_json(out,records)
    return records
