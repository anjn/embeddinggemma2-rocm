"""Process boundary for potentially failing compiler and GPU operations."""
import argparse,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('job');a=p.parse_args()
job=json.loads(Path(a.job).read_text())
if job['kind']=='micro':
    from .micro import run
    run(job)
elif job['kind']=='compile':
    from ..preparation import compile_images
    compile_images(job['model'],job['output'],[job['budget']],[job['batch']],
                   tuple(job['aspect']),job['precision'],job.get('vision_batch'),
                   cache_dirs=job.get('cache_dirs',[]),kernel_config=job.get('kernel_config'),use_presets=False)
elif job['kind']=='probe':
    import torch
    assert (torch.ones(16,device='cuda')+1).sum().item()==32
else:raise ValueError('Unknown worker job')
