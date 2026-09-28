"""Verify temporal order and surviving frame differences in official input tensors."""
import json
from pathlib import Path
import sys
import numpy as np
import torch
from transformers import AutoProcessor
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.benchmark_official_relations import hf_inputs,sha

out=Path(sys.argv[1]);torch.set_num_threads(4)
processor=AutoProcessor.from_pretrained('/tmp/model-cache/qwen3-vl-4b-official',local_files_only=True)
jobs=json.loads((out/'jobs.json').read_text());rows=[]
for j in jobs:
    if j['arm']!='official_base' or j['repeat']!=0:continue
    batch=hf_inputs(processor,j)
    t,h,w=batch['video_grid_thw'][0].tolist()
    p=processor.video_processor.patch_size;m=processor.video_processor.merge_size
    flat=batch['pixel_values_videos'].cpu().numpy()
    packed=flat.reshape(1,t,h//m,w//m,m,m,3,2,p,p)
    inverse=np.argsort([0,1,4,7,5,8,3,2,6,9])
    frames=packed.transpose(*inverse).reshape(t*2,3,h*p,w*p)
    assert np.array_equal(frames[0],frames[1]) and np.array_equal(frames[2],frames[3])
    changed=np.any(frames[0]!=frames[2],axis=0)
    assert bool(changed.any())==(j['case']!='synthetic_static')
    rows.append({'case':j['case'],'shape_tchw':list(frames.shape),
        'first_pair_equal':True,'last_pair_equal':True,
        'changed_spatial_pixels_after_preprocessing':int(changed.sum()),
        'max_normalized_channel_difference':float(np.abs(frames[0]-frames[2]).max()),
        'mean_normalized_absolute_difference':float(np.abs(frames[0]-frames[2]).mean())})
result={'passed':True,'method':'Invert official flattened input patch tensor to normalized TCHW; verify A,A,B,B equality and preservation of changes before visual encoder. Does not inspect learned features.',
    'rows':rows,'script_sha256':sha(__file__)}
(out/'official-tensor-audit.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
