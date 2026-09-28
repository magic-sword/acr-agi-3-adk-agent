"""Compare native/official video paths and relational prompts on frozen cases.

Backend comparison includes precision and implementation changes, not just preprocessing.
No labels, coordinates, game roles or truth are passed to either model.
"""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random
import re
import shutil
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.benchmark_attention_selection import http, now
from scripts.benchmark_parallel import digest, read_lines
from scripts.benchmark_recognition import save_json

SOURCE = Path('outputs/video-comparison-20260928')
ARMS = ['llama_base', 'llama_relation', 'official_base', 'official_relation']
RELATION = '''For each object, compare its relation to the SAME nearby visible reference in BEFORE and AFTER: a boundary, corner, or another object. Check whether their gap, contact, overlap, or alignment changed. Verify whether the reference itself stayed fixed; a changing relation alone does not prove which object moved. If no reliable reference is visible, use uncertain rather than inventing evidence. In the brief description name the object and the reference when useful. Do not assume similarly moving parts form one object.\n'''
ALLOWED = {'unchanged','moved','shape_changed','separated','resized','color_changed','appeared','disappeared','uncertain'}

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def prepare(out):
    out.mkdir(parents=True, exist_ok=False)
    for name in ['cases.json','catalogs.json']:
        shutil.copyfile(SOURCE/name, out/name)
    jobs = []
    for old in json.loads((SOURCE/'jobs.json').read_text()):
        if old['arm'] != 'grouped':
            continue
        for arm in ARMS:
            j = deepcopy(old)
            j.update(id=len(jobs), arm=arm)
            # Both engines generate JSON from instructions only; no grammar advantage.
            j['payload'].pop('response_format', None)
            if arm.endswith('relation'):
                c = j['payload']['messages'][1]['content'][-1]
                c['text'] = c['text'].replace('BEFORE inventory:\n', RELATION+'BEFORE inventory:\n')
            j['payload_digest'] = digest(j['payload'])
            jobs.append(j)
    save_json(out/'jobs.json', jobs)
    manifest = {str(p):sha(p) for p in SOURCE.glob('*.png') if p.stem.endswith(('before','after'))}
    manifest.update({str(p):sha(p) for p in SOURCE.glob('*-grouped.mkv')})
    save_json(out/'input-manifest.json',manifest)
    (out/'sources').mkdir()
    files = [Path(__file__), Path('scripts/summarize_temporal_overlay.py')]
    for f in files:
        shutil.copyfile(f,out/'sources'/f.name)
    save_json(out/'plan.json',dict(created_at=now(), arms=ARMS, requests=len(jobs), warmups=4,
        jobs_digest=digest(jobs), cases_digest=digest(json.loads((out/'cases.json').read_text())),
        catalog_digest=digest(json.loads((out/'catalogs.json').read_text())),
        source_hashes={str(f):sha(f) for f in files},
        design='11 frozen pairs x 3 frozen BEFORE inventories x 2 backends x 2 prompts. Four frames A A B B at 4 fps, original 640x820 RGB. Greedy, 512 output tokens, no JSON grammar in either engine. No motion interpolation.',
        fairness='Prompt comparison is paired within each backend. Backend comparison additionally changes Q4_K_M/Q8_0 to BF16, tokenizer/runtime, resize and temporal processing; not a causal test of preprocessing alone.',
        budget='Recognition is scored within a common 120-second ceiling; 20-second practical budget is reported separately. No repeated sampling or tuning on outcomes.',
        limitations='One real pair plus ten synthetic pairs. Three inventories are not independent stochastic repeats. No gameplay or causal-role score. Assistant semantic review is not independent human review.'))
    print('Prepared',len(jobs),flush=True)

def validate(out):
    plan=json.loads((out/'plan.json').read_text())
    assert digest(json.loads((out/'jobs.json').read_text()))==plan['jobs_digest']
    for f,h in json.loads((out/'input-manifest.json').read_text()).items():
        assert sha(f)==h
    for f,h in plan['source_hashes'].items():
        assert sha(f)==h

def parse(row, answer, stopped, cat):
    row['answer']=answer
    try:
        parsed=json.loads(answer)
        row['parsed']=parsed
        observations=parsed['observations']
        expected=[x['id'] for x in cat['items']]+['X']
        row['valid']=bool(stopped and sorted(x['id'] for x in observations)==sorted(expected)
            and all(x['kind'] in ALLOWED and isinstance(x['description'],str) for x in observations))
    except (ValueError,TypeError,KeyError) as e:
        row.update(valid=False, parse_error=str(e))

def hf_inputs(processor,j):
    import numpy as np
    from PIL import Image
    from transformers.video_utils import VideoMetadata
    messages=deepcopy(j['payload']['messages'])
    messages[1]['content'][1]={'type':'video'}
    text=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
    frames=np.stack([np.asarray(Image.open(SOURCE/f'{j["case"]}-{fr}.png').convert('RGB'))
        for fr in ['before','before','after','after']])
    metadata=VideoMetadata(total_num_frames=4,fps=4,width=640,height=820,duration=1,
        frames_indices=[0,1,2,3])
    return processor(text=[text],videos=[frames],video_metadata=[metadata],
        do_sample_frames=False,return_tensors='pt')

def audit(out, modeldir):
    import inspect
    import torch
    import transformers
    from transformers import AutoProcessor
    torch.set_num_threads(4)
    processor=AutoProcessor.from_pretrained(modeldir,local_files_only=True)
    jobs=json.loads((out/'jobs.json').read_text())
    j=next(j for j in jobs if j['arm']=='official_base')
    inputs=hf_inputs(processor,j)
    text=processor.tokenizer.decode(inputs['input_ids'][0],skip_special_tokens=False)
    grid=inputs['video_grid_thw'].tolist()
    result=dict(transformers=transformers.__version__,torch=torch.__version__,
        sampled_frames=[0,1,2,3],sequence=['before','before','after','after'],fps=4,
        grid_thw=grid,resized_hw=[grid[0][1]*16,grid[0][2]*16],
        visual_tokens=int((inputs['input_ids']==processor.video_token_id).sum()),
        prompt_tokens=inputs['input_ids'].shape[1],timestamps=re.findall(r'<[\d.]+ seconds>',text),
        pixel_values_shape=list(inputs['pixel_values_videos'].shape),
        video_processor=processor.video_processor.to_dict(),
        source_hashes={})
    for obj in [type(processor),type(processor.video_processor),type(processor.image_processor)]:
        path=Path(inspect.getfile(obj));shutil.copyfile(path,out/'sources'/path.name)
        result['source_hashes'][path.name]=sha(path)
    save_json(out/'official-processor-audit.json',result)
    (out/'official-expanded-prompt.txt').write_text(text)
    print(json.dumps(result,indent=2),flush=True)

def run(out,backend,modeldir,base,limit):
    validate(out)
    jobs=[j for j in json.loads((out/'jobs.json').read_text()) if j['arm'].startswith(backend+'_')]
    if limit: jobs=jobs[:limit]
    path=out/f'{backend}-measurements.jsonl'
    assert not path.exists()
    cats={c['key']:c for c in json.loads((out/'catalogs.json').read_text())}
    model=processor=None
    if backend=='official':
        import torch
        import transformers
        from transformers import AutoProcessor,Qwen3VLForConditionalGeneration
        torch.set_num_threads(4)
        processor=AutoProcessor.from_pretrained(modeldir,local_files_only=True)
        start=time.monotonic()
        model=Qwen3VLForConditionalGeneration.from_pretrained(modeldir,local_files_only=True,
            dtype=torch.bfloat16,device_map={'':'cuda:0'},attn_implementation='sdpa').eval()
        save_json(out/'official-model.json',dict(load_seconds=time.monotonic()-start,
            transformers=transformers.__version__,torch=torch.__version__,dtype='bfloat16',
            attention='sdpa',model_config=model.config.to_dict(),
            cuda_device=torch.cuda.get_device_name(0),generation=model.generation_config.to_dict()))
    else:
        save_json(out/'server-before.json',http(base.removesuffix('/v1')+'/props'))
    shuffled=list(jobs);random.Random(290928).shuffle(shuffled)
    warm=[next(j for j in jobs if j['arm']==a) for a in dict.fromkeys(j['arm'] for j in jobs)]
    with path.open('w') as file:
        for i,(j,iswarm) in enumerate([(x,True) for x in warm]+[(x,False) for x in shuffled]):
            row={k:v for k,v in j.items() if k!='payload'}
            row.update(warmup=iswarm,started_at=now());start=time.monotonic()
            try:
                if backend=='llama':
                    response=http(base+'/chat/completions',j['payload'],120)
                    row['response']=response
                    choice=response['choices'][0]
                    parse(row,choice['message']['content'],choice['finish_reason']=='stop',cats[j['catalog_key']])
                    row['prompt_tokens']=response['usage']['prompt_tokens']
                else:
                    torch.cuda.reset_peak_memory_stats()
                    inputs=hf_inputs(processor,j).to('cuda:0')
                    row['prompt_tokens']=inputs['input_ids'].shape[1]
                    row['video_grid_thw']=inputs['video_grid_thw'].tolist()
                    with torch.inference_mode():
                        output=model.generate(**inputs,do_sample=False,max_new_tokens=512,
                            repetition_penalty=1.0,use_cache=True)
                    torch.cuda.synchronize()
                    generated=output[0,inputs['input_ids'].shape[1]:]
                    answer=processor.tokenizer.decode(generated,skip_special_tokens=True)
                    eos=model.generation_config.eos_token_id
                    parse(row,answer,int(generated[-1]) in ([eos] if isinstance(eos,int) else eos),cats[j['catalog_key']])
                    row.update(output_tokens=len(generated),peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                        peak_reserved_bytes=torch.cuda.max_memory_reserved())
                    del inputs,output,generated
            except Exception as e:
                row.update(valid=False,error=f'{type(e).__name__}: {e}')
            row['seconds']=time.monotonic()-start
            row['within_budget']=row['seconds']<=120
            row['within_20_seconds']=row['seconds']<=20
            file.write(json.dumps(row,ensure_ascii=False)+'\n');file.flush()
            print(i,j['case'],j['arm'],row['valid'],round(row['seconds'],2),row.get('error',''),flush=True)
            if iswarm and 'error' in row: raise RuntimeError(row['error'])
    if backend=='llama':save_json(out/'server-after.json',http(base.removesuffix('/v1')+'/props'))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','audit','run'])
    p.add_argument('--output',type=Path,required=True);p.add_argument('--backend',choices=['llama','official'])
    p.add_argument('--modeldir',default='/tmp/model-cache/qwen3-vl-4b-official')
    p.add_argument('--base',default='http://127.0.0.1:8080/v1');p.add_argument('--limit',type=int,default=0)
    a=p.parse_args()
    if a.command=='prepare':prepare(a.output)
    elif a.command=='audit':audit(a.output,a.modeldir)
    else:run(a.output,a.backend,a.modeldir,a.base,a.limit)
