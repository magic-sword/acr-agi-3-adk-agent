"""Frozen images: direct classification vs observable before/after evidence first."""
import argparse,json,random,shutil,sys,time
from pathlib import Path
from copy import deepcopy
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.benchmark_attention_selection import http,now
from scripts.benchmark_parallel import digest
from scripts.benchmark_recognition import save_json
from scripts.benchmark_official_relations import hf_inputs,ALLOWED,sha

SOURCE=Path('outputs/official-relations-20260928')
MAX_TOKENS=1024
EVIDENCE='''For each ID, write visible evidence BEFORE choosing its kind. Use this key order: id, before, after, description, kind.
In before, describe its visible appearance in the first frame (at most 8 words).
In after, inspect the last frame afresh and describe its visible appearance (at most 8 words). Do not merely copy the inventory or the before sentence.
In description, state the observed agreement or difference (at most 12 words). Check color arrangement, shape, and relation to the SAME nearby visible reference. Look for places occupied before but not after, or after but not before. Do not assume a reference is stationary or that different moving parts share an identity.
Only then choose kind from that visual evidence. A change is not guaranteed. If evidence cannot establish the comparison, use uncertain; do not invent a difference. If the visible state is the same, use unchanged. For X describe unlisted objects, or explicitly report none.
Prefer the images over the inventory or earlier generated sentences when they conflict. Do not provide coordinates, exact directions, roles, causes, or a long explanation.
'''

def prepare(out):
    out.mkdir(parents=True,exist_ok=False);(out/'sources').mkdir()
    for n in ['cases.json','catalogs.json']:shutil.copyfile(SOURCE/n,out/n)
    jobs=[]
    for old in json.loads((SOURCE/'jobs.json').read_text()):
        if old['arm']!='llama_base':continue
        for backend in ['llama','official']:
            for method in ['direct','evidence']:
                j=deepcopy(old);j.update(id=len(jobs),backend=backend,method=method,arm=backend+'_'+method)
                j['payload']['max_tokens']=MAX_TOKENS
                if method=='evidence':
                    c=j['payload']['messages'][1]['content'][-1]
                    old_schema='{"id":"A","kind":"unchanged","description":"brief visual evidence"}'
                    new_schema='{"id":"A","before":"visible first-frame state","after":"visible last-frame state","description":"observed agreement or difference","kind":"unchanged"}'
                    assert old_schema in c['text']
                    c['text']=c['text'].replace(old_schema,new_schema).replace('BEFORE inventory:\n',EVIDENCE+'BEFORE inventory:\n')
                j['payload_digest']=digest(j['payload']);jobs.append(j)
    assert len(jobs)==132
    save_json(out/'jobs.json',jobs)
    manifest={str(p):sha(p) for p in Path('outputs/video-comparison-20260928').glob('*.png') if p.stem.endswith(('before','after'))}
    manifest.update({str(p):sha(p) for p in Path('outputs/video-comparison-20260928').glob('*-grouped.mkv')})
    save_json(out/'input-manifest.json',manifest)
    shutil.copyfile(__file__,out/'sources'/Path(__file__).name)
    save_json(out/'plan.json',dict(created_at=now(),arms=['llama_direct','llama_evidence','official_direct','official_evidence'],requests=132,warmups=4,
        jobs_digest=digest(jobs),script_sha256=sha(__file__),cases_digest=digest(json.loads((out/'cases.json').read_text())),catalog_digest=digest(json.loads((out/'catalogs.json').read_text())),
        design='11 frozen image pairs x 3 frozen BEFORE inventories x 2 prompts x 2 backends; A A B B original video; greedy; no grammar; 1024 maximum output tokens in both prompts.',
        evidence_instruction=EVIDENCE,primary='Abstract identity and event type of moving objects, not coordinates or exact direction. Monitor static and color-only false motion, other change types, output validity, latency and tokens.',
        limitations='Three inventories are not independent scene samples. Before/after descriptions are in one autoregressive response, not independent model calls. Evidence prompt changes requested observations and output order together; not an isolated causal attention manipulation. No feature-difference map is supplied. Direct budget differs from prior 512-token baseline; historical reproduction is checked separately.'))
    print('Prepared',len(jobs),flush=True)

def validate(out):
    plan=json.loads((out/'plan.json').read_text());jobs=json.loads((out/'jobs.json').read_text())
    assert digest(jobs)==plan['jobs_digest'] and sha(__file__)==plan['script_sha256']
    for p,h in json.loads((out/'input-manifest.json').read_text()).items():assert sha(p)==h
    return jobs

def parse(row,answer,stopped,cat):
    row['answer']=answer;row['stopped']=stopped
    try:
        p=json.loads(answer);row['parsed']=p;obs=p['observations']
        expected=[x['id'] for x in cat['items']]+['X']
        row['valid']=bool(stopped and sorted(o['id'] for o in obs)==sorted(expected)
            and all(o['kind'] in ALLOWED and isinstance(o['description'],str) for o in obs))
        if row['method']=='evidence':
            row['evidence_order_valid']=all(list(o)==['id','before','after','description','kind'] and isinstance(o['before'],str) and isinstance(o['after'],str) for o in obs)
            row['valid']=row['valid'] and row['evidence_order_valid']
    except (ValueError,TypeError,KeyError) as e:row.update(valid=False,parse_error=str(e))

def run(out,backend):
    jobs=[j for j in validate(out) if j['backend']==backend];path=out/f'{backend}-measurements.jsonl';assert not path.exists()
    cats={c['key']:c for c in json.loads((out/'catalogs.json').read_text())}
    if backend=='official':
        import torch,transformers
        from transformers import AutoProcessor,Qwen3VLForConditionalGeneration
        torch.set_num_threads(4);modeldir='/tmp/model-cache/qwen3-vl-4b-official'
        processor=AutoProcessor.from_pretrained(modeldir,local_files_only=True)
        model=Qwen3VLForConditionalGeneration.from_pretrained(modeldir,local_files_only=True,dtype=torch.bfloat16,device_map={'':'cuda:0'},attn_implementation='sdpa').eval()
        save_json(out/'official-runtime.json',dict(torch=torch.__version__,transformers=transformers.__version__,gpu=torch.cuda.get_device_name(0),dtype='bfloat16',attention='sdpa'))
    else:save_json(out/'server-before.json',http('http://127.0.0.1:8080/props'))
    order=list(jobs);random.Random(281429).shuffle(order)
    warm=[next(j for j in jobs if j['method']==method) for method in ['direct','evidence']]
    with path.open('w') as f:
        for i,(j,iswarm) in enumerate([(j,True) for j in warm]+[(j,False) for j in order]):
            r={k:v for k,v in j.items() if k!='payload'};r.update(warmup=iswarm,started_at=now());start=time.monotonic()
            try:
                if backend=='llama':
                    response=http('http://127.0.0.1:8080/v1/chat/completions',j['payload'],120);r['response']=response
                    c=response['choices'][0];parse(r,c['message']['content'],c['finish_reason']=='stop',cats[j['catalog_key']])
                    r.update(prompt_tokens=response['usage']['prompt_tokens'],output_tokens=response['usage']['completion_tokens'])
                else:
                    torch.cuda.reset_peak_memory_stats();inputs=hf_inputs(processor,j).to('cuda:0')
                    r.update(prompt_tokens=inputs['input_ids'].shape[1],video_grid_thw=inputs['video_grid_thw'].tolist())
                    with torch.inference_mode():output=model.generate(**inputs,do_sample=False,max_new_tokens=MAX_TOKENS,repetition_penalty=1.0,use_cache=True)
                    torch.cuda.synchronize();generated=output[0,inputs['input_ids'].shape[1]:]
                    answer=processor.tokenizer.decode(generated,skip_special_tokens=True);eos=model.generation_config.eos_token_id
                    parse(r,answer,int(generated[-1]) in ([eos] if isinstance(eos,int) else eos),cats[j['catalog_key']])
                    r.update(output_tokens=len(generated),peak_reserved_bytes=torch.cuda.max_memory_reserved());del inputs,output,generated
            except Exception as e:r.update(valid=False,error=f'{type(e).__name__}: {e}')
            r['seconds']=time.monotonic()-start;r['within_budget']=r['seconds']<=120;r['within_20_seconds']=r['seconds']<=20
            f.write(json.dumps(r,ensure_ascii=False)+'\n');f.flush();print(i,j['case'],j['repeat'],j['method'],r['valid'],r.get('output_tokens'),round(r['seconds'],2),r.get('error',''),flush=True)
            if iswarm and 'error' in r:raise RuntimeError(r['error'])
    if backend=='llama':save_json(out/'server-after.json',http('http://127.0.0.1:8080/props'))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','run']);p.add_argument('--output',type=Path,required=True);p.add_argument('--backend',choices=['llama','official']);a=p.parse_args()
    if a.command=='prepare':prepare(a.output)
    else:run(a.output,a.backend)
