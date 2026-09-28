"""2x2: appearance/relation evidence x repeated AFTER/boundary difference cue."""
import argparse,json,random,shutil,sys,time
from pathlib import Path
from copy import deepcopy
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.benchmark_attention_selection import http,now
from scripts.benchmark_parallel import digest
from scripts.benchmark_recognition import encode,save_json
from scripts.benchmark_official_relations import sha
from scripts.benchmark_observe_compare import parse

SOURCE=Path('outputs/official-relations-20260928')
IMAGES=Path('outputs/overlay-ablation-20260928')
SELECTION={'ls20_up':1,'block_1px':2,'cross_1px':2,'opposite_1px':2,'synthetic_new':2,'synthetic_static':2,'color_only':2}
METHODS=['appearance_repeat','relation_repeat','appearance_edges','relation_edges']
EVIDENCE={
 'appearance':'''In before, describe the object's visible appearance in the original BEFORE (at most 12 words).
In after, inspect the original AFTER afresh and describe its visible appearance (at most 12 words). Do not merely copy the before sentence or inventory.
In description, state observed agreements or differences (at most 12 words), including color arrangement, shape, and spatial relations when useful.''',
 'relation':'''In before, name a nearby visible reference and describe the object's gap, contact, overlap, alignment, or side relative to it in original BEFORE (at most 12 words).
In after, inspect original AFTER afresh and describe those spatial relations to the SAME named reference (at most 12 words). A broad label like "centered" or "upper left" alone is insufficient. Include appearance only to identify the object or a color/shape change.
In description, compare those relations and appearance (at most 12 words). Check newly occupied and vacated places. Check whether the reference itself changed; relative change alone does not prove which object moved. If no reliable reference can be identified, say so rather than inventing one.'''
}
COMMON='''Compare the first two original images: BEFORE and AFTER. The third image is an auxiliary rendering of the SAME pair, not a new observation.
If present, magenta dashed lines mark color boundaries present only in BEFORE, and cyan solid lines mark boundaries present only in AFTER. These lines are annotations, not object colors, new objects, or proven motion. Original images are authoritative. The auxiliary image otherwise retains original AFTER colors and brightness; it may have no marks.
For every listed ID and X, first record BEFORE and AFTER observations, then the observed agreement/difference, and only then classify. Use the key order id, before, after, description, kind.
{evidence}
The BEFORE inventory is an imperfect hypothesis, not a game rule. Prefer images over inventory or earlier generated sentences. Position changes count even when appearance is the same. Do not assume different moving parts form one object or move in the same direction. A change is not guaranteed.
Use one kind: unchanged, moved, shape_changed, separated, resized, color_changed, appeared, disappeared, uncertain. If comparison is unsupported use uncertain; if the visible state agrees use unchanged. Do not give coordinates, distances, exact directions, roles, causes, or a long explanation.
For X, report changed or new objects not individually listed, even if inside a listed broad region. Otherwise report unchanged. Describe actual objects, never annotation marks as objects.
Return a single JSON object with an observations array, following this complete outer structure: {"observations":[{"id":"A","before":"first-frame observation","after":"last-frame observation","description":"observed agreement or difference","kind":"unchanged"}]}.
Include ALL listed IDs and X as separate rows in that array. A single row object is not a complete answer.
BEFORE inventory:
'''

def prepare(out):
    out.mkdir(exist_ok=False);(out/'sources').mkdir()
    cats=[c for c in json.loads((SOURCE/'catalogs.json').read_text()) if SELECTION.get(c['case'])==c['repeat']]
    cases=[c for c in json.loads((SOURCE/'cases.json').read_text()) if c['name'] in SELECTION]
    originals=json.loads((SOURCE/'jobs.json').read_text());jobs=[]
    for c in cases:
        for fr in ['before','after','after_edges']:
            p=IMAGES/f'{c["name"]}-{fr}.png';shutil.copyfile(p,out/p.name)
            if fr!='after_edges':
                old=Image.open(Path('outputs/video-comparison-20260928')/p.name).convert('RGB')
                assert old.tobytes()==Image.open(p).convert('RGB').tobytes()
        cat=next(x for x in cats if x['case']==c['name'])
        old=next(j for j in originals if j['case']==c['name'] and j['repeat']==cat['repeat'] and j['arm']=='llama_base')
        inventory=old['payload']['messages'][1]['content'][-1]['text'].split('BEFORE inventory:\n',1)[1]
        for backend in ['llama','official']:
            for variant in METHODS:
                prompt,visual=variant.split('_');payload=deepcopy(old['payload']);payload['max_tokens']=1024
                aux='after_edges' if visual=='edges' else 'after'
                paths=[f'{c["name"]}-{fr}.png' for fr in ['before','after',aux]]
                content=[]
                for label,path in zip(['ORIGINAL BEFORE','ORIGINAL AFTER','AUXILIARY AFTER'],paths):
                    content.extend([dict(type='text',text=label),encode(Image.open(out/path).convert('RGB'))])
                content.append(dict(type='text',text=COMMON.replace('{evidence}',EVIDENCE[prompt])+inventory))
                payload['messages'][1]['content']=content
                jobs.append(dict(id=len(jobs),case=c['name'],repeat=cat['repeat'],catalog_key=cat['key'],backend=backend,variant=variant,method='evidence',arm=backend+'_'+variant,paths=paths,payload=payload,payload_digest=digest(payload)))
    save_json(out/'jobs.json',jobs);save_json(out/'catalogs.json',cats);save_json(out/'cases.json',cases)
    save_json(out/'input-manifest.json',{p.name:sha(p) for p in out.glob('*.png')})
    for p in [Path(__file__),Path('scripts/benchmark_observe_compare.py'),Path('scripts/summarize_temporal_overlay.py'),Path('scripts/render_overlay_ablation.cjs'),Path('outputs/temporal-overlay-preview-20260928/preview.html')]:shutil.copyfile(p,out/'sources'/p.name)
    save_json(out/'plan.json',dict(created_at=now(),arms=[b+'_'+m for b in ['llama','official'] for m in METHODS],requests=len(jobs),warmups=8,jobs_digest=digest(jobs),script_sha256=sha(__file__),selection=SELECTION,
        cases_digest=digest(cases),catalog_digest=digest(cats),design='7 fixed pairs x 1 fixed inventory x 2 evidence prompts x 2 auxiliary images x 2 backends. Original BEFORE+AFTER+auxiliary AFTER: always three 640x820 images. 1024 output tokens, greedy, no grammar.',
        annotation='Existing verified AFTER+different color boundaries; no dimming, RGB blending, object matching, motion direction grouping or trajectories. Annotation lines overwrite some auxiliary pixels only. Color-only change with unchanged color boundaries has no annotation.',
        controls='Same system prompt, labels, legend, inventory, dimensions, image count, output keys and budgets. Only the evidence instruction or third-image pixels vary. Repeated AFTER controls extra image token count.',
        limitations='Small selected set; no statistical generalization. Static images replace previous AABB video in ALL arms, so prior video scores are not an isolated baseline. Native and BF16 differ in precision and runtime. Appearance and relation prompts request different evidence within the same word limits. This tests sparse boundary difference annotations, not every possible pixel mask or learned-feature map. No attention weights collected.'))
    print('Prepared',len(jobs),flush=True)

def validate(out):
    p=json.loads((out/'plan.json').read_text());jobs=json.loads((out/'jobs.json').read_text())
    assert sha(__file__)==p['script_sha256'] and digest(jobs)==p['jobs_digest']
    for name,h in json.loads((out/'input-manifest.json').read_text()).items():assert sha(out/name)==h
    return jobs

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
    order=list(jobs);random.Random(281803).shuffle(order)
    warm=[next(j for j in jobs if j['variant']==m) for m in METHODS]
    with path.open('w') as f:
        for i,(j,iswarm) in enumerate([(j,True) for j in warm]+[(j,False) for j in order]):
            r={k:v for k,v in j.items() if k!='payload'};r.update(warmup=iswarm,started_at=now());start=time.monotonic()
            try:
                if backend=='llama':
                    response=http('http://127.0.0.1:8080/v1/chat/completions',j['payload'],120);r['response']=response;c=response['choices'][0]
                    parse(r,c['message']['content'],c['finish_reason']=='stop',cats[j['catalog_key']]);r.update(prompt_tokens=response['usage']['prompt_tokens'],output_tokens=response['usage']['completion_tokens'])
                else:
                    torch.cuda.reset_peak_memory_stats();messages=deepcopy(j['payload']['messages'])
                    for c in messages[1]['content']:
                        if c['type']=='image_url':c.clear();c['type']='image'
                    text=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
                    inputs=processor(text=[text],images=[Image.open(out/p).convert('RGB') for p in j['paths']],return_tensors='pt').to('cuda:0')
                    r.update(prompt_tokens=inputs['input_ids'].shape[1],image_grid_thw=inputs['image_grid_thw'].tolist())
                    with torch.inference_mode():output=model.generate(**inputs,do_sample=False,max_new_tokens=1024,repetition_penalty=1.0,use_cache=True)
                    torch.cuda.synchronize();generated=output[0,inputs['input_ids'].shape[1]:];answer=processor.tokenizer.decode(generated,skip_special_tokens=True);eos=model.generation_config.eos_token_id
                    parse(r,answer,int(generated[-1]) in ([eos] if isinstance(eos,int) else eos),cats[j['catalog_key']]);r.update(output_tokens=len(generated),peak_reserved_bytes=torch.cuda.max_memory_reserved());del inputs,output,generated
            except Exception as e:r.update(valid=False,error=f'{type(e).__name__}: {e}')
            r['seconds']=time.monotonic()-start;r['within_budget']=r['seconds']<=120;r['within_20_seconds']=r['seconds']<=20
            f.write(json.dumps(r,ensure_ascii=False)+'\n');f.flush();print(i,j['case'],j['variant'],r['valid'],r.get('output_tokens'),round(r['seconds'],2),r.get('error',''),flush=True)
            if iswarm and 'error' in r:raise RuntimeError(r['error'])
    if backend=='llama':save_json(out/'server-after.json',http('http://127.0.0.1:8080/props'))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','run']);p.add_argument('--output',type=Path,required=True);p.add_argument('--backend',choices=['llama','official']);a=p.parse_args()
    if a.command=='prepare':prepare(a.output)
    else:run(a.output,a.backend)
