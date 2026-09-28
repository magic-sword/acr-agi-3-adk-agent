"""Frozen-case XAI: input interventions, visual features and selected attention rows."""
import argparse, base64, hashlib, json, random, re, shutil, sys, time
from copy import deepcopy
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.benchmark_attention_selection import http,now
from scripts.benchmark_parallel import digest,read_lines
from scripts.benchmark_recognition import save_json

SOURCE=Path('outputs/official-relations-20260928')
IMAGES=Path('outputs/video-comparison-20260928')
CASES=['block_1px','cross_1px','color_only','synthetic_static','synthetic_new']
TARGET={'block_1px':'D','cross_1px':'B','color_only':'B','synthetic_static':'D','synthetic_new':'X'}
ROIS={'block_1px':[29,30,36,39],'cross_1px':[10,18,16,23],
      'color_only':[10,18,15,23],'synthetic_static':[29,31,36,39],'synthetic_new':[15,37,26,44]}
ARMS=['base','target_gray','target_black','control_gray','control_black','ui_gray',
      'restore_target','before_only','after_only','free_inventory','free_no_inventory']
FREE='''Compare the first BEFORE and last AFTER frames. Describe every visible changed or new object by appearance. Distinguish moved, shape_changed, separated, resized, color_changed, appeared, disappeared, uncertain. No coordinates, distances, exact direction, roles or causes are needed. Return only JSON: {"observations":[{"kind":"moved","description":"brief actual object and visual evidence"}]}. Use an empty observations array if no change is visible. Repeated frames are not additional observations.''' 

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def prepare(out):
    from PIL import Image
    from scripts.benchmark_video_comparison import ffmpeg
    out.mkdir(parents=True,exist_ok=False);(out/'sources').mkdir()
    old=json.loads((SOURCE/'jobs.json').read_text());jobs=[];manifest=[]
    for name in CASES:
        base=next(j for j in old if j['case']==name and j['repeat']==0 and j['arm']=='llama_base')
        original=[Image.open(IMAGES/f'{name}-{fr}.png').convert('RGB') for fr in ['before','after']]
        x0,y0,x1,y1=ROIS[name];roi=[59+8*x0,80+8*y0,59+8*x1,80+8*y1]
        control=[59+8*43,80+8*43,59+8*(43+x1-x0),80+8*(43+y1-y0)]
        assert original[0].crop(control).tobytes()==original[1].crop(control).tobytes()
        for arm in ARMS:
            frames=[a.copy() for a in original]
            if arm.startswith(('target_','control_')):
                box=roi if arm.startswith('target_') else control
                fill=102 if arm.endswith('gray') else 0
                for a in frames:a.paste((fill,fill,fill),box)
            elif arm=='ui_gray':
                for a in frames:
                    board=a.crop((59,80,571,592));a.paste((51,51,51),(0,0,640,820));a.paste(board,(59,80))
            elif arm=='restore_target':frames[1].paste(frames[0].crop(roi),roi[:2])
            elif arm=='before_only':frames[1]=frames[0].copy()
            elif arm=='after_only':frames[0]=frames[1].copy()
            for fr,a in zip(['before','after'],frames):a.save(out/f'{name}-{arm}-{fr}.png')
            data=b''.join(a.tobytes() for a in [frames[0],frames[0],frames[1],frames[1]])
            clip=ffmpeg(['-f','rawvideo','-pixel_format','rgb24','-video_size','640x820','-framerate','4','-i','pipe:0','-frames:v','4','-c:v','ffv1','-level','3','-f','matroska','pipe:1'],data)
            decoded=ffmpeg(['-i','pipe:0','-vf','fps=4','-f','rawvideo','-pix_fmt','rgb24','pipe:1'],clip)
            assert decoded==data
            (out/f'{name}-{arm}.mkv').write_bytes(clip)
            p=deepcopy(base['payload']);content=p['messages'][1]['content']
            content[1]={'type':'input_video','input_video':{'data':'data:video/x-matroska;base64,'+base64.b64encode(clip).decode()}}
            if arm.startswith('free_'):
                cat=next(c for c in json.loads((SOURCE/'catalogs.json').read_text()) if c['key']==base['catalog_key'])
                listing='\nBEFORE-only hypotheses, not rules:\n'+'\n'.join(c['description'] for c in cat['items'])
                content[-1]['text']=FREE+(listing if arm=='free_inventory' else '')
            jobs.append(dict(id=len(jobs),case=name,arm=arm,target=TARGET[name],roi=roi,control=control,payload=p,payload_digest=digest(p)))
            manifest.append(dict(case=name,arm=arm,sha256=sha(out/f'{name}-{arm}.mkv'),decoded_sha256=hashlib.sha256(decoded).hexdigest(),frames=4,fps=4,lossless=True))
    save_json(out/'jobs.json',jobs);save_json(out/'video-manifest.json',manifest)
    f=Path(__file__);shutil.copyfile(f,out/'sources'/f.name)
    save_json(out/'plan.json',dict(created_at=now(),cases=CASES,arms=ARMS,requests_per_backend=len(jobs),jobs_digest=digest(jobs),script_sha256=sha(f),
        design='Five frozen synthetic cases, inventory repeat0. Nine image interventions using original inventory task; two separate free-description tasks differ only by BEFORE inventory text. Per backend:55 main requests plus one warmup.',
        controls='Equal-area stationary control masks, gray and black fills. No object-role labels supplied. Known target ROIs used only by intervention/evaluation. Same frames and prompts across backends.',
        attribution='Official BF16 SDPA outputs retained. Recompute selected attention query rows separately; verify recorded vs unrecorded next-token logits. Attention is association, not proof of causal use.',
        scoring='Free generation plus official conditional log likelihood of unchanged vs alternative kind after fixed baseline answer prefix. These scores are diagnostic, not calibrated probabilities or original task accuracy.',
        limitations='One inventory and five synthetic scenes, no game success estimate. Backend precision differs. No gradients or training. Attention grid26x20; DeepStack pathways included.'))
    save_json(out/'input-hashes.json',{p.name:sha(p) for p in out.iterdir() if p.suffix in ['.png','.mkv']})
    print('Prepared',len(jobs),'verified clips',flush=True)

def inputs_for(processor,j,out):
    import numpy as np
    from PIL import Image
    from transformers.video_utils import VideoMetadata
    msg=deepcopy(j['payload']['messages']);msg[1]['content'][1]={'type':'video'}
    text=processor.apply_chat_template(msg,tokenize=False,add_generation_prompt=True)
    frames=np.stack([np.asarray(Image.open(out/f'{j["case"]}-{j["arm"]}-{fr}.png').convert('RGB')) for fr in ['before','before','after','after']])
    return processor(text=[text],videos=[frames],video_metadata=[VideoMetadata(total_num_frames=4,fps=4,width=640,height=820,duration=1,frames_indices=[0,1,2,3])],do_sample_frames=False,return_tensors='pt')

def parse(answer):
    parsed=json.loads(answer);assert isinstance(parsed['observations'],list)
    return parsed

def run_native(out):
    jobs=json.loads((out/'jobs.json').read_text());assert not (out/'llama.jsonl').exists()
    save_json(out/'server-before.json',http('http://127.0.0.1:8080/props'))
    order=list(jobs);random.Random(281023).shuffle(order)
    with (out/'llama.jsonl').open('w') as f:
        for i,(j,warm) in enumerate([(jobs[0],True)]+[(j,False) for j in order]):
            start=time.monotonic();r=dict(id=j['id'],case=j['case'],arm=j['arm'],warmup=warm,payload_digest=j['payload_digest'])
            response=http('http://127.0.0.1:8080/v1/chat/completions',j['payload'],120)
            answer=response['choices'][0]['message']['content'];r.update(answer=answer,response=response,seconds=time.monotonic()-start)
            try:r.update(parsed=parse(answer),valid=response['choices'][0]['finish_reason']=='stop')
            except Exception as e:r.update(valid=False,error=str(e))
            f.write(json.dumps(r)+'\n');f.flush();print(i,j['case'],j['arm'],r['valid'],round(r['seconds'],2),flush=True)
    save_json(out/'server-after.json',http('http://127.0.0.1:8080/props'))

def append_answer(inputs,ids):
    import torch
    result=dict(inputs);suffix=torch.tensor([ids],device=inputs['input_ids'].device,dtype=torch.long)
    result['input_ids']=torch.cat([inputs['input_ids'],suffix],dim=1)
    result['attention_mask']=torch.ones_like(result['input_ids'])
    return result

def feature_metrics(store,out,name):
    import numpy as np
    maps={};stats=[]
    for label,a in store.items():
        if a.shape[0]==4160:
            a=a.reshape(2,26,20,2,2,-1).transpose(0,1,3,2,4,5).reshape(2,52,40,-1)
        else:a=a.reshape(2,26,20,-1)
        delta=np.linalg.norm(a[1]-a[0],axis=-1)
        norm=(np.linalg.norm(a[1],axis=-1)+np.linalg.norm(a[0],axis=-1))/2
        relative=delta/(norm+1e-8)
        cosine=1-(a[0]*a[1]).sum(-1)/(np.linalg.norm(a[0],axis=-1)*np.linalg.norm(a[1],axis=-1)+1e-8)
        maps[label]=relative;stats.append(dict(stage=label,shape=list(relative.shape),mean_relative=float(relative.mean()),max_relative=float(relative.max()),mean_cosine_distance=float(cosine.mean())))
    np.savez_compressed(out/f'{name}-feature-maps.npz',**maps)
    return stats

def inspect_baseline(model,processor,inputs,j,row,out):
    import numpy as np
    import torch
    from transformers.models.qwen3_vl import modeling_qwen3_vl as q
    text=row['answer'];target=j['target']
    match=re.search(r'"id"\s*:\s*"'+target+r'"\s*,\s*"kind"\s*:\s*"([^"\n]+)',text)
    assert match,(j['case'],text)
    char=match.start(1);tokens=processor.tokenizer(text,add_special_tokens=False,return_offsets_mapping=True)
    index=next(i for i,(a,b) in enumerate(tokens['offset_mapping']) if a<=char<b)
    prompt_n=inputs['input_ids'].shape[1];query=prompt_n+index-1
    full=append_answer(inputs,tokens['input_ids']);selected=torch.tensor([query],device='cuda')
    features={};handles=[]
    def feature_hook(label):
        def hook(module,args,result):
            if isinstance(result,tuple):result=result[0]
            features[label]=result.detach().float().cpu().numpy()
        return hook
    visual=model.model.visual
    handles.append(visual.patch_embed.register_forward_hook(feature_hook('patch_embed')))
    for i in [5,11,17,23]:handles.append(visual.blocks[i].register_forward_hook(feature_hook(f'vision_{i+1}')))
    handles.append(visual.merger.register_forward_hook(feature_hook('merger')))
    for i,m in enumerate(visual.deepstack_merger_list):handles.append(m.register_forward_hook(feature_hook(f'deepstack_{i+1}')))
    with torch.inference_mode():plain=model(**full,use_cache=False,logits_to_keep=selected).logits.float().cpu()
    for h in handles:h.remove()
    stats=feature_metrics(features,out,j['case']);del features
    original=q.ALL_ATTENTION_FUNCTIONS['sdpa'];attentions={}
    layers=[0,11,23,35]
    def capture(module,query_states,key,value,attention_mask,**kw):
        result=original(module,query_states,key,value,attention_mask,**kw)
        if isinstance(module,q.Qwen3VLTextAttention) and module.layer_idx in layers:
            k=q.repeat_kv(key,module.num_key_value_groups)
            scores=torch.matmul(query_states[:,:,query:query+1].float(),k.float().transpose(-2,-1))*kw['scaling']
            if attention_mask is not None:
                mask=attention_mask[:,:,query:query+1,:k.shape[-2]]
                scores=scores.masked_fill(~mask,float('-inf')) if mask.dtype==torch.bool else scores+mask
            else:scores=scores.masked_fill(torch.arange(k.shape[-2],device=k.device)>query,float('-inf'))
            weights=scores.softmax(-1)[0,:,0].cpu().numpy()
            attentions[str(module.layer_idx)]=weights
        return result
    q.ALL_ATTENTION_FUNCTIONS.register('sdpa',capture)
    try:
        with torch.inference_mode():captured=model(**full,use_cache=False,logits_to_keep=selected).logits.float().cpu()
    finally:q.ALL_ATTENTION_FUNCTIONS.register('sdpa',original)
    delta=float((plain-captured).abs().max());assert delta==0,(j['case'],delta)
    allids=full['input_ids'][0].tolist();visual_indices=[i for i,t in enumerate(allids) if t==processor.video_token_id]
    assert len(visual_indices)==1040 and set(attentions)==set(map(str,layers))
    np.savez_compressed(out/f'{j["case"]}-attention.npz',**attentions,visual_indices=np.array(visual_indices))
    # Save token labels and exact prompt/generated boundaries for area/text attribution.
    save_json(out/f'{j["case"]}-attention-meta.json',dict(query_index=query,kind=match.group(1),target=target,
        predicted_token_index=prompt_n+index,predicted_token_id=tokens['input_ids'][index],
        prompt_tokens=prompt_n,token_strings=processor.tokenizer.convert_ids_to_tokens(allids),
        decoded_text=processor.tokenizer.decode(allids,skip_special_tokens=False),
        layers=layers,max_logit_difference=delta,feature_stats=stats,roi=j['roi'],control=j['control']))
    return text[:char]

def candidate_scores(model,processor,inputs,prefix,alternative):
    import torch
    prefix_ids=processor.tokenizer.encode(prefix,add_special_tokens=False);scores={}
    for label in ['unchanged',alternative]:
        ids=processor.tokenizer.encode(prefix+label,add_special_tokens=False)
        assert ids[:len(prefix_ids)]==prefix_ids
        suffix=ids[len(prefix_ids):];n=inputs['input_ids'].shape[1]
        full=append_answer(inputs,ids)
        selected=torch.arange(n+len(prefix_ids)-1,n+len(ids)-1,device='cuda')
        with torch.inference_mode():logits=model(**full,use_cache=False,logits_to_keep=selected).logits[0].float().log_softmax(-1)
        scores[label]=sum(float(logits[i,t]) for i,t in enumerate(suffix))
    return dict(log_likelihood=scores,alternative=alternative,alternative_minus_unchanged=scores[alternative]-scores['unchanged'])

def run_official(out):
    import torch,transformers
    from transformers import AutoProcessor,Qwen3VLForConditionalGeneration
    torch.set_num_threads(4);modeldir='/tmp/model-cache/qwen3-vl-4b-official'
    processor=AutoProcessor.from_pretrained(modeldir,local_files_only=True)
    model=Qwen3VLForConditionalGeneration.from_pretrained(modeldir,local_files_only=True,dtype=torch.bfloat16,device_map={'':'cuda:0'},attn_implementation='sdpa').eval()
    jobs=json.loads((out/'jobs.json').read_text());assert not (out/'official.jsonl').exists()
    save_json(out/'official-runtime.json',dict(torch=torch.__version__,transformers=transformers.__version__,gpu=torch.cuda.get_device_name(0),dtype='bfloat16',attention='sdpa'))
    # Baselines must run first to define the actual-answer prefix; interventions shuffled after.
    baselines=[j for j in jobs if j['arm']=='base'];rest=[j for j in jobs if j['arm']!='base'];random.Random(281023).shuffle(rest)
    prefixes={}
    with (out/'official.jsonl').open('w') as f:
        for i,(j,warm) in enumerate([(baselines[0],True)]+[(j,False) for j in baselines+rest]):
            torch.cuda.reset_peak_memory_stats();start=time.monotonic();inputs=inputs_for(processor,j,out).to('cuda:0')
            with torch.inference_mode():output=model.generate(**inputs,do_sample=False,max_new_tokens=512,repetition_penalty=1.0,use_cache=True)
            torch.cuda.synchronize();generated=output[0,inputs['input_ids'].shape[1]:]
            answer=processor.tokenizer.decode(generated,skip_special_tokens=True)
            r=dict(id=j['id'],case=j['case'],arm=j['arm'],warmup=warm,payload_digest=j['payload_digest'],answer=answer,
                seconds=time.monotonic()-start,output_tokens=len(generated),input_grid=inputs['video_grid_thw'].tolist())
            eos=model.generation_config.eos_token_id
            try:r.update(parsed=parse(answer),valid=int(generated[-1]) in ([eos] if isinstance(eos,int) else eos))
            except Exception as e:r.update(valid=False,error=str(e))
            del output,generated
            if not warm:
                if j['arm']=='base':prefixes[j['case']]=inspect_baseline(model,processor,inputs,j,r,out)
                if not j['arm'].startswith('free_'):
                    alt={'color_only':'color_changed','synthetic_new':'appeared'}.get(j['case'],'moved')
                    r['candidate_scores']=candidate_scores(model,processor,inputs,prefixes[j['case']],alt)
            r.update(total_seconds=time.monotonic()-start,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
            f.write(json.dumps(r)+'\n');f.flush();del inputs
            print(i,j['case'],j['arm'],r['valid'],round(r['seconds'],2),flush=True)
    save_json(out/'candidate-prefixes.json',prefixes)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','native','official']);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.command=='prepare':prepare(a.output)
    elif a.command=='native':run_native(a.output)
    else:run_official(a.output)
