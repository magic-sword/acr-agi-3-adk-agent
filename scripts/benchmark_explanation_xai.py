"""Trace actual spatial descriptions, with matched input and prefix interventions."""
import argparse,json,re,shutil,sys,time
from copy import deepcopy
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from PIL import Image
from scripts.benchmark_parallel import digest,read_lines
from scripts.benchmark_official_relations import sha
from scripts.benchmark_recognition import encode,save_json
from scripts.benchmark_attention_selection import now
from scripts.benchmark_xai_motion import ROIS,append_answer

SOURCE=Path('outputs/relation-overlay-20260928-final')
TARGET={'block_1px':'D','cross_1px':'B'}
ARMS=['base','target_black','control_black','inventory_no_position','inventory_wrong_position']
POSITION={'block_1px':'centered vertically in the middle of the game board','cross_1px':'positioned toward the upper left of the game board'}
WRONG={'block_1px':'positioned in the upper left corner of the game board','cross_1px':'positioned toward the lower right of the game board'}
LAYERS=[0,11,23,35]

def prepare(out):
    out.mkdir(exist_ok=False);(out/'sources').mkdir();old=json.loads((SOURCE/'jobs.json').read_text());rows=read_lines(SOURCE/'official-measurements.jsonl');jobs=[];baselines={}
    for case,target in TARGET.items():
        parent=next(j for j in old if j['case']==case and j['arm']=='official_relation_repeat')
        baseline=next(r for r in rows if r['id']==parent['id'] and not r['warmup']);baselines[case]=baseline
        x0,y0,x1,y1=ROIS[case];roi=[59+8*x0,80+8*y0,59+8*x1,80+8*y1]
        control=[59+8*43,80+8*43,59+8*(43+x1-x0),80+8*(43+y1-y0)]
        for arm in ARMS:
            j=deepcopy(parent);j.update(id=len(jobs),arm=arm,target=target,roi=roi,control=control,parent_id=parent['id'])
            paths=[]
            for frame,path in zip(['before','after','aux'],parent['paths']):
                im=Image.open(SOURCE/path).convert('RGB')
                if arm in ['target_black','control_black']:im.paste((0,0,0),roi if arm=='target_black' else control)
                name=f'{case}-{arm}-{frame}.png';im.save(out/name);paths.append(name)
            j['paths']=paths
            for i,path in zip([1,3,5],paths):j['payload']['messages'][1]['content'][i]=encode(Image.open(out/path).convert('RGB'))
            text=j['payload']['messages'][1]['content'][-1]['text']
            if arm.startswith('inventory_'):
                # Exactly one BEFORE inventory item's position phrase, preserving ID and appearance.
                prefix,inventory=text.split('BEFORE inventory:\n');lines=inventory.splitlines()
                line=next(s for s in lines if s.startswith(target+': '));assert POSITION[case] in line
                new=line.replace(POSITION[case],'' if arm=='inventory_no_position' else WRONG[case]).rstrip()
                text=prefix+'BEFORE inventory:\n'+inventory.replace(line,new)
                j['payload']['messages'][1]['content'][-1]['text']=text
            j['payload_digest']=digest(j['payload']);jobs.append(j)
    save_json(out/'jobs.json',jobs);save_json(out/'baselines.json',baselines)
    save_json(out/'input-hashes.json',{p.name:sha(p) for p in out.glob('*.png')})
    save_json(out/'plan.json',dict(created_at=now(),jobs_digest=digest(jobs),script_sha256=sha(__file__),requests=10,warmups=1,cases=list(TARGET),arms=ARMS,layers=LAYERS,
        design='Official BF16 only. Two frozen relation_repeat failures, exact previous payloads. Five input conditions per case; free generation and fixed-baseline-answer likelihood. Two additional prefix-only likelihood interventions per case.',
        attention='Actual baseline target before/after spatial phrases, full before/after, comparison and kind. Recompute selected SDPA rows without changing model output; partition image/inventory/instruction/generated text. No attention-as-causality claim.',
        interventions='Target and equal-area control black masks in all three images; remove or contradict only target position in BEFORE inventory; omit preceding object rows or replace own BEFORE text with not recorded for conditional scores.',
        limitations='Two selected failures, one inventory, greedy, no general accuracy estimate. Occlusion changes scene; misleading inventory is diagnostic only. Fixed-prefix scores are conditional sensitivities, not free-answer probabilities or percentage causal contributions. BF16 findings not assumed to transfer to quantized runtime.'))
    shutil.copyfile(__file__,out/'sources'/Path(__file__).name);print('Prepared',len(jobs),flush=True)

def inputs_for(processor,j,out):
    msg=deepcopy(j['payload']['messages'])
    for c in msg[1]['content']:
        if c['type']=='image_url':c.clear();c['type']='image'
    text=processor.apply_chat_template(msg,tokenize=False,add_generation_prompt=True)
    return processor(text=[text],images=[Image.open(out/p).convert('RGB') for p in j['paths']],return_tensors='pt').to('cuda:0')

def spans(text,target):
    match=re.search(r'\{\s*"id"\s*:\s*"'+target+r'"[^{}]*\}',text);assert match
    fields={}
    for key in ['before','after','description','kind']:
        m=re.search(r'"'+key+r'"\s*:\s*"([^"\n]*)"',match.group());assert m
        fields[key]=(match.start()+m.start(1),match.start()+m.end(1))
    for key in ['before','after']:
        a,b=fields[key];s=text[a:b]
        m=re.search('centered vertically|near upper left.*',s)
        if m:fields[key+'_position']=(a+m.start(),a+m.end())
    return match,fields

def score_and_trace(model,processor,inputs,text,target,out,case,trace=False):
    import numpy as np
    import torch
    from transformers.models.qwen3_vl import modeling_qwen3_vl as q
    enc=processor.tokenizer(text,add_special_tokens=False,return_offsets_mapping=True);match,fields=spans(text,target)
    groups={k:[i for i,(a,b) in enumerate(enc['offset_mapping']) if a<end and b>start] for k,(start,end) in fields.items()}
    selected_ids=sorted({i for ids in groups.values() for i in ids});n=inputs['input_ids'].shape[1]
    full=append_answer(inputs,enc['input_ids']);queries=torch.tensor([n+i-1 for i in selected_ids],device='cuda');lookup={v:i for i,v in enumerate(selected_ids)}
    with torch.inference_mode():plain=model(**full,use_cache=False,logits_to_keep=queries).logits[0].float()
    logp=plain.log_softmax(-1);token_scores={i:float(logp[k,enc['input_ids'][i]]) for k,i in enumerate(selected_ids)}
    scores={key:dict(text=text[slice(*fields[key])],tokens=len(ids),sum_logp=sum(token_scores[i] for i in ids),mean_logp=sum(token_scores[i] for i in ids)/len(ids)) for key,ids in groups.items()}
    if not trace:return scores
    plain=plain.cpu();weights={};original=q.ALL_ATTENTION_FUNCTIONS['sdpa']
    def capture(module,query,key,value,attention_mask,**kw):
        result=original(module,query,key,value,attention_mask,**kw)
        if isinstance(module,q.Qwen3VLTextAttention) and module.layer_idx in LAYERS:
            k=q.repeat_kv(key,module.num_key_value_groups)
            s=torch.matmul(query.index_select(2,queries).float(),k.float().transpose(-2,-1))*kw['scaling']
            if attention_mask is not None:
                mask=attention_mask.index_select(2,queries)[:,:,:,:k.shape[-2]]
                s=s.masked_fill(~mask,float('-inf')) if mask.dtype==torch.bool else s+mask
            else:s=s.masked_fill(torch.arange(k.shape[-2],device=k.device)[None,None,None,:]>queries[None,None,:,None],float('-inf'))
            weights[str(module.layer_idx)]=s.softmax(-1)[0].cpu().numpy()
        return result
    q.ALL_ATTENTION_FUNCTIONS.register('sdpa',capture)
    try:
        with torch.inference_mode():recorded=model(**full,use_cache=False,logits_to_keep=queries).logits[0].float().cpu()
    finally:q.ALL_ATTENTION_FUNCTIONS.register('sdpa',original)
    delta=float((plain-recorded).abs().max());assert delta==0
    allids=full['input_ids'][0].tolist();decoded=processor.tokenizer.decode(allids,skip_special_tokens=False)
    tok=processor.tokenizer(decoded,add_special_tokens=False,return_offsets_mapping=True);assert tok['input_ids']==allids
    invstart=decoded.index('BEFORE inventory:');invend=decoded.index('<|im_end|>',invstart)
    vis=[i for i,x in enumerate(allids) if x==processor.image_token_id];assert len(vis)==1560
    labels=['instruction_and_structure']*len(allids)
    for i,(a,b) in enumerate(tok['offset_mapping']):
        if i<n and a<invend and b>invstart:labels[i]='inventory'
    for frame,name in enumerate(['before_image','after_image','aux_image']):
        for i in vis[frame*520:(frame+1)*520]:labels[i]=name
    for i,(a,b) in enumerate(enc['offset_mapping']):
        labels[n+i]='previous_objects' if b<=match.start() else 'target_structure'
        for key,(start,end) in fields.items():
            if key.endswith('_position'):continue
            if a<end and b>start:labels[n+i]='generated_'+key
    np.savez_compressed(out/f'{case}-attention.npz',**weights,visual_indices=np.array(vis),queries=queries.cpu().numpy())
    meta=dict(case=case,target=target,prompt_tokens=n,query_indices=queries.cpu().tolist(),selected_answer_token_indices=selected_ids,groups=groups,labels=labels,token_ids=allids,token_strings=processor.tokenizer.convert_ids_to_tokens(allids),answer=text,max_logit_difference=delta,layers=LAYERS)
    save_json(out/f'{case}-attention-meta.json',meta)
    traced=[];maps={}
    for key,ids in groups.items():
        maps[key]={};indices=[lookup[i] for i in ids]
        for layer in LAYERS:
            w=weights[str(layer)][:,indices,:].mean(1);assert np.allclose(w.sum(-1),1,atol=1e-6)
            for qi,querypos in enumerate(queries.cpu().tolist()):assert np.all(weights[str(layer)][:,qi,querypos+1:]==0)
            avg=w.mean(0);mass={label:float(avg[np.array(labels)==label].sum()) for label in sorted(set(labels))}
            top=sorted([i for i in range(len(allids)) if i not in set(vis)],key=lambda i:-avg[i])[:12]
            traced.append(dict(field=key,layer=layer+1,mass=mass,top_text=[dict(index=i,weight=float(avg[i]),group=labels[i],token=processor.tokenizer.decode([allids[i]]),context=processor.tokenizer.decode(allids[max(0,i-3):i+4])) for i in top]))
            maps[key][str(layer+1)]=avg[vis].reshape(3,26,20).tolist()
    save_json(out/f'{case}-trace.json',dict(scores=scores,attention=traced,image_maps=maps,max_logit_difference=delta))
    return scores

def run(out):
    import torch,transformers
    from transformers import AutoProcessor,Qwen3VLForConditionalGeneration
    plan=json.loads((out/'plan.json').read_text());jobs=json.loads((out/'jobs.json').read_text())
    assert sha(__file__)==plan['script_sha256'] and digest(jobs)==plan['jobs_digest']
    assert not (out/'measurements.jsonl').exists()
    torch.set_num_threads(4);modeldir='/tmp/model-cache/qwen3-vl-4b-official'
    processor=AutoProcessor.from_pretrained(modeldir,local_files_only=True)
    model=Qwen3VLForConditionalGeneration.from_pretrained(modeldir,local_files_only=True,dtype=torch.bfloat16,device_map={'':'cuda:0'},attn_implementation='sdpa').eval()
    save_json(out/'runtime.json',dict(torch=torch.__version__,transformers=transformers.__version__,gpu=torch.cuda.get_device_name(0),dtype='bfloat16',attention='sdpa'))
    baselines=json.loads((out/'baselines.json').read_text());prefix_scores=[]
    order=[j for j in jobs if j['arm']=='base']+[j for j in jobs if j['arm']!='base']
    with (out/'measurements.jsonl').open('w') as f:
        for j,warm in [(jobs[0],True)]+[(j,False) for j in order]:
            start=time.monotonic();torch.cuda.reset_peak_memory_stats();inputs=inputs_for(processor,j,out)
            with torch.inference_mode():output=model.generate(**inputs,do_sample=False,max_new_tokens=1024,repetition_penalty=1.0,use_cache=True)
            torch.cuda.synchronize();generated=output[0,inputs['input_ids'].shape[1]:];answer=processor.tokenizer.decode(generated,skip_special_tokens=True);eos=model.generation_config.eos_token_id
            stopped=int(generated[-1]) in ([eos] if isinstance(eos,int) else eos)
            row=dict(id=j['id'],case=j['case'],arm=j['arm'],warmup=warm,payload_digest=j['payload_digest'],answer=answer,generation_seconds=time.monotonic()-start,output_tokens=len(generated),image_grid_thw=inputs['image_grid_thw'].tolist())
            try:
                parsed=json.loads(answer);assert [o['id'] for o in parsed['observations']]==['A','B','C','D','E','X'];row.update(parsed=parsed,valid=stopped)
            except Exception as e:row.update(valid=False,parse_error=str(e))
            del generated,output
            if not warm:
                text=baselines[j['case']]['answer']
                if j['arm']=='base':
                    row['baseline_reproduced']=answer==text;assert row['baseline_reproduced'],j['case']
                row['fixed_answer_scores']=score_and_trace(model,processor,inputs,text,j['target'],out,j['case'],trace=j['arm']=='base')
                if j['arm']=='base':
                    m,fields=spans(text,j['target']);without_others='{"observations":['+m.group()+']}'
                    a,b=fields['before'];without_before=text[:a]+'not recorded'+text[b:]
                    for name,changed in [('omit_previous_objects',without_others),('replace_own_before',without_before)]:
                        scored=score_and_trace(model,processor,inputs,changed,j['target'],out,j['case'])
                        prefix_scores.append(dict(case=j['case'],arm=name,answer_prefix_source=changed,scores=scored))
            row.update(total_seconds=time.monotonic()-start,peak_reserved_bytes=torch.cuda.max_memory_reserved());f.write(json.dumps(row,ensure_ascii=False)+'\n');f.flush()
            print(j['case'],j['arm'],'warm' if warm else 'main',row['valid'],round(row['total_seconds'],2),flush=True);del inputs
    save_json(out/'prefix-scores.json',prefix_scores)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','run']);p.add_argument('--output',required=True,type=Path);a=p.parse_args()
    if a.command=='prepare':prepare(a.output)
    else:run(a.output)
