"""Compare role-free geometric evidence x prior interpretation, on fixed frames."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path
import random
import statistics
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.benchmark_parallel import digest, read_lines, restore_images
from scripts.benchmark_recognition import request, save_json, decode, validate
from scripts.motion_evidence import analyze

ARMS=('raw_no_prior','raw_prior','geometry_no_prior','geometry_prior')
INSTRUCTION='''Compare BEFORE and AFTER. Identify which visible parts changed and how,
and up to three relevant unchanged landmarks. Keep appearance, correspondence,
and game role separate. Do not infer movement from the action name. Prior roles,
if supplied, are hypotheses to reassess, not observations. Geometric candidates,
if supplied, measure color shapes, not object identity or causal control.
Use original 64x64 coordinates: inclusive boxes [left,top,right,bottom], positive
dx rightward, positive dy downward. Give dx/dy only for translation or unchanged,
otherwise null. Pixel count delta is for the described colored part, or null if
unknown. Do not invent intermediate frames. Report a role as unverified when
its function cannot be established. Return one concise submit_motion tool call.'''


def schema():
    text={'type':'string','maxLength':160}
    box={'type':'array','minItems':4,'maxItems':4,'items':{'type':'integer','minimum':0,'maximum':63}}
    nullable={'type':['integer','null']}
    fields={'appearance':text,'before_box':box,'after_box':box,
            'change':{'type':'string','enum':['translation','unchanged','color_or_shape_change','uncertain']},
            'dx':nullable,'dy':nullable,'pixel_count_delta':nullable}
    return {'type':'object','additionalProperties':False,'properties':{
        'regions':{'type':'array','minItems':1,'maxItems':6,'items':{
            'type':'object','additionalProperties':False,'properties':fields,'required':list(fields)}},
        'prior_role_status':{'type':'string','enum':['supported','contradicted','unverified','no_prior']},
        'role_reason':{'type':'string','maxLength':300},
        'uncertainty':{'type':'string','maxLength':200}},
        'required':['regions','prior_role_status','role_reason','uncertainty']}


def compact(evidence):
    static=[];changing=[]
    for row in evidence['regions']:
        if row['old_support_unchanged']:
            static.append({'color_id':row['color_id'],'box':row['before_box'],'pixels':row['before_pixels']})
        else:changing.append(row)
    return {'measured_unchanged_color_supports':static,'changed_color_regions':changing,
            'common_motion_candidates':evidence['common_motion_candidates'],
            'limit':evidence['interpretation_limit']}


def prepare(source,output):
    output.mkdir(parents=True,exist_ok=False)
    directory=source/'ls20-9607627b/cognition'
    reqs=[r for p in directory.glob('*.requests.jsonl') for r in read_lines(p)]
    r=next(r for r in reqs if r['work']=='reconcile')
    payload=restore_images(r['request'],directory);parts=payload['messages'][1]['content']
    images=[p for p in parts if p['type']=='image_url'];assert len(images)==2
    context=json.loads(parts[-1]['text'])
    files=sorted((directory/'frames').glob('*.json'))
    before,after=[json.loads(p.read_text())['frames'][-1] for p in files]
    palette={}
    image=decode(images[0])
    for y,row in enumerate(before):
        for x,c in enumerate(row):palette[str(c)]='#%02x%02x%02x'%image.getpixel((32+x*6,44+y*6))
    common={'board_size':[64,64],'palette':palette,
            'question':'Which parts changed, which stayed in place, and what does this imply for any supplied prior roles?'}
    raw={k:v for k,v in context['last_result'].items() if k in (
        'before_observation_id','after_observation_id','action','acknowledged','changed_cell_count','changed_cells','change_bounds')}
    raw['action']=raw['action']['action']
    cases=[];evidence={}
    for name in ('up','action_hidden','same_frame'):
        old=before;new=before if name=='same_frame' else after
        t=time.perf_counter();facts=analyze(old,new);elapsed=time.perf_counter()-t
        evidence[name]={'full':facts,'prepare_seconds':elapsed}
        obs=deepcopy(raw)
        if name=='action_hidden':obs['action']='withheld for diagnostic'
        if name=='same_frame':
            obs.update(action='none; duplicate-frame diagnostic',acknowledged=False,
                       after_observation_id=obs['before_observation_id'],changed_cell_count=0,changed_cells=[],change_bounds=None)
        for arm in ARMS:
            ctx={**common,'observation':obs}
            if arm.endswith('_prior') and not arm.endswith('no_prior'):ctx['prior_understanding']=context['understanding']
            if arm.startswith('geometry'):ctx['geometric_evidence']=compact(facts)
            body=[{'type':'text','text':'BEFORE'},images[0],{'type':'text','text':'AFTER'},
                  images[0] if name=='same_frame' else images[1],
                  {'type':'text','text':json.dumps(ctx,separators=(',',':'))}]
            p={'model':'qwen3-vl-4b-instruct','messages':[{'role':'system','content':INSTRUCTION},{'role':'user','content':body}],
               'temperature':0,'seed':123,'max_tokens':1000,'stream':False,'cache_prompt':True,'id_slot':0,
               'tools':[{'type':'function','function':{'name':'submit_motion','description':'Submit visual changes and prior-role assessment.','parameters':schema()}}],
               'tool_choice':'required','parallel_tool_calls':False}
            cases.append({'name':name,'arm':arm,'payload':p,'payload_digest':digest(p),
                          'source_request':r['request_sha256']})
    save_json(output/'cases.json',cases);save_json(output/'evidence.json',evidence)
    save_json(output/'frames.json',{'before':before,'after':after})
    save_json(output/'rubric.json',{'created_at':datetime.now(timezone.utc).isoformat(),
        'moving_before':[34,45,38,49],'moving_after':[34,40,38,44],'dx':0,'dy':-5,
        'white_cross':[20,31,22,33],'bar':[13,61,54,62],'bar_pixel_delta':-2,
        'note':'Predeclared box-based scoring checked against raw pixels. Semantic role review remains assistant review. Same frame is a diagnostic, not an environment transition.',
        'motion_pass':'Correct displacement and >=80% coverage of moving reference box, union over predicted boxes, without individual predicted boxes exceeding 2x reference area.',
        'cross_pass':'Unchanged dx=dy=0 with IoU>=0.4 against white cross box.',
        'bar_pass':'color_or_shape_change with -2 pixel delta and box intersects the changed yellow column at x=13,y=61..62.',
        'control_pass':'All regions unchanged with zero dx/dy and zero/null count delta; no prior role is marked supported.'})
    save_json(output/'experiment.json',{'created_at':datetime.now(timezone.utc).isoformat(),'case_digest':digest(cases),
        'source':str(source),'source_frame_hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        'repeats':3,'warmups_per_input':1,'arms':ARMS,
        'scope':'Single real transition plus action-hidden and identical-frame diagnostics. Not three independent transitions.',
        'prompt':'Common motion-specific task and schema in all arms. Not directly comparable to prior understand-task success rates.'})
    decode(images[0]).save(output/'before.png');decode(images[1]).save(output/'after.png')
    print('Prepared',len(cases),'inputs',flush=True)


def extract(response):
    c=response['choices'][0]
    if c.get('finish_reason')=='length':raise ValueError('truncated output')
    calls=c['message'].get('tool_calls',[])
    if len(calls)!=1 or calls[0]['function']['name']!='submit_motion':raise ValueError('expected one submit_motion')
    result=calls[0]['function']['arguments'];result=json.loads(result) if isinstance(result,str) else result
    validate(result,schema())
    for r in result['regions']:
        for key in ('before_box','after_box'):
            b=r[key]
            if any(type(x)!=int or not 0<=x<64 for x in b) or b[0]>b[2] or b[1]>b[3]:raise ValueError('invalid box')
        for key in ('dx','dy','pixel_count_delta'):
            if r[key] is not None and type(r[key])!=int:raise ValueError('invalid numeric measurement')
    return result


def pixels(box):return {(x,y) for y in range(box[1],box[3]+1) for x in range(box[0],box[2]+1)}


def score(value,name):
    regions=value.get('regions',[])
    if not regions:return {'motion':False,'cross':False,'bar':False,'control':False}
    target=pixels([34,45,38,49]);covered=set();cross=pixels([20,31,22,33]);cross_ok=False;bar_ok=False
    for r in regions:
        b=pixels(r['before_box']);a=pixels(r['after_box'])
        if r['change']=='translation' and (r['dx'],r['dy'])==(0,-5) and len(b)<=2*len(target):
            shifted={(x,y+5) for x,y in a};covered|=b&shifted&target
        if r['change']=='unchanged' and (r['dx'],r['dy'])==(0,0) and len(b&cross)/len(b|cross)>=.4 and b==a:cross_ok=True
        if r['change']=='color_or_shape_change' and r['pixel_count_delta']==-2 and b&{(13,61),(13,62)}:bar_ok=True
    return {'motion':name!='same_frame' and len(covered)/len(target)>=.8,
            'cross':cross_ok,'bar':name!='same_frame' and bar_ok,
            'control':name=='same_frame' and all(r['change']=='unchanged' and (r['dx'],r['dy'])==(0,0)
                and r['pixel_count_delta'] in (0,None) and r['before_box']==r['after_box'] for r in regions)
                and value.get('prior_role_status')!='supported'}


def run(output,base):
    cases=json.loads((output/'cases.json').read_text());frames=json.loads((output/'frames.json').read_text())
    props=request(base.removesuffix('/v1')+'/props')
    if props['total_slots']!=1:raise ValueError('expected single-slot server')
    if (output/'measurements.jsonl').exists():raise ValueError('measurement file already exists')
    save_json(output/'environment.json',{'server':props,'case_digest':digest(cases),
        'source_hashes':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path(__file__).with_name('motion_evidence.py')]}})
    with (output/'measurements.jsonl').open('w') as log:
        for rep in (-1,0,1,2):
            indices=list(range(len(cases)));random.Random(8300+rep).shuffle(indices)
            for index in indices:
                case=cases[index];row={k:v for k,v in case.items() if k!='payload'}
                row.update(repeat=rep,warmup=rep==-1,started_at=datetime.now(timezone.utc).isoformat())
                t=time.perf_counter();p=deepcopy(case['payload']);pre=0.0
                if case['arm'].startswith('geometry'):
                    start=time.perf_counter();e=analyze(frames['before'],frames['before'] if case['name']=='same_frame' else frames['after']);pre=time.perf_counter()-start
                    ctx=json.loads(p['messages'][1]['content'][-1]['text']);ctx['geometric_evidence']=compact(e)
                    p['messages'][1]['content'][-1]['text']=json.dumps(ctx,separators=(',',':'))
                if digest(p)!=case['payload_digest']:raise ValueError('input changed at replay')
                row['preprocess_seconds']=pre
                start=time.perf_counter()
                try:
                    response=request(base+'/chat/completions',p);row['response']=response
                    row['usage']=response.get('usage');row['timings']=response.get('timings')
                    row['parsed']=extract(response);row['valid']=True
                except Exception as exc:row.update(valid=False,error=f'{type(exc).__name__}: {exc}')
                row['model_seconds']=time.perf_counter()-start;row['seconds']=time.perf_counter()-t
                row['score']=score(row.get('parsed',{}),row['name'])
                log.write(json.dumps(row,ensure_ascii=False)+'\n');log.flush()
                print(rep,row['name'],row['arm'],round(row['seconds'],2),row['valid'],row['score'],row.get('error',''),flush=True)
    save_json(output/'server-after.json',request(base.removesuffix('/v1')+'/props'));report(output)


def report(output):
    rows=[r for r in read_lines(output/'measurements.jsonl') if not r['warmup']]
    summary=[]
    for name in ('up','action_hidden','same_frame'):
        for arm in ARMS:
            rs=[r for r in rows if (r['name'],r['arm'])==(name,arm)]
            if not rs:continue
            summary.append({'case':name,'arm':arm,'n':len(rs),'valid':sum(r['valid'] for r in rs),
                **{k:sum(r['score'][k] for r in rs) for k in ('motion','cross','bar','control')},
                'seconds_median':statistics.median(r['seconds'] for r in rs),
                'preprocess_ms_median':1000*statistics.median(r['preprocess_seconds'] for r in rs),
                'prompt_tokens_median':statistics.median((r.get('usage') or {}).get('prompt_tokens',0) for r in rs),
                'completion_tokens_median':statistics.median((r.get('usage') or {}).get('completion_tokens',0) for r in rs)})
    save_json(output/'summary.json',summary)
    unique={}
    for r in rows:
        v=r.get('parsed',r.get('response',{}));key=digest(v)
        unique.setdefault(key,{'id':key,'output':v,'members':[]})['members'].append({k:r[k] for k in ('name','arm','repeat','score','valid')})
    save_json(output/'review-candidates.json',list(unique.values()))
    lines=['# 動きの対応候補 × 過去の役割説明','',
        '単一のls20操作前後と診断条件。同じ画像の反復であり、未見ゲームの認識精度ではない。',
        'motion:橙青領域の対応と上5画素、cross:白十字の不変、bar:帯の2画素減少、control:同一画像で変化を主張しない。',
        '得点は事前の座標・数値条件による自動採点。意味の正確さと役割訂正は応答全文で別に確認する。','',
        '|ケース|条件|n|形式通過|motion|cross|bar|control|秒|前処理ms|入力/出力tokens|',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for s in summary:lines.append(f'|{s["case"]}|{s["arm"]}|{s["n"]}|{s["valid"]}|{s["motion"]}|{s["cross"]}|{s["bar"]}|{s["control"]}|{s["seconds_median"]:.2f}|{s["preprocess_ms_median"]:.2f}|{s["prompt_tokens_median"]:g}/{s["completion_tokens_median"]:g}|')
    lines+=['','時間はホストの対応候補計算とHTTP要求を含む。KVキャッシュあり。','全条件で同じ動き比較用の問い・出力Schema。以前のunderstand課題との直接的な成功率比較はできない。']
    (output/'report.md').write_text('\n'.join(lines)+'\n')
    page=['<!doctype html><html lang="ja"><meta charset="utf-8"><title>動きの認識比較</title><style>body{font:16px system-ui;margin:24px}pre{white-space:pre-wrap;background:#f2f4f6;padding:12px}article{border:1px solid #bbb;padding:16px;margin:16px 0}img{image-rendering:pixelated}</style>',
          '<h1>動きの認識比較</h1><img src="before.png"><img src="after.png">']
    for r in unique.values():
        page.append('<article><h2>'+html.escape(', '.join(f'{m["name"]}/{m["arm"]}/r{m["repeat"]}' for m in r['members']))+'</h2><pre>'+html.escape(json.dumps(r,ensure_ascii=False,indent=2))+'</pre></article>')
    (output/'comparison.html').write_text('\n'.join(page)+'</html>')
    print('Summarized',len(rows),'measurements;',len(unique),'unique responses',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=['prepare','run','report'])
    parser.add_argument('--source',type=Path,default=Path('outputs/evaluations/20260926T172310706477Z'))
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--base',default='http://127.0.0.1:8080/v1')
    a=parser.parse_args()
    if a.command=='prepare':prepare(a.source,a.output)
    elif a.command=='run':run(a.output,a.base)
    else:report(a.output)


if __name__=='__main__':main()
