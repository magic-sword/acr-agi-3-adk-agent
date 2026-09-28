"""Exploratory coordinate-free probe after the primary visual-emphasis test.

Keeps the same four input images; asks directional/relative comparisons rather
than coordinates. Results must not replace the frozen primary evaluation.
"""
import argparse
import hashlib
import html
import json
from pathlib import Path
import random
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.benchmark_visual_emphasis import ARMS, now, payload, views
from scripts.benchmark_recognition import save_json, validate
from scripts.benchmark_parallel import digest, read_lines
from scripts.benchmark_attention_selection import http

OPTIONS={
    'orange_blue_change':['up','down','left','right','unchanged','uncertain'],
    'white_cross_relative_to_orange_blue':['above_left','above','above_right','left','overlap','right','below_left','below','below_right','uncertain'],
    'white_cross_change':['moved','unchanged','uncertain'],
    'yellow_strip_change':['shorter','longer','unchanged','uncertain']}
INSTRUCTION='''Compare the original BEFORE and AFTER images. The auxiliary pair
depicts the SAME two observations, possibly with softened background or outline
and number annotations. Marks are not game objects. Use the originals for colors
and static references. No coordinates or exact pixel counts are requested.
Report the orange-blue group's movement, the white cross's position relative to
the orange-blue group in AFTER, whether the white cross moved, and whether the
yellow strip became shorter, longer, or stayed the same. Use uncertain when you
cannot determine an answer. Appearance names do not establish any game roles.
Return one submit_coarse_comparison call with a short visual evidence summary.'''


def schema():
    fields={k:{'type':'string','enum':v} for k,v in OPTIONS.items()}
    fields['summary']={'type':'string','maxLength':140}
    return {'type':'object','additionalProperties':False,'properties':fields,'required':list(fields)}


def make_payload(case,arm):
    p=payload(views(case,arm)[0]);p['messages'][0]['content']=INSTRUCTION
    p['messages'][1]['content'][-1]['text']='Compare observed directions, relative positions, and strip length. Do not assign game roles.'
    p['max_tokens']=220
    p['tools']=[{'type':'function','function':{'name':'submit_coarse_comparison','description':'Submit coordinate-free visual comparisons.','parameters':schema()}}]
    return p


def parse(response):
    c=response['choices'][0]
    if c.get('finish_reason')=='length':raise ValueError('truncated response')
    calls=c['message'].get('tool_calls',[])
    if len(calls)!=1 or calls[0]['function']['name']!='submit_coarse_comparison':raise ValueError('unexpected tool')
    value=calls[0]['function']['arguments'];value=json.loads(value) if isinstance(value,str) else value
    validate(value,schema());return value


def score(v,name):
    expected={'orange_blue_change':'up' if name=='forward' else 'down' if name=='reverse' else 'unchanged',
              'white_cross_relative_to_orange_blue':'above_left','white_cross_change':'unchanged',
              'yellow_strip_change':'shorter' if name=='forward' else 'longer' if name=='reverse' else 'unchanged'}
    result={k:v.get(k)==x for k,x in expected.items()};result['all_correct']=all(result.values())
    return result


def prepare(source,output):
    output.mkdir(parents=True,exist_ok=False);cases=json.loads((source/'cases.json').read_text())
    jobs=[{'name':c['name'],'arm':arm,'payload':make_payload(c,arm)} for c in cases for arm in ARMS]
    for j in jobs:j['payload_digest']=digest(j['payload'])
    save_json(output/'jobs.json',jobs)
    save_json(output/'experiment.json',{'created_at':now(),'source':str(source),'jobs_digest':digest(jobs),
        'status':'exploratory follow-up; motivated by failed coordinate localization in primary warmup/early measurements',
        'repeats':2,'warmups_per_input':1,'wall_budget_seconds':20,'max_tokens':220,
        'scope':'Same 15 imagesets. No coordinates or counts requested. Does not replace primary scores.',
        'truth':{'forward':['up','above_left','unchanged','shorter'],
                 'reverse':['down','above_left','unchanged','longer'],
                 'same_frame':['unchanged','above_left','unchanged','unchanged']}})
    (output/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    print('Prepared',len(jobs),'coarse probe inputs',flush=True)


def run(output,base):
    assert not (output/'measurements.jsonl').exists()
    jobs=json.loads((output/'jobs.json').read_text())
    save_json(output/'environment.json',{'server':http(base.removesuffix('/v1')+'/props'),
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'jobs_digest':digest(jobs)})
    with (output/'measurements.jsonl').open('w') as log:
        for rep in (-1,0,1):
            order=list(jobs);random.Random(10700+rep).shuffle(order)
            for job in order:
                row={k:v for k,v in job.items() if k!='payload'};row.update(repeat=rep,warmup=rep==-1,started_at=now())
                start=time.monotonic()
                try:
                    r=http(base+'/chat/completions',job['payload'],20);row['response']=r;row['parsed']=parse(r);row['valid']=True
                except Exception as exc:row.update(valid=False,error=f'{type(exc).__name__}: {exc}')
                row['seconds']=time.monotonic()-start;row['score']=score(row.get('parsed',{}),job['name'])
                row['within_budget']=row['seconds']<=20
                log.write(json.dumps(row,ensure_ascii=False)+'\n');log.flush()
                print(rep,job['name'],job['arm'],round(row['seconds'],2),row['score'],row.get('error',''),flush=True)
    save_json(output/'server-after.json',http(base.removesuffix('/v1')+'/props'));report(output)


def report(output):
    rows=[r for r in read_lines(output/'measurements.jsonl') if not r['warmup']];summary=[];unique={}
    for name in ('forward','reverse','same_frame'):
        for arm in ARMS:
            rs=[r for r in rows if (r['name'],r['arm'])==(name,arm)]
            if not rs:continue
            summary.append({'case':name,'arm':arm,'n':len(rs),'valid':sum(r['valid'] for r in rs),
                **{k:sum(r['score'][k] for r in rs) for k in [*OPTIONS,'all_correct']},
                'seconds_median':statistics.median(r['seconds'] for r in rs)})
    for r in rows:
        v=r.get('parsed',{'error':r.get('error')});key=digest(v)
        unique.setdefault(key,{'id':key,'output':v,'members':[]})['members'].append({k:r[k] for k in ('name','arm','repeat','score')})
    save_json(output/'summary.json',summary);save_json(output/'review-candidates.json',list(unique.values()))
    lines=['# 座標なしの追加診断（探索的）','','|ケース|条件|n|移動方向|十字の相対位置|十字の不変|帯の伸縮|全項目|秒|',
           '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for s in summary:
        lines.append('|'+ '|'.join(str(s[k]) for k in ['case','arm','n',*OPTIONS,'all_correct'])+f'|{s["seconds_median"]:.2f}|')
    (output/'report.md').write_text('\n'.join(lines)+'\n')
    (output/'comparison.html').write_text('<!doctype html><meta charset="utf-8"><title>座標なし診断</title><style>pre{white-space:pre-wrap}</style>'+''.join('<pre>'+html.escape(json.dumps(r,ensure_ascii=False,indent=2))+'</pre><hr>' for r in unique.values()))
    print('Summarized',len(rows),'coarse measurements;',len(unique),'unique responses',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['prepare','run','report'])
    p.add_argument('--source',type=Path,default=Path('outputs/visual-emphasis-comparison-20260927'))
    p.add_argument('--output',type=Path,required=True);p.add_argument('--base',default='http://127.0.0.1:8080/v1')
    a=p.parse_args()
    if a.command=='prepare':prepare(a.source,a.output)
    elif a.command=='run':run(a.output,a.base)
    else:report(a.output)


if __name__=='__main__':main()
