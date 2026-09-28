"""Fixed-frame 2x2 visual recognition experiment; never sends game actions.

Only run contacts an explicitly supplied local inference endpoint. Prepare saves
all input bytes before measurement. Semantic scoring is a separate visual review.
"""
from __future__ import annotations

import argparse
import base64
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import random
import statistics
import sys
import time
import urllib.request

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.benchmark_parallel import restore_images, digest, read_lines


ARMS = ('original_full', 'original_crops', 'grounded_full', 'grounded_crops')
GROUNDED = '''Inspect the CURRENT board to answer the supplied question. First distinguish
the relevant visible parts or groups by appearance and spatial relations, including
small contrasting parts. Describe observed appearance before proposing any role.
Do not replace the parts with a single description of the whole pixel grid.
Roles may be unknown. A familiar shape is not proof of its function. For each role
hypothesis state its visual or action evidence and what is still unverified.
When prior interpretations and action results are supplied, compare the actual
parts before and after; revise unsupported roles. Repeated descriptions are not
new evidence. Separate visible change from goal achievement. Do not invent rules.
Use at most four shared concepts and six relevant instances or groups, without
coordinates. State a plausible goal and a specific uncertainty an interaction
could test. Choose backchain for prerequisites or ground for a one-action probe.
Submit one concise submit_understanding call. You are not writing an execution plan.'''


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


def encode(image):
    data=io.BytesIO(); image.save(data,format='PNG')
    return {'type':'image_url','image_url':{'url':'data:image/png;base64,'+base64.b64encode(data.getvalue()).decode()}}


def decode(part):
    return Image.open(io.BytesIO(base64.b64decode(part['image_url']['url'].split(',',1)[1]))).convert('RGB')


def crop_sheet(part):
    """Fixed overlapping quadrants, independent of game semantics and responses.

    Source recording renderer has origin (32,44), scale 6 and a 64x64 board.
    Use nearest-neighbor enlargement; preserve the original input separately.
    """
    source=decode(part)
    if source.size!=(428,452):raise ValueError('unsupported source rendering geometry')
    board=source.crop((32,44,416,428)).resize((64,64),getattr(Image, 'Resampling', Image).NEAREST)
    windows=[(0,0,36,36),(28,0,64,36),(0,28,36,64),(28,28,64,64)]
    sheet=Image.new('RGB',(744,784),'#18202b');draw=ImageDraw.Draw(sheet)
    for i,box in enumerate(windows):
        x=(i%2)*372+6;y=(i//2)*392+26
        draw.text((x,y-20),f'VIEW {i+1}: x={box[0]}..{box[2]-1}, y={box[1]}..{box[3]-1}',fill='white')
        sheet.paste(board.crop(box).resize((360,360),getattr(Image, 'Resampling', Image).NEAREST),(x,y))
    return encode(sheet)


def variant(payload, arm):
    result=deepcopy(payload)
    if arm.startswith('grounded'):
        result['messages'][0]['content']=GROUNDED
        target=result['tools'][0]['function']['parameters']['$defs']['Target']
        target['properties']['role_evidence']={'type':'string','maxLength':180,
            'description':'Observed support for the role, or unknown when not established.'}
        target['properties']['unverified']={'type':'string','maxLength':180,
            'description':'What remains unverified about this role.'}
        target['required']+=['role_evidence','unverified']
    if arm.endswith('crops'):
        for msg in result['messages']:
            if not isinstance(msg.get('content'),list):continue
            parts=[]
            for part in msg['content']:
                parts.append(part)
                if part['type']=='image_url':
                    parts.extend([{'type':'text','text':'HOST detail views of the SAME image above. Overlapping quadrants, not additional objects or new observations.'},crop_sheet(part)])
            msg['content']=parts
    result.update(seed=123,temperature=0,max_tokens=1000,id_slot=0,cache_prompt=True)
    return result


def prepare(source, output):
    output.mkdir(parents=True,exist_ok=False)
    cases=[]
    for directory in sorted(source.glob('*/cognition')):
        rows=[r for p in directory.glob('*.requests.jsonl') for r in read_lines(p)]
        first=next(r for r in rows if r['work']=='understand')
        payload=restore_images(first['request'],directory)
        case={'name':directory.parent.name+'-initial','phase':'initial','game':directory.parent.name,
              'source_request':first['request_sha256'],'payload':payload}
        cases.append(case)
        if directory.parent.name.startswith('ls20'):
            follow=next(r for r in rows if r['work']=='reconcile')
            temporal=restore_images(follow['request'],directory)
            context=json.loads(temporal['messages'][1]['content'][-1]['text'])
            # Remove downstream goal/plan/completion claims; retain the fixed prior
            # understanding and only the acknowledged action evidence available then.
            initial_context=json.loads(payload['messages'][1]['content'][-1]['text'])
            initial_context.update(observation_id=follow['observation_id'],
                question='Which visible parts changed after the acknowledged action, and which earlier role hypotheses should be revised?',
                prior_understanding=context['understanding'],last_result=context['last_result'],
                reason='Compare the supplied before/current observations and reassess the earlier interpretation.')
            initial_context['remaining_actions']=context['remaining_actions']
            initial_context['last_result']={k:v for k,v in context['last_result'].items()
                                            if k not in ('prediction','skill','binding')}
            initial_context.pop('memory_brief',None)
            tp=deepcopy(payload)
            tp['messages'][1]['content']=temporal['messages'][1]['content'][:-1]+[
                {'type':'text','text':json.dumps(initial_context,separators=(',',':'))}]
            tp['tools'][0]['function']['parameters']['properties']['observation_id']['enum']=[follow['observation_id']]
            cases.append({'name':directory.parent.name+'-transition','phase':'transition','game':directory.parent.name,
                          'source_request':follow['request_sha256'],'payload':tp})
    if len(cases)!=4:raise ValueError('expected three initial cases and one transition')
    prepared=[]
    for case in cases:
        for arm in ARMS:
            row={k:v for k,v in case.items() if k!='payload'}
            row.update(arm=arm,payload=variant(case['payload'],arm))
            row['payload_digest']=digest(row['payload']);prepared.append(row)
    save_json(output/'cases.json',prepared)
    save_json(output/'experiment.json',{'created_at':datetime.now(timezone.utc).isoformat(),
        'source':str(source),'case_digest':digest(prepared),'arms':ARMS,'repeats':3,
        'scope':'Development-set fixed-frame pilot; no held-out games or live game performance.',
        'factor_crops':'Full image plus one fixed 4-quadrant detail sheet per observation.',
        'factor_grounded':'Prompt plus role_evidence/unverified schema fields; a joint format intervention.',
        'timing':'Warm server, mixed prefix cache. Log cached tokens; no cold-cache claim.',
        'semantic_review':'Assistant visual review against predeclared facts; not independent human annotation.'})
    print('Prepared',len(prepared),'requests',flush=True)


def request(base, payload=None):
    req=urllib.request.Request(base,data=None if payload is None else json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=45) as r:return json.load(r)


def validate(value, schema, root=None):
    """Validate the limited JSON-schema vocabulary used by recorded Understanding."""
    root=root or schema
    if '$ref' in schema:return validate(value,root['$defs'][schema['$ref'].split('/')[-1]],root)
    kind=schema.get('type')
    if kind=='object':
        if not isinstance(value,dict):raise ValueError('expected object')
        if not set(schema.get('required',[]))<=value.keys():raise ValueError('missing required keys')
        if schema.get('additionalProperties') is False and value.keys()-schema['properties'].keys():raise ValueError('extra keys')
        for k,v in value.items():validate(v,schema['properties'][k],root)
    elif kind=='array':
        if not isinstance(value,list):raise ValueError('expected array')
        if not schema.get('minItems',0)<=len(value)<=schema.get('maxItems',float('inf')):raise ValueError('array length')
        for v in value:validate(v,schema['items'],root)
    elif kind=='string':
        if not isinstance(value,str):raise ValueError('expected string')
        if not schema.get('minLength',0)<=len(value)<=schema.get('maxLength',float('inf')):raise ValueError('string length')
    if 'enum' in schema and value not in schema['enum']:raise ValueError('enum mismatch')


def extract(response,payload):
    choice=response['choices'][0]
    if choice.get('finish_reason')=='length':raise ValueError('truncated output')
    message=choice['message'];calls=message.get('tool_calls',[])
    if not calls:
        content=message.get('content','').strip()
        if content.startswith('<tool_call>') and content.endswith('</tool_call>'):
            calls=[{'function':json.loads(content[len('<tool_call>'):-len('</tool_call>')])}]
    if len(calls)!=1:raise ValueError('expected one completion tool')
    fn=calls[0]['function']
    if fn['name']!='submit_understanding':raise ValueError('wrong tool')
    value=fn['arguments'];value=json.loads(value) if isinstance(value,str) else value
    validate(value,payload['tools'][0]['function']['parameters'])
    return value


def run(output,base):
    cases=json.loads((output/'cases.json').read_text())
    props=request(base.removesuffix('/v1')+'/props')
    if props['total_slots']!=1:raise ValueError('expected existing single-slot server')
    if (output/'measurements.jsonl').exists():raise ValueError('refusing to overwrite measurement run')
    save_json(output/'environment.json',{'server':props,'case_digest':digest(cases),
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'started_at':datetime.now(timezone.utc).isoformat()})
    schedule=[]
    # Warm each input once, then counterbalance arms via a seeded shuffle per repetition.
    for repeat in (-1,0,1,2):
        indices=list(range(len(cases)));random.Random(7300+repeat).shuffle(indices)
        schedule.extend((repeat,index) for index in indices)
    with (output/'measurements.jsonl').open('w') as log:
        for repeat,index in schedule:
            case=cases[index];row={k:v for k,v in case.items() if k!='payload'}
            row.update(repeat=repeat,warmup=repeat==-1,started_at=datetime.now(timezone.utc).isoformat())
            start=time.monotonic()
            try:
                response=request(base+'/chat/completions',case['payload'])
                row.update(response=response,usage=response.get('usage'),timings=response.get('timings'))
                row['parsed']=extract(response,case['payload']);row['valid']=True
            except Exception as exc:
                row.update(valid=False,error=f'{type(exc).__name__}: {exc}')
            row['seconds']=time.monotonic()-start
            log.write(json.dumps(row,ensure_ascii=False)+'\n');log.flush()
            print(case['name'],case['arm'],repeat,round(row['seconds'],2),row['valid'],row.get('error',''),flush=True)
    save_json(output/'server-after.json',request(base.removesuffix('/v1')+'/props'))
    report(output)


def report(output):
    rows=[r for r in read_lines(output/'measurements.jsonl') if not r['warmup']]
    summaries=[]
    for name,arm in sorted({(r['name'],r['arm']) for r in rows}):
        group=[r for r in rows if (r['name'],r['arm'])==(name,arm)]
        summaries.append({'case':name,'arm':arm,'n':len(group),'valid':sum(r['valid'] for r in group),
            'seconds_median':statistics.median(r['seconds'] for r in group),
            'seconds_range':[min(r['seconds'] for r in group),max(r['seconds'] for r in group)],
            'output_variants':len({digest(r['parsed']) for r in group if r['valid']}),
            **{key:statistics.median((r.get('usage') or {}).get(key,0) for r in group) for key in ['prompt_tokens','completion_tokens']}})
    save_json(output/'summary.json',summaries)
    # De-duplicate for visual review but retain repeat membership and invalid outputs.
    review={}
    for row in rows:
        value=row.get('parsed',row.get('response',{}));key=digest(value)
        item=review.setdefault(key,{'id':key,'case':row['name'],'members':[],'output':value})
        item['members'].append({'arm':row['arm'],'repeat':row['repeat'],'valid':row['valid']})
    save_json(output/'review-candidates.json',list(review.values()))
    print('Summarized',len(rows),'measurements;',len(review),'unique outputs',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['prepare','run','report'])
    parser.add_argument('--source',type=Path,default=Path('outputs/history/evaluations/20260926T172310706477Z'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--base',default='http://127.0.0.1:8080/v1')
    args=parser.parse_args()
    if args.command=='prepare':prepare(args.source,args.output)
    elif args.command=='run':run(args.output,args.base)
    else:report(args.output)


if __name__=='__main__':main()
