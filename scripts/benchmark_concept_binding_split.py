"""Separate visual class discovery from oracle-description candidate binding.

Uses frozen SAM outputs; no game actions or runtime behavior changes. Prepare
freezes questions and gold before run. Free-form class outputs need an explicit
assistant semantic review; binding is scored from independently annotated pixels.
"""
import argparse
import ast
import base64
from collections import Counter
from datetime import datetime, timezone
import hashlib
import html
import io
import json
import math
from pathlib import Path
import random
import statistics
import sys
import time
import urllib.request
from urllib.parse import urlparse

from PIL import Image, ImageColor, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agent.cognition.geometry import extract
from agent.cognition.perception import COLORS

SOURCE = ROOT / 'outputs/sam-proposals-20260928'
PALETTE = ROOT / 'outputs/instance-state-20260928-final/cases.json'
DISCOVERY = ('joint_4', 'concept_only_4', 'concept_only_12')


def save(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


def http(base, route, payload=None):
    req = urllib.request.Request(base + route,
        data=json.dumps(payload).encode() if payload is not None else None,
        headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=60) as response:
        return json.load(response)


def encode(image):
    data = io.BytesIO(); image.save(data, format='PNG')
    return {'type':'image_url', 'image_url':{'url':'data:image/png;base64,' + base64.b64encode(data.getvalue()).decode()}}


def render(grid, palette):
    board = Image.new('RGB', (64,64))
    board.putdata([ImageColor.getrgb(palette[str(c)]) for row in grid for c in row])
    board = board.resize((512,512), getattr(Image, 'Resampling', Image).NEAREST)
    image = Image.new('RGB', (552,552), '#18202b'); image.paste(board, (24,24))
    draw = ImageDraw.Draw(image)
    for n in range(0,64,8):
        draw.text((24+n*8,8), str(n), fill='white')
        draw.text((3,24+n*8), str(n), fill='white')
    return image


def candidate_inventory(c, detection):
    """Same initial box union as runtime: program plus top-128 SAM, exact dedup."""
    boxes = {}
    for obj in extract(c['grid'], 'a')['instances']:
        boxes.setdefault(tuple(obj['bbox']), []).append('program')
    for box in detection['boxes'][:128]:
        native = (max(0,math.floor(box[0]*.064)), max(0,math.floor(box[1]*.064)),
                  min(64,math.ceil(box[2]*.064))-1, min(64,math.ceil(box[3]*.064))-1)
        if 'sam' not in boxes.setdefault(native, []): boxes[native].append('sam')
    result = []
    for i, (box, sources) in enumerate(boxes.items()):
        x,y,r,b = box
        colors = sorted({c['grid'][yy][xx] for yy in range(y,b+1) for xx in range(x,r+1)})
        result.append(dict(id=f'c{i+1:03}', bbox=list(box), colors=colors, sources=sources))
    return result


def eligible(candidate, gold):
    """Tighter than proposal recall: >=90% support, IoU >=.5, area <=2x."""
    x,y,r,b = candidate['bbox']; gx,gy,gr,gb = gold['bbox']
    area = (r-x+1)*(b-y+1); ga = (gr-gx+1)*(gb-gy+1)
    intersection = max(0,min(r,gr)-max(x,gx)+1)*max(0,min(b,gb)-max(y,gy)+1)
    covered = sum(x<=xx<=r and y<=yy<=b for xx,yy in gold['support'])
    return covered/len(gold['support']) >= .9 and area <= 2*ga and intersection/(area+ga-intersection) >= .5


def gold_classes(c):
    """Class-level annotations, independent of locations and rotated instances."""
    if c['group'] == 'single_color':
        return [dict(id=g['id'], description=COLORS[g['colors'][0]]+' '+g['kind'], members=[g['id']]) for g in c['gold']]
    if c['origin'] == 'synthetic':
        descriptions = {'two_color_rectangle':'Blue and orange two-color square block, two adjoining halves.',
            'small_cross':'Small white and light gray plus-shaped cross.',
            'large_cross':'Blue and orange plus-shaped cross.',
            'two_color_L':'Blue and orange L-shaped object.'}
        return [dict(id='class1', description=descriptions[c['group']], members=[g['id'] for g in c['gold']])]
    definitions = {
        'ls20': [('Black square containing a blue pattern.', ['g1']),
                 ('Small orange-and-blue two-color block.', ['g2']),
                 ('Thin yellow horizontal bar.', ['g3']),
                 ('Small solid red square.', ['g4','g5','g6']),
                 ('White and light gray plus-shaped cross.', ['g7'])],
        'vc33': [('Thin pink horizontal line.', ['g1']), ('Solid blue square.', ['g2','g4']),
                 ('Narrow yellow vertical rectangle.', ['g3']),
                 ('Long mostly black horizontal bar with a yellow segment.', ['g5'])],
        'ft09': [('Solid blue square.', ['g1']), ('Solid red square.', ['g2','g4']),
                 ('Thin orange horizontal line.', ['g3']),
                 ('Square panel with a white and gray internal pattern (may include a red border).', ['g5'])]}
    return [dict(id=f'class{i+1}', description=d, members=m) for i,(d,m) in enumerate(definitions[c['name']])]


def target_query(c, g):
    if c['origin'] == 'synthetic':
        description = next(k['description'] for k in c['classes'] if g['id'] in k['members'])
        return description+' The individual whole object in the '+g['location']+' quadrant.'
    overrides = {('ls20','g4'):'The leftmost of the three small red squares along the bottom.',
                 ('ls20','g5'):'The middle of the three small red squares along the bottom.',
                 ('ls20','g6'):'The rightmost of the three small red squares along the bottom.',
                 ('vc33','g4'):'The blue square at the right edge immediately above the long black horizontal bar.',
                 ('ft09','g4'):'The rightmost red square in the topmost row.',
                 ('ft09','g5'):'The entire square panel with a white and gray internal pattern and red border, in the center of the upper-left group of nine squares.'}
    return overrides.get((c['name'],g['id']), g['kind'])


def inventory_text(candidates):
    return 'Candidate rows [id, inclusive bbox, color IDs, sources]:\n'+json.dumps(
        [[o['id'],o['bbox'],o['colors'],o['sources']] for o in candidates],separators=(',',':'))+\
        '\nColor IDs: '+json.dumps(dict(enumerate(COLORS)))+\
        '\nCoordinates: x rightward, y downward, 0..63. Regions overlap and may include background or fragments.'


def runtime_instruction():
    tree = ast.parse((ROOT/'agent/cognition/tasks.py').read_text())
    assignment = next(n for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t,ast.Name) and t.id=='INSTRUCTIONS' for t in n.targets))
    return ast.literal_eval(assignment.value)['understand']


def prepare(out):
    out.mkdir(parents=True, exist_ok=False); (out/'sources').mkdir()
    all_cases = json.loads((SOURCE/'cases.json').read_text())
    cases = [c for c in all_cases if c['name'].endswith('-0') or c['group']=='real']
    detections = {r['case']:r for r in map(json.loads,(SOURCE/'sam-results.jsonl').read_text().splitlines()) if r['arm']=='sam_standard'}
    palette = json.loads(PALETTE.read_text())[0]['palette']; jobs = []
    for c in cases:
        c['classes'] = gold_classes(c); c['candidates'] = candidate_inventory(c,detections[c['name']])
        c['binding'] = [dict(gold_id=g['id'], query=target_query(c,g),
            acceptable=[o['id'] for o in c['candidates'] if eligible(o,g)]) for g in c['gold']]
        image = render(c['grid'],palette); image.save(out/(c['name']+'.png')); encoded = encode(image)
        inventory = inventory_text(c['candidates'])
        for arm in DISCOVERY:
            cap = 12 if arm.endswith('12') else 4
            if arm == 'joint_4':
                system = runtime_instruction()
                prompt = ('Question: What useful target relation or uncertainty should be investigated on this board?\n'
                    'Return JSON with concepts (1..4 entries, each name and description), targets (1..6 entries, '
                    'each concept, appearance, role_hypothesis, relations, candidate_refs), observed, goal_hypothesis, '
                    'causal_hypotheses, question, next (backchain or ground), observation_id (current). '
                    'candidate_refs must name current candidates, or [] if unresolved. Role and goal claims are hypotheses.')
            else:
                system = ('Describe visible object TYPES using appearance only. Similar copies share a type; '
                    'location and rotation need not create a new type. Keep coherent multicolor objects together. '
                    'No game roles, goals, causal guesses, candidate assignments, or individual-instance list. '
                    'Do not make a generic scene/grid class in place of the visible object types. Return JSON only.')
                prompt = f'Return {{"concepts":[{{"name":"short type name","description":"visible distinguishing appearance"}}]}}. At most {cap} types. Describe the foreground object types visible in this board.'
            payload = dict(model='qwen3-vl-4b-instruct', temperature=0, seed=929610, max_tokens=1000,
                cache_prompt=False, stream=False, response_format={'type':'json_object'},
                messages=[dict(role='system',content=system),dict(role='user',content=[encoded,dict(type='text',text=prompt+'\n'+inventory)])])
            jobs.append(dict(id=len(jobs),case=c['name'],stage='discovery',arm=arm,payload=payload,payload_digest=digest(payload)))
        controls = [dict(gold_id='absent',query='The solid magenta triangular object.',acceptable=[])]
        if c['origin'] != 'synthetic':
            controls.append(dict(gold_id='role_unknown',query='The player controlled by directional buttons. No action history or established role is available.',acceptable=[]))
        for target in c['binding'] + controls:
            for permutation in range(2):
                candidates = c['candidates'].copy(); random.Random(929620+permutation).shuffle(candidates)
                system = ('Bind ONE supplied appearance description to a measured candidate. Choose the tight region '
                    'that represents the ENTIRE described object, including all its colors. Do not select a fragment '
                    'or a broad background/group region. Multiple equivalent tight boxes are allowed: choose any one. '
                    'Choose X when the target is absent, only fragments exist, the description is ambiguous between '
                    'distinct objects, or a role cannot be established. No game rules or actions are known. '
                    'Output only one candidate ID or X.')
                payload = dict(model='qwen3-vl-4b-instruct',temperature=0,seed=929610,max_tokens=16,cache_prompt=False,stream=False,
                    grammar='root ::= '+' | '.join(json.dumps(o['id']) for o in candidates)+' | "X"',
                    messages=[dict(role='system',content=system),dict(role='user',content=[encoded,
                        dict(type='text',text='Target: '+target['query']+'\n'+inventory_text(candidates))])])
                jobs.append(dict(id=len(jobs),case=c['name'],stage='binding',arm='oracle_description',
                    gold_id=target['gold_id'],acceptable=target['acceptable'],permutation=permutation,
                    payload=payload,payload_digest=digest(payload)))
    save(out/'cases.json',cases); save(out/'jobs.json',jobs)
    paths = [Path(__file__),ROOT/'agent/cognition/geometry.py',ROOT/'agent/cognition/perception.py',
             ROOT/'agent/cognition/tasks.py',ROOT/'agent/cognition/state.py']
    for p in paths: (out/'sources'/p.name).write_bytes(p.read_bytes())
    save(out/'plan.json',dict(created=datetime.now(timezone.utc).isoformat(),cases=len(cases),requests=len(jobs),
        cases_digest=digest(cases),jobs_digest=digest(jobs),
        sources={str(p):sha(p) for p in paths},inputs={str(p):sha(p) for p in [SOURCE/'cases.json',SOURCE/'sam-results.jsonl',PALETTE]},
        images={p.name:sha(p) for p in out.glob('*.png')},
        discovery='Class-level recall, not individual object count. 5 synthetic boards (8 annotated classes) plus 3 real boards (13 partially annotated classes). Main comparison joint_4 versus concept_only_4 controls output class cap. concept_only_12 diagnoses capacity limits. Same image and complete candidate index. No gold names supplied. Joint is a runtime-like isolated JSON task using actual instruction, not a full agent replay or exact tool/schema call.',
        rubric='Each gold class recalled only if a concepts entry explicitly describes its identifying colors and coherent whole shape/pattern, without contradicting it. Synonyms, rotation and omitted size accepted. Location not required. Generic block/grid/colored shape insufficient for a specific cross, L, two-color layout, or internally patterned panel. Duplicate type entries recall a class only once. Match entries individually to at most one gold class. Record broader/partial mentions separately; targets do not rescue missing concepts. Fully annotated synthetic scenes permit unmatched output counts; real annotations only support partial recall. Review by implementing assistant, not independent human.',
        binding='37 individual targets with oracle appearance and disambiguating coarse location, plus 8 absent and 3 unknown-role controls. All candidates visible. Two index permutations; primary first permutation, secondary consistency. >=90% annotated support, bbox IoU >=.5, area <=2x gold bbox, no padding. Any eligible candidate accepted; equivalent regions are not distinct physical objects. Report candidate availability separately and conditional accuracy only where an eligible candidate exists.',
        limits='Small reused scenes, one deterministic discovery output, correlated instances, manually authored class taxonomy and oracle queries. Not class learning across games, role discovery, temporal identity, or gameplay. No evidence yet that crop quizzes improve discovery. Runtime remains unchanged.'))
    print(json.dumps(dict(cases=len(cases),requests=len(jobs),classes=sum(len(c['classes']) for c in cases),
        targets=sum(len(c['binding']) for c in cases),available=sum(bool(t['acceptable']) for c in cases for t in c['binding']))))


def verify(out):
    plan = json.loads((out/'plan.json').read_text()); cases = json.loads((out/'cases.json').read_text()); jobs = json.loads((out/'jobs.json').read_text())
    assert digest(cases)==plan['cases_digest'] and digest(jobs)==plan['jobs_digest']
    for p,h in {**plan['sources'],**plan['inputs']}.items(): assert sha(Path(p))==h, p
    for p,h in plan['images'].items(): assert sha(out/p)==h, p
    for j in jobs: assert digest(j['payload'])==j['payload_digest']
    return plan,cases,jobs


def run(out, base):
    parsed = urlparse(base); assert parsed.scheme=='http' and parsed.hostname in ('localhost','127.0.0.1')
    _,_,jobs = verify(out)
    assert not (out/'responses.jsonl').exists(), 'Use a new output directory; no silent repeats.'
    save(out/'server-before.json',http(base,'/props'))
    random.Random(929630).shuffle(jobs); started = time.monotonic()
    with (out/'responses.jsonl').open('x') as f:
        for i,j in enumerate(jobs):
            row = {k:v for k,v in j.items() if k not in ('payload','acceptable')}; start = time.monotonic()
            try:
                response = http(base,'/v1/chat/completions',j['payload'])
                row.update(response=response,answer=response['choices'][0]['message']['content'])
            except Exception as exc:
                row.update(error=f'{type(exc).__name__}: {exc}',answer='')
            row['seconds'] = time.monotonic()-start; f.write(json.dumps(row)+'\n'); f.flush()
            print(i+1,'/',len(jobs),j['case'],j['stage'],j['arm'],round(row['seconds'],2),row.get('error',''),flush=True)
    save(out/'server-after.json',http(base,'/props')); save(out/'health-after.json',http(base,'/health'))
    print('Completed seconds',round(time.monotonic()-started,2),flush=True)


def analyze(out):
    plan,cases,jobs = verify(out); bycase={c['name']:c for c in cases}; byjob={j['id']:j for j in jobs}
    rows = list(map(json.loads,(out/'responses.jsonl').read_text().splitlines()))
    assert Counter(r['id'] for r in rows)==Counter(j['id'] for j in jobs)
    binding=[]; discovery=[]
    for r in rows:
        j=byjob[r['id']]; c=bycase[j['case']]
        if r['stage']=='discovery':
            try:
                concepts=json.loads(r['answer'])['concepts']
                valid=isinstance(concepts,list) and 1<=len(concepts)<=(12 if j['arm'].endswith('12') else 4) and all(isinstance(v,dict) and all(isinstance(v.get(k),str) and v[k].strip() for k in ['name','description']) for v in concepts)
            except (ValueError,KeyError,TypeError): concepts=[];valid=False
            valid=valid and not r.get('error') and r.get('response',{}).get('choices',[{}])[0].get('finish_reason')=='stop'
            discovery.append(dict(id=r['id'],case=r['case'],arm=r['arm'],valid=bool(valid),concepts=concepts,seconds=r['seconds']))
        else:
            control=j['gold_id'] in ('absent','role_unknown'); acceptable=j['acceptable']; answer=r['answer'].strip()
            valid=answer in ['X']+[o['id'] for o in c['candidates']] and not r.get('error')
            binding.append(dict(id=r['id'],case=r['case'],origin=c['origin'],gold_id=j['gold_id'],
                permutation=j['permutation'],control=control,available=bool(acceptable),answer=answer,valid=bool(valid),
                correct=bool(valid and (answer in acceptable if acceptable else answer=='X')),
                acceptable=acceptable,seconds=r['seconds']))
    def metrics(rr):
        primary=[r for r in rr if r['permutation']==0]; positives=[r for r in primary if not r['control']]
        available=[r for r in positives if r['available']]; controls=[r for r in primary if r['control']]
        return dict(targets=len(positives),candidate_available=sum(r['available'] for r in positives),
            bound_correct=sum(r['correct'] for r in available),binding_denominator=len(available),
            missing_candidate_abstentions=sum(r['correct'] for r in positives if not r['available']),
            control_correct=sum(r['correct'] for r in controls),controls=len(controls),
            primary_invalid=sum(not r['valid'] for r in primary),
            median_seconds=statistics.median(r['seconds'] for r in rr))
    summary=dict(binding={group:metrics([r for r in binding if group=='all' or (r['origin']=='synthetic')==(group=='synthetic')]) for group in ['all','synthetic','real']},
        discovery_format={a:dict(valid=sum(r['valid'] for r in discovery if r['arm']==a),total=sum(r['arm']==a for r in discovery),median_seconds=statistics.median(r['seconds'] for r in discovery if r['arm']==a)) for a in DISCOVERY},
        errors=[dict(id=r['id'],error=r['error']) for r in rows if r.get('error')])
    pairs={}
    for r in binding:pairs.setdefault((r['case'],r['gold_id']),[]).append(r)
    summary['binding_stability']=dict(total=len(pairs),both_correct=sum(all(r['correct'] for r in rr) for rr in pairs.values()),
        answer_disagreements=sum(len({r['answer'] for r in rr})>1 for rr in pairs.values()),
        correctness_disagreements=sum(len({r['correct'] for r in rr})>1 for rr in pairs.values()))
    review_path=out/'concept-review.json'
    if review_path.exists():
        reviews=json.loads(review_path.read_text()); assert {r['id'] for r in reviews}=={r['id'] for r in discovery}
        indexed={r['id']:r for r in reviews}; summary['discovery']={}
        for a in DISCOVERY:
            summary['discovery'][a]={}
            for group in ['synthetic','real']:
                rr=[r for r in discovery if r['arm']==a and (bycase[r['case']]['origin']=='synthetic')==(group=='synthetic')]
                found=0;total=0;complete=0
                for r in rr:
                    gold=bycase[r['case']]['classes']; accepted=indexed[r['id']]['matches']
                    assert len({m['gold_id'] for m in accepted})==len(accepted)
                    assert len({m['concept_index'] for m in accepted})==len(accepted)
                    assert all(m['gold_id'] in {g['id'] for g in gold} and 0<=m['concept_index']<len(r['concepts']) for m in accepted)
                    n=len(accepted) if r['valid'] else 0;found+=n;total+=len(gold);complete+=n==len(gold)
                summary['discovery'][a][group]=dict(recalled=found,total=total,complete_scenes=complete,scenes=len(rr))
    save(out/'binding-scores.json',binding);save(out/'discovery-outputs.json',discovery);save(out/'summary.json',summary)
    page=['<!doctype html><meta charset="utf-8"><title>Concept discovery / candidate binding</title>',
        '<style>body{font:16px sans-serif;max-width:1200px;margin:auto}pre{white-space:pre-wrap}img{max-width:450px}td{vertical-align:top;border:1px solid #aaa;padding:10px}</style>',
        '<h1>Concept discovery and oracle-description binding</h1><pre>'+html.escape(json.dumps(summary,ensure_ascii=False,indent=2))+'</pre>']
    for c in cases:
        page+=['<h2>'+c['name']+'</h2><img src="'+c['name']+'.png"><h3>Annotated classes (review only)</h3><pre>'+html.escape(json.dumps(c['classes'],indent=2))+'</pre><table><tr>']
        for a in DISCOVERY:
            r=next(r for r in discovery if r['case']==c['name'] and r['arm']==a)
            page+=['<td><b>'+a+'</b><pre>'+html.escape(json.dumps(r,indent=2))+'</pre></td>']
        page+=['</tr></table><h3>Oracle-description binding, primary permutation</h3>']
        for r in binding:
            if r['case']!=c['name'] or r['permutation']!=0:continue
            j=byjob[r['id']]; query=j['payload']['messages'][1]['content'][1]['text'].split('\n')[0]
            page+=['<p>'+html.escape(query)+'<br>'+html.escape(json.dumps(r))+'</p>']
    (out/'gallery.html').write_text('\n'.join(page))
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['prepare','run','analyze'])
    parser.add_argument('--out',type=Path,required=True);parser.add_argument('--base',default='http://127.0.0.1:8080')
    args=parser.parse_args()
    if args.command=='prepare':prepare(args.out)
    elif args.command=='run':run(args.out,args.base)
    else:analyze(args.out)
