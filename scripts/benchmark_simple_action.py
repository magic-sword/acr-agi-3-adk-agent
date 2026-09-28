"""Single factual question, no rule induction: vision vs state comparison vs retrieval.

The source of truth is the generated pixel pair. Measurement is independently
extracted with the existing heuristic. Future rules/roles are never supplied.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import random
import re
import shutil
import statistics
import sys
import time
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.instance_state import extract, compare
from scripts.benchmark_rule_inference import objects, panels
from scripts.benchmark_rule_brief import COLORS
from scripts.benchmark_recognition import encode, save_json
from scripts.benchmark_parallel import digest, read_lines
from scripts.benchmark_official_relations import sha
from scripts.benchmark_attention_selection import http, now

ARMS = ['images', 'states', 'facts']
SOURCE = Path('outputs/instance-state-20260928-final/cases.json')
SYSTEM = 'Answer the single factual question from the supplied observations. Report what was observed, not a game rule. Reply with only the option letter, or X if the evidence is insufficient.'


def options(texts, answer, index):
    others = [t for t in texts if t != answer]
    random.Random(9290+index).shuffle(others)
    others.insert(index % len(texts), answer)
    return [dict(letter=chr(65+i), text=t) for i, t in enumerate(others)], chr(65+index % len(texts))


def build_cases():
    result = []
    def add(name, group, question, choices, answer, trials, **extra):
        opts, letter = options(choices, answer, len(result))
        result.append(dict(name=name, group=group, question=question, options=opts,
            truth=letter, truth_text=answer, trials=trials, **extra))
    for i, color in enumerate([9, 12, 14, 8]):
        g = objects([[14, 16], [40, 38]], [color, [12, 9, 8, 14][i]])
        add(f'color_{i}', 'color', 'In AFTER, what color is the square in the upper-left part of the screen?',
            ['blue', 'orange', 'green', 'red'], COLORS[color],
            [dict(id='T1', action='A', before=g, after=deepcopy(g))], after_only=True)
    choices = ['blue only', 'orange only', 'both blue and orange', 'neither']
    for variant in range(4):
        amount = 1 if variant < 2 else 4
        axis = variant % 2
        for category, moving in enumerate([[0], [1], [0,1], []]):
            pos = [[14+variant,16],[40,38+variant]]; after=deepcopy(pos)
            for index in moving: after[index][axis] += amount if index == 0 else -amount
            action = ['A','WAIT','B','A'][(variant+category)%4]
            add(f'motion_{variant}_{category}', 'motion',
                f'In the observations immediately before and after {action}, which squares changed position?',
                choices, choices[category], [dict(id='T1',action=action,before=objects(pos,[9,12]),after=objects(after,[9,12]))],
                magnitude=amount, movement_class=category, after_only=False)
    patterns = [
        [[9,12,9],[12,9,12],[9,12,9]],
        [[12,12,9],[9,9,12],[9,12,12]],
        [[9,9,12],[12,9,9],[12,12,9]],
        [[12,9,12],[9,12,9],[12,9,12]]]
    names = ['upper-left panel','upper-right panel','lower-left panel','lower-right panel','none']
    for which in range(5):
        p = deepcopy(patterns)
        if which < 4: p[which][1][1] = 12 if p[which][1][1] == 9 else 9
        action = 'A' if which % 2 == 0 else 'B'
        add(f'panel_{which}', 'panel', f'Immediately after {action}, which 3-by-3 panel had an internal color change?',
            names, names[which], [dict(id='T1',action=action,before=panels(patterns),after=panels(p))], after_only=False)
    choices = ['A only','B only','both A and B','neither A nor B']
    for category, movers in enumerate([['A'],['B'],['A','B'],[]]):
        trials=[]
        for action in (['A','B'] if category % 2 == 0 else ['B','A']):
            pos=[[14,16],[40,38]]; aft=[[18,16],[40-(4 if action in movers else 0),38]]
            trials.append(dict(id=f'T{len(trials)+1}',action=action,before=objects(pos,[9,12]),after=objects(aft,[9,12])))
        add(f'binding_{category}', 'binding', 'In these recorded trials, after which button was the orange square observed to change position?',
            choices, choices[category], trials, after_only=False)
    return result


def render(grid, palette):
    im=Image.new('RGB',(64,64)); im.putdata([tuple(int(palette[str(c)][k:k+2],16) for k in [1,3,5]) for row in grid for c in row])
    return im.resize((512,512),getattr(Image,'Resampling',Image).NEAREST)


def object_name(o):
    if o['class_name']=='regular_grid_3x3':
        x,y=o['bbox'][:2]
        return ('upper' if y<32 else 'lower')+'-'+('left' if x<32 else 'right')+' panel'
    colors=sorted(map(int,o['colors']))
    return '/'.join(COLORS[c] for c in colors)+' square'


def state_text(state):
    parts=[]
    for o in state['instances']:
        text=f"{object_name(o)}: top-left ({o['bbox'][0]}, {o['bbox'][1]})"
        if o['class_name']=='regular_grid_3x3':
            text+='; cell colors by row '+json.dumps([[COLORS[c] for c in row] for row in o['pattern']],separators=(',',':'))
        parts.append(text+'.')
    return ' '.join(parts)


def measure(case):
    output=[]
    for t in case['trials']:
        b=extract(t['before'],t['id']+'b');a=extract(t['after'],t['id']+'a');cmp=compare(b,a)
        old={o['id']:o for o in b['instances']};new={o['id']:o for o in a['instances']}
        changed={e['before_id']:e for e in cmp['changes'] if e['before_id']}
        sentences=[]
        for o in b['instances']:
            e=changed.get(o['id'])
            if e is None: desc='did not change position or color'
            elif e['kind']=='moved':
                nxt=new[e['after_id']];dx=nxt['bbox'][0]-o['bbox'][0];dy=nxt['bbox'][1]-o['bbox'][1]
                desc=f'changed position by dx={dx}, dy={dy} pixels'
            elif e['kind']=='appearance_changed': desc='changed internal colors without changing position'
            else: desc='has uncertain correspondence'
            sentences.append(f'{object_name(o)} {desc}.')
        output.append(dict(id=t['id'],action=t['action'],before=b,after=a,comparison=cmp,facts=' '.join(sentences)))
    return output


def request_for(case, measured, arm, out):
    content=[]
    if len(case['trials'])>1:
        content.append(dict(type='text',text='These are separate recorded trials, each with its own BEFORE and AFTER.'))
    if any(t['action']=='WAIT' for t in case['trials']):
        content.append(dict(type='text',text='WAIT means no button was pressed during the observation interval.'))
    for t,m in zip(case['trials'],measured):
        content.append(dict(type='text',text=f"{t['id']}: recorded action {t['action']}."))
        frames=['after'] if case['after_only'] else ['before','after']
        if arm=='images':
            for frame in frames:
                content.append(dict(type='text',text=frame.upper()))
                content.append(encode(Image.open(out/f"{case['name']}-{t['id']}-{frame}.png").convert('RGB')))
        elif arm=='states' or case['after_only']:
            content.append(dict(type='text',text='Original-pixel coordinates: x increases rightward, y downward.\n'+'\n'.join(frame.upper()+': '+state_text(m[frame]) for frame in frames)))
        else:
            content.append(dict(type='text',text='Program-measured changes (positive dx rightward, positive dy downward): '+m['facts']))
    question=case['question']+'\n'+'\n'.join(o['letter']+'. '+o['text'] for o in case['options'])+'\nX. insufficient evidence'
    content.append(dict(type='text',text=question))
    return dict(model='qwen3-vl-4b-instruct',messages=[dict(role='system',content=SYSTEM),dict(role='user',content=content)],
        temperature=0,max_tokens=16,stream=False,cache_prompt=False,seed=290928)


def parse(answer,case):
    s=answer.strip()
    match=re.fullmatch(r'([A-EX])[.)]?',s)
    if match is None: raise ValueError('not a single option letter')
    value=match.group(1)
    if value not in [o['letter'] for o in case['options']]+['X']: raise ValueError('option outside question')
    return value


def prepare(out):
    out.mkdir(parents=True,exist_ok=False);(out/'sources').mkdir()
    palette=json.loads(SOURCE.read_text())[0]['palette'];cases=build_cases();measured={};jobs=[]
    for c in cases:
        measured[c['name']]=measure(c)
        for t in c['trials']:
            for frame in ['before','after']:render(t[frame],palette).save(out/f"{c['name']}-{t['id']}-{frame}.png")
        for arm in ARMS:
            p=request_for(c,measured[c['name']],arm,out)
            jobs.append(dict(id=len(jobs),case=c['name'],arm=arm,payload=p,payload_digest=digest(p)))
    save_json(out/'cases.json',cases);save_json(out/'measured.json',measured);save_json(out/'jobs.json',jobs)
    paths=[Path(__file__),Path('scripts/instance_state.py'),Path('scripts/benchmark_rule_inference.py')]
    for p in paths:shutil.copyfile(p,out/'sources'/p.name)
    save_json(out/'plan.json',dict(created_at=now(),requests=len(jobs),warmups=3,arms=ARMS,
        cases_digest=digest(cases),measured_digest=digest(json.loads((out/'measured.json').read_text())),jobs_digest=digest(jobs),sources={str(p):sha(p) for p in paths},
        images={p.name:sha(p) for p in out.glob('*.png')},
        primary='Single factual choice. Score separately: static color identification 4, changed-position set 16 (8 micro/8 larger or static), changed panel 5, two-trial action binding 4. Motion gold classes balanced blue-only/orange-only/both/neither. No coordinate/direction output required.',
        design='3 paired sources: 512px nearest-neighbor unannotated images; compact per-frame numeric/color states; program-measured change facts. Same simple system/question/options per case. Only AFTER for color calibration, two frames for motion/panel, four for binding. English short prompts, greedy, no grammar or JSON, 16 output tokens. Shuffled jobs and constrained-random balanced option positions. No post-result tuning.',
        limitations='New small synthetic controlled scenes, no real game/role/causal/next-action score. Shape/classes restricted to squares and regular panels. States/facts use a working heuristic extractor. Facts arm intentionally gives the measured answer facts: it tests reading/selection, not visual understanding. Different source lengths/compute. Single output per condition, no stochastic confidence estimate. Earlier rule-induction trials differ in difficulty, so scores are not an isolated prompting gain.'))
    print('Prepared',len(cases),'questions,',len(jobs),'requests')


def run(out):
    load=lambda n:json.loads((out/n).read_text())
    plan,jobs=load('plan.json'),load('jobs.json');cases={c['name']:c for c in load('cases.json')}
    assert digest(jobs)==plan['jobs_digest']
    for p,h in plan['sources'].items():assert sha(p)==h
    for p,h in plan['images'].items():assert sha(out/p)==h
    assert not (out/'responses.jsonl').exists();save_json(out/'server-before.json',http('http://127.0.0.1:8080/props'))
    warm=[next(j for j in jobs if j['arm']==arm) for arm in ARMS]
    order=jobs.copy();random.Random(925321).shuffle(order)
    with (out/'responses.jsonl').open('w') as log:
        for j,iswarm in [(j,True) for j in warm]+[(j,False) for j in order]:
            r={k:j[k] for k in ['id','case','arm','payload_digest']};r.update(warmup=iswarm,started_at=now());start=time.monotonic()
            try:
                response=http('http://127.0.0.1:8080/v1/chat/completions',j['payload'],180)
                choice=response['choices'][0];r.update(response=response,answer=choice['message']['content'])
                r['choice']=parse(r['answer'],cases[j['case']]);r['valid']=choice['finish_reason']=='stop'
            except Exception as e:r.update(valid=False,error=f'{type(e).__name__}: {e}')
            r['seconds']=time.monotonic()-start;log.write(json.dumps(r,ensure_ascii=False)+'\n');log.flush()
            print(j['id'],j['case'],j['arm'],r.get('choice'),r['valid'],round(r['seconds'],2),r.get('error',''),flush=True)
    save_json(out/'server-after.json',http('http://127.0.0.1:8080/props'));save_json(out/'health-after.json',http('http://127.0.0.1:8080/health'))


def analyze(out):
    cases={c['name']:c for c in json.loads((out/'cases.json').read_text())};rows=[r for r in read_lines(out/'responses.jsonl') if not r['warmup']]
    assert len(rows)==len(cases)*len(ARMS)
    scored=[]
    for r in rows:
        c=cases[r['case']];choice=r.get('choice');answer_text=next((o['text'] for o in c['options'] if o['letter']==choice),'insufficient evidence' if choice=='X' else 'invalid')
        scored.append({k:r.get(k) for k in ['case','arm','valid','choice','answer','error','seconds']}|dict(group=c['group'],truth=c['truth'],truth_text=c['truth_text'],answer_text=answer_text,correct=bool(r['valid'] and choice==c['truth']),magnitude=c.get('magnitude'),movement_class=c.get('movement_class'),action=c['trials'][0]['action'],prompt_tokens=r.get('response',{}).get('usage',{}).get('prompt_tokens')))
    summary=[]
    for arm in ARMS:
        rr=[r for r in scored if r['arm']==arm]
        def score(rows):return dict(correct=sum(r['correct'] for r in rows),total=len(rows))
        summary.append(dict(arm=arm,**score(rr),valid=sum(r['valid'] for r in rr),groups={g:score([r for r in rr if r['group']==g]) for g in ['color','motion','panel','binding']},
            motion_by_magnitude={str(n):score([r for r in rr if r['group']=='motion' and r['magnitude']==n]) for n in [1,4]},
            actual_movement=score([r for r in rr if r['group']=='motion' and r['movement_class']!=3]),
            static_motion=score([r for r in rr if r['group']=='motion' and r['movement_class']==3]),
            wait_motion=score([r for r in rr if r['group']=='motion' and r['action']=='WAIT']),
            median_seconds=statistics.median(r['seconds'] for r in rr),max_prompt_tokens=max(r['prompt_tokens'] or 0 for r in rr)))
    save_json(out/'summary.json',dict(arms=summary,rows=scored));print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','run','analyze']);p.add_argument('--output',required=True,type=Path)
    a=p.parse_args();globals()[a.command](a.output)
