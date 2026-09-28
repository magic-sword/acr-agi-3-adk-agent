"""Frozen ablation: current view, frame pair, raw pixel delta, exact ROI links.

Raw delta never calls a matcher. All arms retain the full static candidate
inventory. No candidate is rejected for being a part. Gold only scores answers.
"""
import argparse
from collections import Counter
from copy import deepcopy
import html
import json
from pathlib import Path
import random
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agent.cognition.geometry import extract
from agent.cognition.proposal_tracking import ProposalTracker
from agent.cognition.perception import COLORS
from scripts.benchmark_concept_binding_split import save, sha, digest, http, encode, render, PALETTE

ARMS = ('current', 'frames', 'delta', 'tracked')
SYSTEM = ('Answer ONE factual question from supplied observations. Candidate regions can be parts, '
          'overlap, or contain several objects. Observed co-response is not proof of one physical '
          'object or causation. Role names are hypotheses unless tested. Independent trials start '
          'from their own BEFORE. WAIT means no button pressed. Coordinates are 0-based, x right, '
          'y down, boxes inclusive. Reply with one option letter, or X if evidence is insufficient.')


def pixel_delta(before, after):
    """Lossless horizontal runs, no object association, displacement or role."""
    if len(before) != len(after) or any(len(a) != len(b) for a, b in zip(before, after)):
        raise ValueError('different grid dimensions')
    runs = []
    for y, (old, new) in enumerate(zip(before, after)):
        x = 0
        while x < len(old):
            if old[x] == new[x]:
                x += 1
                continue
            start, a, b = x, old[x], new[x]
            x += 1
            while x < len(old) and old[x] == a and new[x] == b:
                x += 1
            runs.append([y, start, x-1, a, b])
    return dict(columns=['y', 'x_start', 'x_end_inclusive', 'before_color', 'after_color'],
                runs=runs, changed_pixels=sum(r[2]-r[1]+1 for r in runs))


def grid(objects):
    g = [[5]*64 for _ in range(64)]
    for x, y, w, h, color in objects:
        for yy in range(y, y+h):
            for xx in range(x, x+w):
                g[yy][xx] = color
    return g


def case(name, group, trials, question, choices, correct, seed, **extra):
    values = list(enumerate(choices));random.Random(seed).shuffle(values)
    options = [dict(letter=chr(65+i), text=text) for i, (_, text) in enumerate(values)]
    truth = 'X' if correct is None else next(chr(65+i) for i, (j, _) in enumerate(values) if j == correct)
    return dict(name=name, group=group, origin='new_synthetic', trials=trials, question=question,
                options=options, truth=truth, after_only=False, **extra)


def new_cases():
    cases = []
    for variant in range(4):
        base = [[10, 10, 4, 4, 9], [28, 10, 4, 4, 12], [10, 38, 4, 4, 14]]
        for o in base:o[0] += variant
        pair = [(0, 1), (0, 2), (1, 2), None][variant]
        trials = []
        for i, (action, dx, dy) in enumerate([('A', 3, 0), ('B', 0, -2), ('WAIT', 0, 0)]):
            objs = deepcopy(base)
            for k, o in enumerate(objs):
                if pair and k in pair:o[0] += dx;o[1] += dy
                elif action != 'WAIT':o[0] -= (k+1);o[1] += (k+1)
            trials.append(dict(id=f'T{i+1}', action=action, before=grid(base), after=grid(objs)))
        cases.append(case(f'coupled_translation_{variant}', 'coupled_translation', trials,
            'Which color pair has the same nonzero displacement in BOTH A and B trials, and neither moves in WAIT?',
            ['blue and orange', 'blue and green', 'orange and green', 'none of these pairs'],variant,110+variant))
        # Same change/no-change response across interventions, without motion.
        trials=[]
        for i, action in enumerate(['A','B','WAIT']):
            objs=deepcopy(base)
            for k,o in enumerate(objs):
                active=(k in pair and action in ('A','B')) if pair else False
                if (pair is None or k not in pair):active=action==['A','B','WAIT'][k]
                if active:o[4]=8
            trials.append(dict(id=f'T{i+1}',action=action,before=grid(base),after=grid(objs)))
        cases.append(case(f'coupled_color_{variant}','coupled_color',trials,
            'Which pair changes color in BOTH A and B trials, and neither changes in WAIT? Identify regions by their BEFORE color.',
            ['blue and orange','blue and green','orange and green','none of these pairs'],variant,210+variant))
        # Two linked regions are not necessarily one physical object.
        cases.append(case(f'physical_identity_{variant}','identity_limit',deepcopy(cases[-2]['trials']),
            'Do these recordings determine whether the initially blue and orange regions are parts of ONE physical object rather than separate objects?',
            ['definitely one physical object','definitely separate physical objects'],None,310+variant))
        # Appearance/motion change plus a static candidate whose role is unknown.
        objs=deepcopy(base);objs[1][0]+=variant+1
        cases.append(case(f'static_retention_{variant}','static_retention',
            [dict(id='T1',action='A',before=grid(base),after=grid(objs))],
            'What is supported about the initially blue square after this trial?',
            ['it is still present and unchanged; its gameplay role is undetermined',
             'it has been confirmed to be an exit','it disappeared','it moved with the orange square'],0,410+variant))
        base2=[[10,10,6,6,12],[42,40,4,4,9]];after=deepcopy(base2)
        if variant==0:after[0][0]+=3
        elif variant==1:after[0][4]=8
        elif variant==2:after[0][2]-=2
        cases.append(case(f'change_kind_{variant}','change_kind',
            [dict(id='T1',action='A',before=grid(base2),after=grid(after))],
            'Which visible change happened to the initially orange upper-left region?',
            ['translated right without changing color or size','changed color at the same position and size',
             'became narrower without a rigid translation','no visible change'],variant,510+variant))
        identical=[[10,10,4,4,9],[18,10,4,4,9]]
        cases.append(case(f'identical_ambiguity_{variant}','identity_limit',
            [dict(id='T1',action=['A','B','WAIT','A'][variant],before=grid(identical),after=grid(identical))],
            'Can these endpoints determine whether the two identical blue squares secretly exchanged identities between observations?',
            ['they certainly exchanged identities','they certainly did not exchange identities'],None,610+variant))
    return cases


def real_cases(sequences):
    cases=[]
    for name, indices in [('ls20',[1,2]),('vc33',[1,2]),('ft09',[1,2])]:
        s=next(s for s in sequences if s['name']==name)
        for i in indices:
            before,after=s['frames'][i-1]['grid'],s['frames'][i]['grid']
            if name=='ls20':
                question='Which observed change is supported?'
                choices=['the orange-and-blue block moved up and the bottom yellow bar changed',
                         'only the orange-and-blue block moved up; the bottom bar is unchanged',
                         'only the bottom bar changed; the orange-and-blue block is unchanged','no pixels changed']
                correct=0
            elif name=='vc33':
                question='Which observed change is supported?'
                choices=['the top pink line changed while both right-side blue squares stayed at the same positions',
                         'the lower blue square moved up','the upper blue square moved down','no pixels changed']
                correct=0
            else:
                question='Which observed change is supported?'
                choices=['no pixels changed','the top-left blue square moved','the red squares changed color','the bottom orange line disappeared']
                correct=0
            c=case(f'real_{name}_{i}','real_observation',[dict(id='T1',action='UNRECORDED',before=before,after=after)],
                   question,choices,correct,710+i+len(cases),sequence=name,frame=i)
            c['origin']='real_replay';cases.append(c)
    return cases


def inventory(grid, extra_boxes=()):
    start=time.perf_counter();raw=extract(grid,'p');boxes={tuple(o['bbox']) for o in raw['instances']}
    boxes.update(tuple(b) for b in extra_boxes)
    regions=[]
    for i,(x,y,r,b) in enumerate(sorted(boxes)):
        counts=Counter(v for row in grid[y:b+1] for v in row[x:r+1])
        regions.append([f'r{i+1}',[x,y,r,b],{str(k):v for k,v in sorted(counts.items())}])
    return dict(columns=['local_id_not_temporally_linked','bbox_inclusive','color_pixel_counts'],regions=regions,
                extraction_limited=bool(raw.get('extraction_limited'))),time.perf_counter()-start


def measured_links(before,after,initial_boxes):
    tracker=ProposalTracker();start=time.perf_counter();tracker.advance(before)
    tracker.add([[x,y,r+1,b+1] for x,y,r,b in initial_boxes],'recorded_sam')
    init=time.perf_counter()-start;start=time.perf_counter();update=tracker.advance(after);elapsed=time.perf_counter()-start
    links=[dict(before=[a[0],a[1],a[2]-1,a[3]-1],after=[b[0],b[1],b[2]-1,b[3]-1],
                delta_xy=[b[0]-a[0],b[1]-a[1]]) for row in update['links'] for a,b in [(row['before'],row['after'])]]
    return dict(links=links,unresolved=[dict(before=[*u['track']['box'][:2],u['track']['box'][2]-1,u['track']['box'][3]-1],
                reason=u['reason']) for u in update['unresolved']],
                note='Unique exact-pixel rectangle matches within radius 8; unchanged frame links are endpoint equality, not proof of physical identity.'),init,elapsed


def prepare(out):
    out.mkdir(parents=True,exist_ok=False);(out/'images').mkdir();(out/'sources').mkdir()
    sources=['outputs/simple-action-20260928/cases.json','outputs/temporal-proposals-20260928/sequences.json',
             'outputs/temporal-proposals-20260928/sam-results.jsonl',str(PALETTE.relative_to(ROOT))]
    previous=json.loads((ROOT/sources[0]).read_text())
    for c in previous:c['origin']='reused_synthetic';c['group']='old_'+c['group']
    sequences=json.loads((ROOT/sources[1]).read_text());cases=previous+new_cases()+real_cases(sequences)
    replay={(r['sequence'],r['frame']):r for r in map(json.loads,(ROOT/sources[2]).read_text().splitlines())}
    palette=json.loads(PALETTE.read_text())[0]['palette'];jobs=[];evidence=[]
    for c in cases:
        contents={a:[] for a in ARMS}
        for trial in c['trials']:
            before,after=trial['before'],trial['after'];extra_before=[];extra_after=[]
            if c['origin']=='real_replay':
                # Recorded single-frame SAM detections, NOT temporally tracked boxes.
                convert=lambda boxes:[[round(b[0]*64/1000),round(b[1]*64/1000),round(b[2]*64/1000)-1,round(b[3]*64/1000)-1] for b in boxes]
                extra_before=convert(replay[c['sequence'],c['frame']-1]['boxes'])
                extra_after=convert(replay[c['sequence'],c['frame']]['boxes'])
            # Independently extracted current-frame candidates in EVERY arm.
            bi,bt=inventory(before,extra_before);ai,at=inventory(after,extra_after)
            start=time.perf_counter();delta=pixel_delta(before,after);dt=time.perf_counter()-start
            links,it,lt=measured_links(before,after,extra_before)
            row=dict(case=c['name'],trial=trial['id'],before=bi,after=ai,delta=delta,tracking=links,
                     timing=dict(inventory_seconds=bt+at,delta_seconds=dt,tracking_initial_seconds=it,tracking_update_seconds=lt))
            evidence.append(row)
            images={}
            for label,g in [('before',before),('after',after)]:
                im=render(g,palette);path=out/'images'/f'{c["name"]}-{trial["id"]}-{label}.png';im.save(path);images[label]=encode(im)
            for arm in ARMS:
                cs=contents[arm];cs.append(dict(type='text',text=f'Independent trial {trial["id"]}, action {trial["action"]}.'))
                if arm!='current' and not c['after_only']:
                    cs.extend([dict(type='text',text='BEFORE'),images['before'],dict(type='text',text=json.dumps(bi,separators=(',',':')))])
                cs.extend([dict(type='text',text='AFTER'),images['after'],dict(type='text',text=json.dumps(ai,separators=(',',':')))])
                if arm in ('delta','tracked') and not c['after_only']:
                    cs.append(dict(type='text',text='Raw pixel delta: '+json.dumps(delta,separators=(',',':'))))
                if arm=='tracked' and not c['after_only']:
                    cs.append(dict(type='text',text='Program rectangle correspondence: '+json.dumps(links,separators=(',',':'))))
        question=c['question']+'\n'+'\n'.join(o['letter']+'. '+o['text'] for o in c['options'])+'\nX. insufficient evidence'
        for arm in ARMS:
            contents[arm].append(dict(type='text',text=question))
            payload=dict(model='qwen3-vl-4b-instruct',temperature=0,seed=929830,max_tokens=1,cache_prompt=False,stream=False,
                         grammar='root ::= '+' | '.join(json.dumps(x) for x in [o['letter'] for o in c['options']]+['X']),
                         messages=[dict(role='system',content=SYSTEM+' Color IDs: '+json.dumps(COLORS)),dict(role='user',content=contents[arm])])
            jobs.append(dict(id=len(jobs),case=c['name'],arm=arm,payload=payload,payload_digest=digest(payload)))
    for name,data in [('cases',cases),('jobs',jobs),('evidence',evidence)]:save(out/(name+'.json'),data)
    code=['scripts/benchmark_delta_recognition.py','scripts/benchmark_concept_binding_split.py',
          'agent/cognition/geometry.py','agent/cognition/proposal_tracking.py']
    for p in code:(out/'sources'/Path(p).name).write_bytes((ROOT/p).read_bytes())
    save(out/'plan.json',dict(cases_digest=digest(cases),jobs_digest=digest(jobs),evidence_digest=digest(evidence),
         sources={p:sha(ROOT/p) for p in code+sources},images={str(p.relative_to(out)):sha(p) for p in (out/'images').glob('*.png')},
         conditions=list(ARMS),requests=len(jobs),
         design='Same question, after image, full per-frame candidate inventory in all arms. Frames adds before image/inventory. Delta adds lossless changed-pixel runs only. Tracked adds current ProposalTracker exact-pixel correspondence. No VLM motion detection call. No whole/part gate. No response-driven candidate pruning.',
         limits='Diagnostic small synthetic cases plus 6 reused real transitions. Current-only lacks temporal evidence; accuracy is information sufficiency, not fair temporal recognition. New generators and templates authored by implementing assistant, not held-out. One greedy sample. Real candidate boxes include recorded independent single-frame SAM outputs in every arm; detector time is excluded, so total live runtime cost is NOT measured. Synthetic candidates are independently program-extracted. No causal identification, autonomous question creation, gameplay, or runtime policy change.'))
    print('Prepared',len(cases),'cases',len(jobs),'requests',flush=True)


def verify(out):
    load=lambda n:json.loads((out/n).read_text());plan=load('plan.json');cases=load('cases.json');jobs=load('jobs.json')
    assert digest(cases)==plan['cases_digest'] and digest(jobs)==plan['jobs_digest']
    assert digest(load('evidence.json'))==plan['evidence_digest']
    for p,h in plan['sources'].items():assert sha(ROOT/p)==h,p
    for p,h in plan['images'].items():assert sha(out/p)==h,p
    for j in jobs:assert digest(j['payload'])==j['payload_digest']
    return cases,jobs


def run(out,base):
    _,jobs=verify(out);assert base in ['http://127.0.0.1:8080','http://localhost:8080']
    assert not (out/'responses.jsonl').exists();save(out/'server-before.json',http(base,'/props'))
    random.Random(929831).shuffle(jobs);start=time.perf_counter()
    with (out/'responses.jsonl').open('x') as f:
        for i,j in enumerate(jobs):
            r={k:v for k,v in j.items() if k!='payload'};t=time.perf_counter()
            try:
                response=http(base,'/v1/chat/completions',j['payload']);r.update(response=response,answer=response['choices'][0]['message']['content'])
            except Exception as exc:r.update(error=str(exc),answer='')
            r['seconds']=time.perf_counter()-t;f.write(json.dumps(r)+'\n');f.flush()
            if i%10==0 or r.get('error'):print(i+1,'/',len(jobs),j['arm'],j['case'],r['answer'],round(time.perf_counter()-start,1),r.get('error',''),flush=True)
    save(out/'server-after.json',http(base,'/props'))


def analyze(out):
    cases,jobs=verify(out);bycase={c['name']:c for c in cases};rows=list(map(json.loads,(out/'responses.jsonl').read_text().splitlines()))
    assert Counter(r['id'] for r in rows)==Counter(j['id'] for j in jobs)
    for r in rows:
        c=bycase[r['case']];r['valid']=not r.get('error') and r['answer'] in [o['letter'] for o in c['options']]+['X']
        r.update(correct=bool(r['valid'] and r['answer']==c['truth']),truth=c['truth'],group=c['group'],origin=c['origin'])
    arms={}
    for arm in ARMS:
        rr=[r for r in rows if r['arm']==arm];groups={}
        for key in sorted({r['group'] for r in rr}|{r['origin'] for r in rr}|{'all'}):
            selected=[r for r in rr if key=='all' or r['group']==key or r['origin']==key]
            groups[key]=dict(correct=sum(r['correct'] for r in selected),total=len(selected),abstentions=sum(r['answer']=='X' for r in selected))
        arms[arm]=dict(groups=groups,calls=len(rr),seconds=sum(r['seconds'] for r in rr),median_seconds=statistics.median(r['seconds'] for r in rr),
                       invalid=sum(not r['valid'] for r in rr),prompt_tokens=sum(r.get('response',{}).get('usage',{}).get('prompt_tokens',0) for r in rr))
    paired=[]
    for a,b in [('frames','delta'),('delta','tracked')]:
        aa={r['case']:r for r in rows if r['arm']==a};bb={r['case']:r for r in rows if r['arm']==b}
        paired.append(dict(before=a,after=b,gained=[c for c in aa if not aa[c]['correct'] and bb[c]['correct']],
                           lost=[c for c in aa if aa[c]['correct'] and not bb[c]['correct']]))
    summary=dict(arms=arms,paired=paired,server_unchanged=json.loads((out/'server-before.json').read_text())==json.loads((out/'server-after.json').read_text()))
    save(out/'scores.json',rows);save(out/'summary.json',summary)
    page=['<!doctype html><meta charset="utf-8"><title>Delta recognition ablation</title>',
          '<style>body{font:16px sans-serif;margin:24px}img{width:270px}td,th{border:1px solid #aaa;padding:8px}table{border-collapse:collapse}pre{white-space:pre-wrap}</style>',
          '<h1>静止画・画像対・差分・追跡情報の比較</h1><p>全条件で静的な候補一覧を保持。連動と物体同一性を区別する固定質問。</p>']
    for c in cases:
        page+=['<h2>'+html.escape(c['name'])+'</h2>','<p>'+html.escape(c['question'])+'</p>',
               '<p>'+html.escape(' / '.join(o['letter']+': '+o['text'] for o in c['options']))+' / X: insufficient</p>',
               '<p>正解 '+c['truth']+'</p><table><tr><th>条件</th><th>回答</th><th>正否</th></tr>']
        for arm in ARMS:
            r=next(r for r in rows if r['case']==c['name'] and r['arm']==arm)
            page.append(f'<tr><td>{arm}</td><td>{html.escape(r["answer"])}</td><td>{r["correct"]}</td></tr>')
        page.append('</table>')
        for t in c['trials']:
            page.append('<p>'+html.escape(t['id']+' action '+t['action'])+'</p>')
            for label in ['before','after']:page.append(f'<img src="images/{c["name"]}-{t["id"]}-{label}.png" alt="{label}">')
    page+=['<pre>'+html.escape(json.dumps(summary,indent=2))+'</pre>'];(out/'gallery.html').write_text('\n'.join(page))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','run','analyze']);p.add_argument('--out',required=True,type=Path)
    p.add_argument('--base',default='http://127.0.0.1:8080');a=p.parse_args()
    if a.command=='run':run(a.out,a.base)
    else:globals()[a.command](a.out)
