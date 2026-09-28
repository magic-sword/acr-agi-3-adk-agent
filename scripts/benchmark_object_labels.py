"""Two-stage, fixed-image naming ablation; no game actions or direction grouping."""
import argparse
from copy import deepcopy
import hashlib
import html
import json
from pathlib import Path
import random
import sys

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.benchmark_attention_selection import http, now
from scripts.benchmark_parallel import digest, read_lines
from scripts.benchmark_recognition import encode, save_json
from scripts.benchmark_object_recognition import request, run_jobs
from scripts.benchmark_action_cue import render_board

ARMS = ('direct', 'description', 'identifier', 'nickname', 'oracle_identifier')
STATIC = '''Inspect this single game screenshot. Inventory the visible objects or
regions before learning any rules. Describe color, shape, connected parts and
approximate place. Separate appearance from game roles: do not call anything a
player, goal, cursor or health bar. Include the large background/path region and
small distinct shapes or bars inside the game board; ignore the decorative console.
Parts and a larger grouping may both be listed if marked as such. Do not use numeric
coordinates. Give at most 8 entries. Return only JSON:
{"objects":[{"description":"visible appearance and approximate place",
"nickname":"short tentative appearance-based name"}]}'''
COMPARE = '''Compare BEFORE and AFTER. Identify the visible objects or parts that
changed and explain whether position, shape, color, size, or connectivity changed.
Describe their current appearance. Use supplied identifiers if helpful; update
nicknames when needed. An inventory may be wrong or incomplete: correct or extend
it from the images. A label is not proof of identity across frames. Permit splitting,
merging and uncertain correspondence. Different parts may move in opposite directions.
Do not assume game roles or rules. If nothing visibly changed say none; if you cannot
tell say uncertain. Do not infer a shared mechanism from simultaneous changes alone.
No numeric coordinates or exact distances. At most 130 words.'''
SYSTEM = 'Use only supplied images as visual evidence. Inventories are revisable hypotheses, not observations of a later frame. If images are missing, say visual evidence is unavailable.'


def cases():
    previous = {c['name']:c for c in json.loads(Path('outputs/action-cue-20260927/cases.json').read_text())}
    result = []
    for name in ('ls20_up', 'ls20_blocked'):
        c=deepcopy(previous[name]);c.pop('action',None)
        c['truth']={'category':'motion' if name=='ls20_up' else 'static',
                    'targets':['orange-blue connected block moves upward; yellow strip shortens'] if name=='ls20_up' else [],
                    'note':'No numerical position or direction accuracy required; bar-only detection is secondary.'}
        result.append(c)
    source=previous['synthetic_click_target'];before=deepcopy(source['before'])
    base=deepcopy(before)
    for y in range(32,38):
        for x in range(30,35):base[y][x]=3
    for dx,dy in ((0,0),(1,0),(-1,0),(0,1),(0,-1)):base[20+dy][12+dx]=3
    def block(g,x=30,y=32,rot=False):
        for j in range(6):
            for i in range(5):
                xx,yy=(5-j,i) if rot else (i,j)
                g[y+yy][x+xx]=12 if j<3 else 9
    def cross(g,x=12,y=20):
        for dx,dy in ((0,0),(1,0),(-1,0),(0,1),(0,-1)):g[y+dy][x+dx]=0
    for name,category,targets in [
        ('translate','motion',['connected orange-blue block moves upward']),
        ('cross','motion',['white cross moves right; orange-blue block is static']),
        ('reshape','deformation',['orange-blue rectangle becomes an L-shaped connected arrangement']),
        ('rotate','deformation',['vertical orange-blue stack becomes horizontal blue-left/orange-right arrangement']),
        ('opposite','multi_motion',['orange-blue block moves up and white cross moves down; linkage not established']),
        ('split','split',['orange upper part moves left and blue lower part moves right; formerly touching parts separate']),
        ('static','static',[]),
        ('ambiguous_exchange','ambiguous',['white cross and orange-blue arrangement exchange appearance locations; travel versus replacement is not established by two stills'])]:
        g=deepcopy(base)
        if name=='reshape':
            pixels=[(x,y) for y in range(8) for x in range(6) if x<3 or y>=6]
            assert len(pixels)==30
            for i,(x,y) in enumerate(pixels):g[31+y][30+x]=12 if i<15 else 9
            cross(g)
        elif name=='split':
            for j in range(6):
                for i in range(5):g[32+j][30+i+(-8 if j<3 else 8)]=12 if j<3 else 9
            cross(g)
        elif name=='ambiguous_exchange':block(g,10,18);cross(g,32,34)
        else:
            block(g,y=25 if name in ('translate','opposite') else 32,rot=name=='rotate')
            cross(g,x=19 if name=='cross' else 12,y=27 if name=='opposite' else 20)
        result.append({'name':'synthetic_'+name,'kind':'synthetic','before':deepcopy(before),'after':g,'palette':source['palette'],
            'truth':{'category':category,'targets':targets},'provenance':'Constructed static observations, not a real game rollout; no action or physical identity is supplied to model.'})
    return result


def oracle(c):
    if c['kind']=='synthetic':
        descriptions=['A small white plus-shaped cross toward the upper left.',
            'A small connected rectangular stack near the center, orange above blue.',
            'A blue square with a pale gray border toward the upper right.',
            'A long horizontal yellow strip near the bottom.',
            'A large gray rectangular field surrounded by a darker gray border.']
    else:
        block_place = ('near the top center, just below the blue angular pattern'
                       if c['name']=='ls20_blocked' else 'in the lower central path')
        descriptions=['A small white cross-like shape with pale gray pixels left of center.',
            f'A small connected rectangular block, orange above blue, {block_place}.',
            'A blue angular pattern in a dark square with a gray border near the top.',
            'A blue angular pattern in a dark square near the lower left.',
            'A long horizontal yellow strip along the bottom with separate red marks to its right.',
            'A gray branching path-like region surrounded by darker gray, with a black strip on the left.']
    return [{'description':d,'nickname':''} for d in descriptions]


def make_request(content,rep,prompt,tokens=512):
    p=request(content,True,rep,prompt);p['messages'][0]['content']=SYSTEM;p['max_tokens']=tokens;return p


def prepare(out):
    out.mkdir(parents=True,exist_ok=False)
    cs=cases();template=Image.open('outputs/action-cue-20260927/ui-template.png').convert('RGB')
    scenes={};jobs=[];gallery=['<!doctype html><meta charset="utf-8"><title>Object naming comparison</title><style>body{font:16px system-ui;margin:24px}img{width:40%;max-width:640px;image-rendering:pixelated}</style><h1>BEFORE / AFTER</h1>']
    for c in cs:
        scene=digest({'grid':c['before'],'palette':c['palette']})[:12];c['scene']=scene
        if scene not in scenes:scenes[scene]=c
        gallery.append('<h2>'+html.escape(c['name'])+'</h2>')
        for frame in ('before','after'):
            im=render_board(c,frame,template);im.save(out/f'{c["name"]}-{frame}.png')
            gallery.append(f'<img src="{c["name"]}-{frame}.png">')
        assert (c['before']==c['after'])==(c['truth']['category']=='static')
    for scene,c in scenes.items():
        for rep in range(3):
            p=make_request([encode(Image.open(out/f'{c["name"]}-before.png'))],rep,STATIC,700)
            jobs.append({'id':len(jobs),'case':scene,'representative':c['name'],'arm':'inventory','repeat':rep,'payload':p,'payload_digest':digest(p)})
    save_json(out/'cases.json',cs);save_json(out/'inventory-jobs.json',jobs)
    (out/'gallery.html').write_text('\n'.join(gallery))
    dependencies=['scripts/benchmark_object_labels.py','scripts/benchmark_action_cue.py','scripts/benchmark_object_recognition.py','scripts/benchmark_attention_selection.py','scripts/benchmark_parallel.py','scripts/benchmark_recognition.py','scripts/benchmark_motion_inputs.py','scripts/agent_monitor.py']
    hashes={f:hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in dependencies}
    (out/'sources').mkdir()
    for f in dependencies:(out/'sources'/Path(f).name).write_bytes(Path(f).read_bytes())
    save_json(out/'plan.json',{'created_at':now(),'cases_digest':digest(cs),'inventory_jobs_digest':digest(jobs),'sources':hashes,
        'arms':ARMS,'static_requests':len(jobs),'main_requests':len(cs)*len(ARMS)*3,'repeats':3,
        'rubric':{'static':'Per-scene target coverage, colors/shapes/grouping, omissions, invented objects, unsupported game roles; not exact coordinates.',
        'primary':'Correct changed target(s) and change class; reject invented changes. Motion cases require actual mover, not only bar shortening. Static cases require no visible changes. Ambiguous exchange requires recognizing altered arrangement and retaining multiple possible correspondences rather than declaring identity proved.',
        'secondary':'Shape changes, part split, both opposite-moving targets, role assertions, identity overcommitment, inventory correction. Direction secondary, exact coordinates excluded.',
        'oracle':'Manually authored BEFORE-only appearance list, no AFTER facts, roles, action or target selection. Diagnostic ceiling aid, not autonomous performance.',
        'review':'Implementing assistant semantic review with reasons; condition names hidden and outputs shuffled. Not independent human scoring. Invalid/timeouts/truncation fail.'},
        'fairness':'Every change arm sees exactly the same two full retro UI images; no crops, motion grouping, boxes or visual labels. Generated inventory shared across three naming arms for same scene/seed. Different text lengths and two-stage cost are recorded.',
        'scope':'2 actual ls20 pairs, 8 synthetic controls. No new game actions or game rules supplied. Paired snapshots cannot prove physical identity or causation. This tests immediate label use, not long-horizon persistence.',
        'sampler':'temperature .7 top_p .8 top_k20 min_p0 presence_penalty1.5 repeat_penalty1 seeds3407..3409; llama.cpp Qwen3-VL4B Q4_K_M; official-adapted, not full official reproduction',
        'budget':'Static700tokens/change512tokens/20seconds per call; cache_prompt false; 1 slot',
        'inventory_failure':'Invalid JSON becomes unavailable for all three generated inventory conditions; no repair using truth.',
        'warmups':'One inventory and one direct comparison excluded; 3 no-image controls separately.'})
    print('Prepared',len(cs),'cases;',len(jobs),'static requests',flush=True)


def parse_inventory(row):
    if not row['valid'] or not row['within_budget']:return None
    s=row.get('answer','').strip()
    if s.startswith('```'):s=s.split('\n',1)[1].rsplit('```',1)[0].strip()
    try:
        obj=json.loads(s)['objects']
        assert isinstance(obj,list) and 0<len(obj)<=8
        assert all(isinstance(x,dict) and isinstance(x.get('description'),str) and isinstance(x.get('nickname'),str) for x in obj)
        return obj
    except (ValueError,KeyError,TypeError,AssertionError):return None


def format_inventory(objects,arm):
    if objects is None:return 'Prior inventory unavailable.'
    lines=[]
    for i,obj in enumerate(objects):
        prefix='' if arm=='description' else f'Object {chr(65+i)}: '
        if arm=='nickname':prefix+='[tentative nickname: '+obj['nickname']+'] '
        lines.append(prefix+obj['description'])
    return '\n'.join(lines)


def run(out,base):
    plan=json.loads((out/'plan.json').read_text());cs=json.loads((out/'cases.json').read_text())
    invjobs=json.loads((out/'inventory-jobs.json').read_text())
    assert digest(cs)==plan['cases_digest'] and digest(invjobs)==plan['inventory_jobs_digest']
    for path,h in plan['sources'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==h
    save_json(out/'server-before.json',http(base.removesuffix('/v1')+'/props'))
    run_jobs(out,base,invjobs,'inventory-measurements.jsonl',[invjobs[0]])
    invrows=[r for r in read_lines(out/'inventory-measurements.jsonl') if not r['warmup']]
    inventories={(r['case'],r['repeat']):parse_inventory(r) for r in invrows}
    save_json(out/'inventories.json',[{'scene':r['case'],'repeat':r['repeat'],'objects':parse_inventory(r),'source_id':r['id']} for r in invrows])
    jobs=[]
    for c in cs:
        content=[]
        for frame in ('before','after'):content += [{'type':'text','text':frame.upper()},encode(Image.open(out/f'{c["name"]}-{frame}.png'))]
        for rep in range(3):
            for arm in ARMS:
                prompt=COMPARE
                if arm!='direct':
                    objects=oracle(c) if arm=='oracle_identifier' else inventories[c['scene'],rep]
                    prompt+='\nBEFORE-only revisable inventory:\n'+format_inventory(objects,arm)
                p=make_request(content,rep,prompt)
                jobs.append({'id':len(jobs),'case':c['name'],'scene':c['scene'],'kind':c['kind'],'arm':arm,'repeat':rep,'payload':p,'payload_digest':digest(p)})
    for rep in range(3):
        p=make_request([],rep,COMPARE+'\nBEFORE-only revisable inventory:\n'+format_inventory(oracle(cs[0]),'identifier'))
        jobs.append({'id':len(jobs),'case':'no_images','kind':'control','arm':'no_images','repeat':rep,'payload':p,'payload_digest':digest(p)})
    save_json(out/'jobs.json',jobs);save_json(out/'jobs-digest.json',{'digest':digest(jobs),'created_at':now()})
    run_jobs(out,base,jobs,'measurements.jsonl',[jobs[0]])
    save_json(out/'server-after.json',http(base.removesuffix('/v1')+'/props'))
    rows=[r for r in read_lines(out/'measurements.jsonl') if not r['warmup']];random.Random(45391).shuffle(rows)
    save_json(out/'review-key.json',{str(i):r['id'] for i,r in enumerate(rows)})
    save_json(out/'review-candidates.json',[{'review_id':i,'case':r['case'],'answer':r.get('answer',''),'valid':r['valid'] and r['within_budget']} for i,r in enumerate(rows)])


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('command',choices=['prepare','run']);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--base',default='http://127.0.0.1:8080/v1');a=ap.parse_args()
    if a.command=='prepare':prepare(a.output)
    else:run(a.output,a.base)
