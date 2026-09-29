"""Frozen action-evidence ablation: no history, text history, highlighted input view."""
import argparse
from copy import deepcopy
import hashlib
import html
import json
from pathlib import Path
import random
import sys

from PIL import Image, ImageDraw

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.agent_monitor import PALETTE
from scripts.fonts import ui_font
from scripts.benchmark_attention_selection import http, now
from scripts.benchmark_parallel import digest, read_lines
from scripts.benchmark_recognition import encode, save_json
from scripts.benchmark_motion_inputs import raw_image, NEAREST
from scripts.benchmark_object_recognition import request, run_jobs

ARMS=('no_history','text_history','visual_history')
PROMPT='''Compare BEFORE and AFTER. Identify game objects that visibly changed.
Describe their colors, shapes, connected parts and approximate place, then say
what changed: position, length, color, or another visible property. Distinguish
movement from other changes. If no game object changed, say none; if you cannot
tell, say uncertain. No numerical coordinates or exact distances. At most 100 words.'''
SYSTEM='''Use only visible evidence to judge what changed in the game board.
BEFORE and AFTER are the original observations. The third image, INPUT OVERLAY,
repeats BEFORE and may show a recorded input: a cyan glow marks the button used,
and a hollow cyan ring marks the click location. These are annotations, not game
objects or a third observation. Recorded input does not prove that anything moved,
that the clicked object changed, or that movement followed the input direction.
If no images are supplied, visual evidence is unavailable. Do not invent observations.'''


def logged_case(label,run,step,focus,change):
    root=Path('outputs/context-comparisons/20260925T064149191789Z')/run
    f=next(root.glob('*/cognition/*.observations.jsonl'));observations=read_lines(f)
    execution=f.with_name(f.name.replace('.observations.jsonl','.execution.jsonl'))
    act=next(r for r in read_lines(execution) if r['event']=='action_dispatched' and r['step']==step)
    before=next(r for r in observations if r['step']==step);after=next(r for r in observations if r['step']==step+1)
    assert act['observation_id']==before['observation_id']
    ack=next(r for r in read_lines(execution) if r['event']=='action_acknowledged' and r['step']==step)
    assert ack['next_step']==after['step'] and ack['action']==act['action']
    a=act['action']; action={'type':'UP'} if a['action']=='ACTION1' else {'type':'CLICK','x':a['x'],'y':a['y']}
    return {'name':label,'kind':'observed','before':before['grid'],'after':after['grid'],'palette':{str(i):c for i,c in enumerate(PALETTE)},
        'action':action,'truth':{'focus':focus,'change':change},'provenance':{'observations':str(f),'execution':str(execution),
        'before_observation_id':before['observation_id'],'after_observation_id':after['observation_id'],'dispatch':act,'acknowledgement':ack}}


def make_cases():
    cases=[logged_case('ls20_up','r1-ls20-shared_brief',0,'orange-over-blue connected block','moved_up'),
           logged_case('ls20_blocked','r1-ls20-shared_brief',6,None,'unchanged'),
           logged_case('vc33_click_remote','r1-vc33-baseline',0,'pink strip at the top right end','shortened'),
           logged_case('ft09_click_static','r1-ft09-shared_brief',0,None,'unchanged')]
    old={c['name']:c for c in json.loads(Path('outputs/retro-ui-20260927/cases.json').read_text())}
    for name,source,reverse,action,focus,change in [
        ('synthetic_up_opposite','synthetic_block',True,{'type':'UP'},'orange-over-blue connected block','moved_down'),
        ('synthetic_up_cross','synthetic_cross',False,{'type':'UP'},'white cross','moved_right'),
        ('synthetic_click_target','synthetic_block',False,{'type':'CLICK','x':32,'y':34},'orange-over-blue connected block','moved_up'),
        ('synthetic_click_other','synthetic_cross',False,{'type':'CLICK','x':32,'y':34},'white cross','moved_right'),
        ('synthetic_click_static','synthetic_static',False,{'type':'CLICK','x':32,'y':34},None,'unchanged')]:
        c=deepcopy(old[source]);c.update(name=name,kind='synthetic',action=action,truth={'focus':focus,'change':change})
        if reverse:c['before'],c['after']=c['after'],c['before']
        c['provenance']={'synthetic':'Constructed action-observation pair, not an actual game rollout. Includes opposite response and non-clicked mover to test action-following guesses.'}
        cases.append(c)
    for c in cases:
        changed=[(x,y) for y,row in enumerate(c['before']) for x,v in enumerate(row) if v!=c['after'][y][x]]
        c['changed_pixels']=changed
        assert bool(changed)==(c['truth']['change']!='unchanged')
    return cases


def render_board(case,frame,template):
    im=template.copy();im.paste(raw_image(case[frame],case['palette']).resize((512,512),NEAREST),(59,80));return im


def overlay(before,action,controls,marked):
    im=before.copy();d=ImageDraw.Draw(im)
    font=ui_font(16)
    d.text((50,761),'INPUT OVERLAY — BEFORE',font=font,fill='#ffffff')
    d.text((50,785),'Glow / ring = recorded input',font=font,fill='#b8eaf2')
    if not marked:return im
    r=controls['up' if action['type']=='UP' else 'click'];cx=r['x']+r['width']/2;cy=r['y']+r['height']/2
    # Procedural UI highlight: soft outer bands plus bright rim, no replacement of the arrow.
    for pad,color in [(10,'#184b58'),(7,'#22788a'),(4,'#42c5de'),(1,'#b1f5ff')]:
        d.rounded_rectangle((int(r['x']-pad),int(r['y']-pad),int(r['x']+r['width']+pad),int(r['y']+r['height']+pad)),radius=15,outline=color,width=3)
    if action['type']=='CLICK':
        x=59+action['x']*8+4;y=80+action['y']*8+4
        d.ellipse((x-17,y-17,x+17,y+17),outline='#092a36',width=5)
        d.ellipse((x-16,y-16,x+16,y+16),outline='#7df5ff',width=3)
        for dx,dy in [(-1,0),(1,0),(0,-1),(0,1)]:d.line((x+dx*21,y+dy*21,x+dx*28,y+dy*28),fill='#7df5ff',width=2)
    return im


def action_text(action):
    if action['type']=='UP':return 'Recorded input between BEFORE and AFTER: UP.'
    return f'Recorded input between BEFORE and AFTER: CLICK at column {action["x"]}, row {action["y"]} of the original 64 by 64 board (zero-based).'


def prepare(output):
    if (output/'jobs.json').exists():raise ValueError('refusing to overwrite frozen requests')
    cases=make_cases();controls=json.loads((output/'controls.json').read_text());template=Image.open('outputs/retro-ui-20260927/blank-shell.png').convert('RGB');jobs=[]
    gallery=['<!doctype html><meta charset="utf-8"><title>Action cue comparison</title><style>body{font:16px system-ui;margin:24px}img{width:24%;image-rendering:pixelated}</style><h1>BEFORE / AFTER / neutral overlay / input overlay</h1>']
    for c in cases:
        before=render_board(c,'before',template);after=render_board(c,'after',template)
        neutral=overlay(before,c['action'],controls,False);marked=overlay(before,c['action'],controls,True)
        assert before.crop((59,80,571,592)).tobytes()==raw_image(c['before'],c['palette']).resize((512,512),NEAREST).tobytes()
        assert after.crop((59,80,571,592)).tobytes()==raw_image(c['after'],c['palette']).resize((512,512),NEAREST).tobytes()
        if c['truth']['change']=='unchanged':assert before.tobytes()==after.tobytes()
        if c['action']['type']=='UP':assert neutral.crop((59,80,571,592)).tobytes()==marked.crop((59,80,571,592)).tobytes()
        if c['action']['type']=='CLICK':
            x=59+c['action']['x']*8+4;y=80+c['action']['y']*8+4
            assert marked.getpixel((x,y))==before.getpixel((x,y))
        gallery.append('<h2>'+html.escape(c['name'])+'</h2>')
        for label,im in [('before',before),('after',after),('neutral',neutral),('input',marked)]:
            file=f'{c["name"]}-{label}.png';im.save(output/file);gallery.append(f'<img src="{file}">')
        for arm in ARMS:
            content=[{'type':'text','text':'BEFORE'},encode(before),{'type':'text','text':'AFTER'},encode(after),
                     {'type':'text','text':'INPUT OVERLAY ON BEFORE'},encode(marked if arm=='visual_history' else neutral)]
            for rep in range(3):
                question=PROMPT+('\n'+action_text(c['action']) if arm=='text_history' else '')
                p=request(content,True,rep,question);p['messages'][0]['content']=SYSTEM
                jobs.append({'id':len(jobs),'case':c['name'],'kind':c['kind'],'arm':arm,'repeat':rep,'payload':p,'payload_digest':digest(p)})
    for typ in ('UP','CLICK'):
        for rep in range(3):
            action={'type':typ,'x':32,'y':34};p=request([],True,rep,PROMPT+'\n'+action_text(action));p['messages'][0]['content']=SYSTEM
            jobs.append({'id':len(jobs),'case':'no_image_'+typ.lower(),'kind':'control','arm':'action_only','repeat':rep,'payload':p,'payload_digest':digest(p)})
    assert len(jobs)==87
    # Within each case/seed, original BEFORE/AFTER images and sampling settings are identical.
    for c in cases:
        for rep in range(3):
            ps=[j['payload'] for j in jobs if j['case']==c['name'] and j['repeat']==rep]
            assert len(ps)==3
            assert all(p['messages'][1]['content'][:4]==ps[0]['messages'][1]['content'][:4] for p in ps)
    save_json(output/'cases.json',cases);save_json(output/'jobs.json',jobs)
    paths=[Path(__file__),Path('scripts/inspect_retro_controls.cjs'),Path('scripts/benchmark_object_recognition.py')]
    sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    for p in paths:(output/p.name).write_bytes(p.read_bytes())
    save_json(output/'plan.json',{'created_at':now(),'jobs_digest':digest(jobs),'cases_digest':digest(cases),'sources':sources,'main_requests':87,
        'arms':ARMS,'sampler':'Official-adapted parameters from previous benchmark: temperature .7, top_p .8, top_k 20, presence_penalty 1.5, repeat_penalty 1, min_p 0, seeds 3407..3409',
        'output_budget':256,'wall_budget_seconds':20,'warmup':'one ls20_up per arm, excluded',
        'cue_reading':'Two separate one-image diagnostic requests after main: name the glowing UP or CLICK button; exclude from change-recognition scores.',
        'scope':'Four logged real pairs (including blocked UP, remote click effect and no-effect click); five constructed synthetic action-response cases. Real and synthetic separated.',
        'fairness':'Each main request has three 640x820 images, original board 512x512. Original observations immutable in all arms; only auxiliary overlay or text metadata differ. Text has extra input tokens and numerical click location; no coordinate output or precision scoring.',
        'rubric':{'primary':'Identify the truth.focus object and correct change class (motion or strip shortening); no-change cases explicitly no game changes. No required numerical coordinates or exact distance. Direction is a secondary diagnostic. Reject invented changes to other objects or treating annotation as game change.',
        'secondary':'Track orange-blue mover and yellow strip separately in ls20_up. Correct label for recorded action alone is not change recognition. Record wrong-direction following of UP in synthetic down/right cases; wrongly identifying clicked block when cross moves; false positives on unchanged cases.',
        'review':'Implementing assistant semantic review of shuffled arm-hidden groups, not independent human scoring; invalid/truncated/timeouts fail.',
        'no_image':'Must acknowledge lack of visual evidence rather than infer motion from the supplied action.'}})
    (output/'gallery.html').write_text('\n'.join(gallery));print('Prepared 87 requests; logged action-observation alignment and immutable originals checked')


def run(output,base):
    jobs=json.loads((output/'jobs.json').read_text());plan=json.loads((output/'plan.json').read_text())
    assert digest(jobs)==plan['jobs_digest']
    for p,h in plan['sources'].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h
    save_json(output/'server-before.json',http(base.removesuffix('/v1')+'/props'))
    warm=[next(j for j in jobs if j['case']=='ls20_up' and j['arm']==a) for a in ARMS]
    run_jobs(output,base,jobs,'measurements.jsonl',warm)
    rows=[r for r in read_lines(output/'measurements.jsonl') if not r['warmup']];random.Random(41498).shuffle(rows)
    save_json(output/'review-key.json',{str(i):r['id'] for i,r in enumerate(rows)});groups={}
    for i,r in enumerate(rows):groups.setdefault((r['case'],r.get('answer',''),r['valid'] and r['within_budget']),[]).append(i)
    save_json(output/'review-groups.json',[{'group':i,'case':k[0],'answer':k[1],'valid':k[2],'review_ids':v} for i,(k,v) in enumerate(groups.items())])
    probes=[]
    for name in ('ls20_up','vc33_click_remote'):
        p=request([encode(Image.open(output/f'{name}-input.png'))],True,0,'Which input button is marked by the cyan glow? Return only UP, CLICK, or UNCERTAIN.');p['messages'][0]['content']='Read the input annotation. Do not infer a game response.';p['max_tokens']=16
        probes.append({'id':len(probes),'case':name,'arm':'cue_reading','repeat':0,'payload':p,'payload_digest':digest(p)})
    save_json(output/'cue-reading-jobs.json',probes);run_jobs(output,base,probes,'cue-reading.jsonl')
    save_json(output/'server-after.json',http(base.removesuffix('/v1')+'/props'))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['prepare','run']);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--base',default='http://127.0.0.1:8080/v1');a=p.parse_args()
    if a.command=='prepare':prepare(a.output)
    else:run(a.output,a.base)
