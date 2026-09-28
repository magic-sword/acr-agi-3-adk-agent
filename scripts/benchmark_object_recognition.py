"""Frozen free-description comparison of moving-object recognition; no game actions."""
import argparse
from copy import deepcopy
import hashlib
import html
import json
from pathlib import Path
import random
import sys
import time

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.benchmark_attention_selection import http, now
from scripts.benchmark_parallel import digest, read_lines
from scripts.benchmark_recognition import encode, save_json
from scripts.benchmark_motion_inputs import raw_image, NEAREST
from scripts.benchmark_visual_emphasis import mask_boxes
from scripts.preview_change_emphasis import masks

ARMS = ('plain_greedy', 'plain_official', 'diff_greedy', 'diff_official')
PROMPT = '''Compare BEFORE and AFTER. Identify any object whose position changed.
Describe its visible colors, shape, parts belonging together, and approximate place
so another observer can identify the same object. Distinguish an object moving
from only its length or color changing. If no object moved, say none; if you cannot
tell, say uncertain. Describe visible evidence, without assigning game roles.
Do not give numerical coordinates or exact distances. Use at most 100 words.'''
SYSTEM = '''Use only the supplied visual evidence. BEFORE and AFTER are two observations.
Auxiliary images, if supplied, repeat these observations with machine-detected
change regions outlined. Outlines are annotations, not game objects; a marked
region is not necessarily one object or a moving object. No images means the
visual evidence is unavailable. Do not invent observations.'''
RUBRIC = {
    'identity': 'Moving case: unambiguously identifies the actual moving object by appearance/grouping and says it moved. Mere appearance description is insufficient. None/uncertain is a miss. Static case: explicitly no movement, no invented mover.',
    'grouping': 'Moving multicolor parts are described as one connected object, not independent unrelated movers; not applicable to no-movement case.',
    'false_mover': 'Any stationary object asserted to move, including interpreting a shrinking strip as translating. Contradictory claims count.',
    'relation': 'Correct qualitative direction (secondary only); no coordinate or distance score.',
    'strip': 'Real pair only: correctly identifies shortening/lengthening, or unchanged in repeated frame, without inventing translation.',
    'strict': 'Identity success AND no false mover; partial/ambiguous descriptions do not pass.',
    'review': 'Semantic review by the implementing assistant, with synonyms accepted and written rationale; not an independent human evaluation. Review text is shuffled and arm names hidden. Invalid/truncated/over-budget outputs fail primary metrics.',
    'handoff': 'Stage 2: correct target reference, role as a hypothesis supported by observed change and action, and a concrete next check. This is textual usefulness, not measured execution success.'}


def make_cases(source):
    cases = json.loads(source.read_text())
    for c in cases:
        c['kind'] = 'observed' if c['name'] == 'forward' else 'derived'
        c['truth'] = {'mover': 'orange-over-blue connected rectangular block' if c['name'] != 'same_frame' else None,
                      'direction': {'forward': 'up', 'reverse': 'down', 'same_frame': None}[c['name']],
                      'strip': {'forward': 'shorter', 'reverse': 'longer', 'same_frame': 'unchanged'}[c['name']],
                      'stationary': ['white cross', 'upper blue shape', 'lower-left blue shape', 'red bottom marks']}
    palette = cases[0]['palette']
    for name, mover, colors, shift in [('block', 'block', (12, 9), (0, -7)),
                                      ('cross', 'cross', (12, 9), (7, 0)),
                                      ('recolor', 'block', (8, 11), (-7, 0)),
                                      ('static', None, (8, 11), (0, 0))]:
        base = [[3 if 4 <= x < 60 and 4 <= y < 60 else 4 for x in range(64)] for y in range(64)]
        for y in range(56, 58):
            for x in range(8, 54): base[y][x] = 11
        for y in range(7, 15):
            for x in range(45, 53): base[y][x] = 1 if x in (45,52) or y in (7,14) else 9
        frames = []
        for t in range(2):
            g = deepcopy(base)
            bx, by = 30 + (shift[0]*t if mover == 'block' else 0), 32 + (shift[1]*t if mover == 'block' else 0)
            for y in range(6):
                for x in range(5): g[by+y][bx+x] = colors[y//3]
            cx, cy = 12 + (shift[0]*t if mover == 'cross' else 0), 20 + (shift[1]*t if mover == 'cross' else 0)
            for dx, dy in [(0,0),(1,0),(-1,0),(0,1),(0,-1)]: g[cy+dy][cx+dx] = 0
            frames.append(g)
        cases.append({'name': 'synthetic_'+name, 'kind': 'synthetic', 'before': frames[0], 'after': frames[1], 'palette': palette,
                      'truth': {'mover': {'block': 'orange-over-blue rectangular block', 'cross': 'white cross', 'recolor': 'red-over-yellow rectangular block', 'static': None}[name],
                                'direction': {'block': 'up', 'cross': 'right', 'recolor': 'left', 'static': None}[name],
                                'strip': 'unchanged', 'stationary': ['upper blue square', 'yellow bottom strip'] + (['two-color central block'] if mover == 'cross' else ['white cross'])}})
    return cases


def render(grid, palette, mask=None):
    raw = raw_image(grid, palette).resize((384,384), NEAREST)
    im = Image.new('RGB', (416,416), '#18202b'); im.paste(raw, (16,16))
    if mask:
        draw = ImageDraw.Draw(im)
        for l,t,r,b in mask_boxes(mask):
            draw.rectangle((16+l*6-4,16+t*6-4,16+(r+1)*6+3,16+(b+1)*6+3), outline='#ff00ff', width=2)
        # Guarantee every marked source pixel retains its exact original RGB.
        for x,y in mask:
            im.paste(raw.crop((x*6,y*6,x*6+6,y*6+6)), (16+x*6,16+y*6))
    return im


def media(case, auxiliary=False):
    result = []
    for k in ('before','after'):
        result += [{'type':'text','text':k.upper()}, encode(render(case[k],case['palette']))]
    if auxiliary:
        _, mask = masks(case['before'],case['after'])
        for k in ('before','after'):
            result += [{'type':'text','text':'AUXILIARY '+k.upper()}, encode(render(case[k],case['palette'],mask))]
    return result


def request(content, official, rep, prompt=PROMPT):
    params = {'temperature': .7, 'top_p': .8, 'top_k': 20, 'min_p': 0, 'repeat_penalty': 1., 'presence_penalty': 1.5, 'seed':3407+rep} if official else {
        'temperature':0, 'top_p':.95, 'top_k':40, 'min_p':.05, 'repeat_penalty':1., 'presence_penalty':0., 'seed':123+rep}
    return dict(params, model='qwen3-vl-4b-instruct', messages=[{'role':'system','content':SYSTEM},
        {'role':'user','content':content+[{'type':'text','text':prompt}]}], max_tokens=256, cache_prompt=False, id_slot=0, stream=False)


def prepare(source, output):
    output.mkdir(parents=True, exist_ok=False); cases = make_cases(source); jobs = []
    gallery = ['<!doctype html><meta charset="utf-8"><title>Object recognition inputs</title><style>body{font:16px system-ui;margin:24px}img{width:24%;image-rendering:pixelated}</style>']
    for c in cases:
        gallery.append('<h2>'+html.escape(c['name'])+'</h2><p>Original BEFORE / AFTER, auxiliary BEFORE / AFTER</p>')
        _, mask = masks(c['before'],c['after'])
        for aux in (False,True):
            for k in ('before','after'):
                f=f'{c["name"]}-{k}-{int(aux)}.png';render(c[k],c['palette'],mask if aux else None).save(output/f)
                gallery.append(f'<img src="{f}">')
        for arm in ARMS:
            content = media(c,arm.startswith('diff'))
            for rep in range(3):
                payload = request(content,arm.endswith('official'),rep)
                jobs.append({'id':len(jobs),'case':c['name'],'kind':c['kind'],'arm':arm,'repeat':rep,'payload':payload,'payload_digest':digest(payload)})
    for official in (False, True):
        for rep in range(3):
            p=request([],official,rep)
            jobs.append({'id':len(jobs),'case':'no_image','kind':'control','arm':'no_image_'+('official' if official else 'greedy'),'repeat':rep,'payload':p,'payload_digest':digest(p)})
    save_json(output/'cases.json',cases);save_json(output/'jobs.json',jobs)
    save_json(output/'plan.json',{'created_at':now(),'jobs_digest':digest(jobs),'cases_digest':digest(cases),'main_requests':len(jobs),'arms':ARMS,
        'wall_budget_seconds':20,'max_tokens':256,'rubric':RUBRIC,'primary':'strict moving-object identity; no coordinate scoring',
        'scope':'One real UP transition, its reversed and repeated controls, four synthetic counterfactuals. Three repeats are not independent game samples.',
        'fairness':'Same output/time budget; plain=2 images, diff=4 images. No equal-input-token/compute claim. Diff is assisted localization.',
        'official':'Adapted official generation parameters, llama.cpp repeat_penalty alias; min_p disabled. Not reproduction of vLLM benchmark.',
        'stage2':'12 requests: each of the 4x3 forward descriptions frozen, original images and actual UP action; role hypothesis and next check. Same settings/budget.',
        'warmup':'one forward request per image arm, excluded', 'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    (output/'gallery.html').write_text('\n'.join(gallery));(output/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    print('Prepared',len(jobs),'primary requests',flush=True)


def run_jobs(output, base, jobs, filename, warm=()):
    path=output/filename
    if path.exists(): raise ValueError('refusing to overwrite measurements')
    shuffled=list(jobs);random.Random(27103).shuffle(shuffled)
    with path.open('w') as log:
        for i,(j,iswarm) in enumerate([(j,True) for j in warm]+[(j,False) for j in shuffled]):
            row={k:v for k,v in j.items() if k!='payload'};row.update(warmup=iswarm,started_at=now());start=time.monotonic()
            try:
                r=http(base+'/chat/completions',j['payload'],20);row['response']=r
                choice=r['choices'][0];row['answer']=choice['message']['content'].strip()
                row['valid']=bool(row['answer']) and choice['finish_reason']=='stop'
            except Exception as e: row.update(valid=False,error=f'{type(e).__name__}: {e}')
            row['seconds']=time.monotonic()-start;row['within_budget']=row['seconds']<=20
            log.write(json.dumps(row,ensure_ascii=False)+'\n');log.flush()
            print(i,j['case'],j['arm'],row['valid'],round(row['seconds'],2),row.get('error',''),flush=True)


def run(output,base):
    jobs=json.loads((output/'jobs.json').read_text());plan=json.loads((output/'plan.json').read_text())
    assert digest(jobs)==plan['jobs_digest']
    assert hashlib.sha256(Path(__file__).read_bytes()).hexdigest()==plan['source_sha256']
    save_json(output/'server-before.json',http(base.removesuffix('/v1')+'/props'))
    warm=[next(j for j in jobs if j['case']=='forward' and j['arm']==a) for a in ARMS]
    run_jobs(output,base,jobs,'measurements.jsonl',warm)
    save_json(output/'server-after.json',http(base.removesuffix('/v1')+'/props'))
    rows=[r for r in read_lines(output/'measurements.jsonl') if not r['warmup']];random.Random(19271).shuffle(rows)
    save_json(output/'review-key.json',{str(i):r['id'] for i,r in enumerate(rows)})
    save_json(output/'review-candidates.json',[{'review_id':i,'case':r['case'],'answer':r.get('answer',''),'valid':r['valid'] and r['within_budget']} for i,r in enumerate(rows)])


def stage2(output,base):
    c=next(c for c in json.loads((output/'cases.json').read_text()) if c['name']=='forward');jobs=[]
    for row in read_lines(output/'measurements.jsonl'):
        if row['warmup'] or row['case']!='forward': continue
        prompt='''An actual UP action occurred between BEFORE and AFTER. A prior observer wrote:
<prior_description>'''+row.get('answer','unavailable')+'''</prior_description>
Use the original images to check this description; correct it if needed. Which visible
object is a candidate for the controlled object, and why? Give a tentative role hypothesis,
an appearance-based reference usable by the next observer, and one concrete next action
with the visual result to check. One action does not prove controllability. Do not infer
observed movement solely from the action name. No coordinates. At most 100 words.'''
        p=request(media(c),row['arm'].endswith('official'),row['repeat'],prompt)
        jobs.append({'id':row['id'],'case':'forward','arm':row['arm'],'repeat':row['repeat'],'stage1_valid':row['valid'] and row['within_budget'],'payload':p,'payload_digest':digest(p)})
    save_json(output/'stage2-jobs.json',jobs);run_jobs(output,base,jobs,'stage2-measurements.jsonl')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['prepare','run','stage2'])
    p.add_argument('--source',type=Path,default=Path('outputs/visual-emphasis-comparison-20260927/cases.json'))
    p.add_argument('--output',type=Path,required=True);p.add_argument('--base',default='http://127.0.0.1:8080/v1');a=p.parse_args()
    if a.command=='prepare': prepare(a.source,a.output)
    elif a.command=='run': run(a.output,a.base)
    else: stage2(a.output,a.base)
