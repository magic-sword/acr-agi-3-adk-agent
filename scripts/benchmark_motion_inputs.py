"""Frozen-input ablation of Qwen motion presentation. No game actions."""
import argparse
import base64
from collections import Counter
from copy import deepcopy
import hashlib
import html
import json
from pathlib import Path
import random
import re
import statistics
import subprocess
import sys
import time

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.benchmark_attention_selection import http, now
from scripts.benchmark_parallel import digest, read_lines
from scripts.benchmark_recognition import encode, save_json
from scripts.benchmark_visual_emphasis import mask_boxes
from scripts.preview_change_emphasis import masks, rgb

ARMS = ('pair', 'sheet', 'zoom', 'video', 'no_image')
NEAREST = getattr(Image, 'Resampling', Image).NEAREST
OPTIONS = {'motion': ['up', 'down', 'left', 'right', 'unchanged', 'uncertain'],
           'strip': ['shorter', 'longer', 'unchanged', 'uncertain'],
           'cross': ['moved', 'unchanged', 'uncertain']}
QUESTIONS = {
    'motion': 'From BEFORE to AFTER, which direction did the adjacent orange-and-blue group move?',
    'strip': 'From BEFORE to AFTER, did the long yellow strip become shorter, longer, or stay the same?',
    'cross': 'From BEFORE to AFTER, did the white cross move or stay in the same position?'}


def raw_image(grid, palette):
    im = Image.new('RGB', (64, 64))
    im.putdata([rgb(palette[str(c)]) for row in grid for c in row])
    return im


def crop_boxes(before, after):
    _, mask = masks(before, after)
    boxes = mask_boxes(mask)
    # Every changed component is retained, with the SAME crop at both times.
    return [[max(0,l-4), max(0,t-4), min(64,r+5), min(64,b+5)] for l,t,r,b in boxes] or [[0,0,64,64]]


def zoom_panel(im, boxes):
    panel = Image.new('RGB', (512, max(256, 256*len(boxes))), '#18202b')
    for i, box in enumerate(boxes):
        crop = im.crop(box); factor = min(496//crop.width, 240//crop.height)
        crop = crop.resize((crop.width*factor, crop.height*factor), NEAREST)
        panel.paste(crop, ((512-crop.width)//2, i*256+(256-crop.height)//2))
    return panel


def movie(images):
    command = ['docker','compose','exec','-T','vlm','ffmpeg','-hide_banner','-loglevel','error',
               '-f','rawvideo','-pixel_format','rgb24','-video_size','384x384','-framerate','4',
               '-i','pipe:0','-frames:v','2','-c:v','ffv1','-level','3','-f','matroska','pipe:1']
    return subprocess.run(command, input=b''.join(im.tobytes() for im in images),
                          capture_output=True, check=True, timeout=30).stdout


def case_media(case, output):
    raw = [raw_image(case[k],case['palette']) for k in ('before','after')]
    full = [im.resize((384,384),NEAREST) for im in raw]
    boxes = crop_boxes(case['before'],case['after'])
    zoom = [zoom_panel(im,boxes) for im in raw]
    sheet = Image.new('RGB',(784,408),'#18202b'); draw=ImageDraw.Draw(sheet)
    for i,im in enumerate(full):
        sheet.paste(im,(i*400,24));draw.text((i*400+8,6),('BEFORE','AFTER')[i],fill='white')
    for i,im in enumerate(full+zoom): im.save(output/f'{case["name"]}-{i}.png')
    sheet.save(output/f'{case["name"]}-sheet.png')
    clip=movie(full);(output/f'{case["name"]}.mkv').write_bytes(clip)
    pair=[{'type':'text','text':'BEFORE'},encode(full[0]),{'type':'text','text':'AFTER'},encode(full[1])]
    return {'pair':pair, 'sheet':[encode(sheet)],
            'zoom':pair+[{'type':'text','text':'Additional BEFORE crops'},encode(zoom[0]),
                         {'type':'text','text':'Additional AFTER crops (same regions and scale)'},encode(zoom[1])],
            'video':[{'type':'input_video','input_video':{'data':'data:video/x-matroska;base64,'+base64.b64encode(clip).decode()}}],
            'no_image':[]}, boxes


def make_cases(source):
    cases=json.loads(source.read_text())
    for c in cases:
        c['kind']='observed' if c['name']=='forward' else 'derived'
        c['truth']={'motion':'up' if c['name']=='forward' else 'down' if c['name']=='reverse' else 'unchanged',
                    'strip':'shorter' if c['name']=='forward' else 'longer' if c['name']=='reverse' else 'unchanged',
                    'cross':'unchanged'}
    palette=cases[0]['palette']
    for layout in range(2):
        for direction,(dx,dy) in {'up':(0,-7),'down':(0,7),'left':(-7,0),'right':(7,0),'unchanged':(0,0)}.items():
            b=[[3]*64 for _ in range(64)]
            for y in range(64):
                for x in range(64):
                    if x<4 or y<4 or x>59 or y>59:b[y][x]=4
            for y in range(56,58):
                for x in range(8,54):b[y][x]=11
            for x,y in [(12,12),(11,12),(13,12),(12,11),(12,13)]:b[y][x]=0
            for y in range(7,12):
                for x in range(45,50):b[y][x]=9
            a=deepcopy(b); x0,y0=(22,30) if layout==0 else (38,22);w,h=(4,6) if layout==0 else (6,4)
            for grid,sx,sy in [(b,x0,y0),(a,x0+dx,y0+dy)]:
                for y in range(h):
                    for x in range(w):grid[sy+y][sx+x]=12 if y<h//2 else 9
            cases.append({'name':f'synthetic_{layout}_{direction}','kind':'synthetic','before':b,'after':a,
                          'palette':palette,'truth':{'motion':direction}})
    return cases


def request(media, task, options):
    question=QUESTIONS[task]+'\n'+'\n'.join(f'{chr(65+i)}. {x}' for i,x in enumerate(options))
    question+='\nReturn only the letter of the best answer. Choose uncertain if not visible.'
    return {'model':'qwen3-vl-4b-instruct','messages':[
        {'role':'system','content':'Compare the first observation (BEFORE) with the second (AFTER). '
         'A video presents these same two observations in order; its playback timing is synthetic. '
         'Additional crops, if present, repeat the same observations. Describe visible changes only.'},
        {'role':'user','content':media+[{'type':'text','text':question}]}],
        'temperature':0,'seed':123,'max_tokens':16,'cache_prompt':False,'id_slot':0,'stream':False}


def parse(response, options):
    choice=response['choices'][0]
    if choice['finish_reason']=='length':raise ValueError('truncated')
    text=choice['message']['content'].strip()
    if not re.fullmatch(r'[A-Z][.]?',text):raise ValueError('not a single option letter: '+text)
    i=ord(text[0])-65
    if not 0<=i<len(options):raise ValueError('unknown option')
    return options[i]


def prepare(source, output):
    output.mkdir(parents=True,exist_ok=False);cases=make_cases(source);jobs=[];media_by_case={};crop_metadata={}
    for c in cases:
        media,boxes=case_media(c,output);media_by_case[c['name']]=media;crop_metadata[c['name']]=boxes
        repeats=3 if c['kind']!='synthetic' else 1
        for task,expected in c['truth'].items():
            for rep in range(repeats):
                options=list(OPTIONS[task]);random.Random(24000+rep+len(jobs)).shuffle(options)
                for arm in ARMS:
                    p=request(media[arm],task,options)
                    jobs.append({'case':c['name'],'kind':c['kind'],'task':task,'repeat':rep,'arm':arm,
                                 'expected':expected,'options':options,'payload':p,'payload_digest':digest(p)})
    save_json(output/'cases.json',cases);save_json(output/'jobs.json',jobs);save_json(output/'crops.json',crop_metadata)
    save_json(output/'plan.json',{'created_at':now(),'jobs_digest':digest(jobs),'cases_digest':digest(cases),
        'main_requests':len(jobs),'arms':ARMS,'wall_budget_seconds':20,'max_tokens':16,'cache_prompt':False,
        'repeats':'3 rotated option orders per real/derived case; 1 per synthetic case',
        'scope':'One observed pair; reversed and repeated controls; 10 synthetic direction cases, separately scored.',
        'video':'2 lossless FFV1 frames at 4 fps, no interpolation; synthetic presentation times 0 and .25s; llama.cpp native input_video',
        'crop':'All connected expanded change masks plus 4-cell margin; fixed coordinates across time. Empty mask falls back to full board.',
        'fairness':'Same original frames and per-request time budget. Image/token counts differ and are reported; no equal-compute claim.',
        'rubric':'Exact option correctness; uncertain is incorrect when truth is known. Invalid/timeouts count as failures. Changed motion excludes unchanged controls.',
        'warmup':'One forward-motion request per arm, excluded from main',
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    (output/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    page=['<!doctype html><meta charset="utf-8"><title>Qwen motion input comparison</title><style>img{max-width:48%;image-rendering:pixelated}body{font:16px system-ui;margin:24px}</style>']
    for c in cases:
        n=c['name'];page.append(f'<h2>{n}</h2><img src="{n}-0.png"><img src="{n}-1.png"><p>Crops (same positions at both times)</p><img src="{n}-2.png"><img src="{n}-3.png">')
    (output/'gallery.html').write_text('\n'.join(page));print('Prepared',len(jobs),'main requests',flush=True)


def run(output, base):
    assert not (output/'measurements.jsonl').exists()
    jobs=json.loads((output/'jobs.json').read_text());plan=json.loads((output/'plan.json').read_text())
    assert digest(jobs)==plan['jobs_digest']
    assert hashlib.sha256(Path(__file__).read_bytes()).hexdigest()==plan['source_sha256']
    save_json(output/'server-before.json',http(base.removesuffix('/v1')+'/props'))
    warm=[next(j for j in jobs if j['case']=='forward' and j['task']=='motion' and j['arm']==a) for a in ARMS]
    shuffled=list(jobs);random.Random(26000).shuffle(shuffled)
    with (output/'measurements.jsonl').open('w') as log:
        for i,(j,iswarm) in enumerate([(j,True) for j in warm]+[(j,False) for j in shuffled]):
            row={k:v for k,v in j.items() if k!='payload'};row.update(warmup=iswarm,started_at=now());start=time.monotonic()
            try:
                r=http(base+'/chat/completions',j['payload'],20);row['response']=r;row['answer']=parse(r,j['options']);row['valid']=True
            except Exception as exc:row.update(valid=False,error=f'{type(exc).__name__}: {exc}')
            row['seconds']=time.monotonic()-start;row['within_budget']=row['seconds']<=20
            row['correct']=row['valid'] and row['within_budget'] and row.get('answer')==j['expected']
            log.write(json.dumps(row,ensure_ascii=False)+'\n');log.flush()
            print(i,j['case'],j['task'],j['arm'],row.get('answer'),row['correct'],round(row['seconds'],2),row.get('error',''),flush=True)
    save_json(output/'server-after.json',http(base.removesuffix('/v1')+'/props'));report(output)


def report(output):
    rows=[r for r in read_lines(output/'measurements.jsonl') if not r['warmup']];summary=[]
    for subset,predicate in [('forward',lambda r:r['case']=='forward'),('reverse',lambda r:r['case']=='reverse'),
                              ('same_frame',lambda r:r['case']=='same_frame'),('synthetic',lambda r:r['kind']=='synthetic')]:
        for task in OPTIONS:
            for arm in ARMS:
                rs=[r for r in rows if predicate(r) and r['task']==task and r['arm']==arm]
                if not rs:continue
                summary.append({'subset':subset,'task':task,'arm':arm,'n':len(rs),'correct':sum(r['correct'] for r in rs),
                    'valid':sum(r['valid'] for r in rs),'seconds_median':statistics.median(r['seconds'] for r in rs),
                    'prompt_tokens_median':statistics.median(r.get('response',{}).get('usage',{}).get('prompt_tokens',0) for r in rs),
                    'answers':dict(Counter(r.get('answer','invalid') for r in rs))})
    save_json(output/'summary.json',summary)
    lines=['# Motion input comparison','','|Subset|Task|Arm|Correct|Valid|Seconds|Prompt tokens|','|---|---|---|---:|---:|---:|---:|']
    for s in summary:lines.append(f'|{s["subset"]}|{s["task"]}|{s["arm"]}|{s["correct"]}/{s["n"]}|{s["valid"]}|{s["seconds_median"]:.2f}|{s["prompt_tokens_median"]}|')
    (output/'report.md').write_text('\n'.join(lines)+'\n');print('Summarized',len(rows),'measurements',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['prepare','run','report'])
    p.add_argument('--source',type=Path,default=Path('outputs/visual-emphasis-comparison-20260927/cases.json'))
    p.add_argument('--output',type=Path,required=True);p.add_argument('--base',default='http://127.0.0.1:8080/v1');a=p.parse_args()
    if a.command=='prepare':prepare(a.source,a.output)
    elif a.command=='run':run(a.output,a.base)
    else:report(a.output)
