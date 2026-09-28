"""Compare equal-size board renderings with and without the ARC Prize handheld UI."""
import argparse
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
from scripts.benchmark_motion_inputs import raw_image, NEAREST
from scripts.benchmark_object_recognition import request, run_jobs, RUBRIC

VIEWS=('plain','shell','shell_grid')
ARMS=tuple(v+'_'+s for v in VIEWS for s in ('greedy','official'))


def prepare(output):
    if (output/'jobs.json').exists(): raise ValueError('refusing to overwrite frozen jobs')
    cases=json.loads((output/'cases.json').read_text());geo=json.loads((output/'geometry.json').read_text())
    x,y=geo['x'],geo['y'];box=(x,y,x+512,y+512);jobs=[];image_hashes={}
    first_shell=Image.open(output/'forward-before-shell.png').convert('RGB')
    first_grid=Image.open(output/'forward-before-shell_grid.png').convert('RGB')
    def outside(im):
        copy=im.copy();copy.paste((0,0,0),box);return copy.tobytes()
    assert outside(first_shell)==outside(first_grid)
    gallery=['<!doctype html><meta charset="utf-8"><title>Retro UI comparison</title><style>body{font:16px system-ui;margin:24px;background:#eee}img{width:32%;image-rendering:pixelated}</style>',
             '<h1>Equal-size inputs: plain / official shell / shell + grid</h1><p>Frozen source boards; UI is not a new game observation.</p>']
    for c in cases:
        for k in ('before','after'):
            raw=raw_image(c[k],c['palette']).resize((512,512),NEAREST)
            im=Image.new('RGB',(640,820),'#0e0c0d');im.paste(raw,(x,y));im.save(output/f'{c["name"]}-{k}-plain.png')
            shell=Image.open(output/f'{c["name"]}-{k}-shell.png').convert('RGB')
            grid=Image.open(output/f'{c["name"]}-{k}-shell_grid.png').convert('RGB')
            assert shell.size==grid.size==im.size==(640,820)
            assert shell.crop(box).tobytes()==raw.tobytes()==im.crop(box).tobytes()
            assert outside(shell)==outside(grid)==outside(first_shell)
            # Every grid-cell center retains the original color; only separator lines differ.
            for sy in range(64):
                for sx in range(64):assert grid.getpixel((x+sx*8+4,y+sy*8+4))==raw.getpixel((sx*8+4,sy*8+4))
            if c['name'] in ('same_frame','synthetic_static') and k=='after':
                for v in VIEWS:
                    assert Image.open(output/f'{c["name"]}-before-{v}.png').tobytes()==Image.open(output/f'{c["name"]}-after-{v}.png').tobytes()
            gallery.append(f'<h2>{html.escape(c["name"])} {k}</h2>')
            for v in VIEWS:
                f=f'{c["name"]}-{k}-{v}.png';gallery.append(f'<img src="{f}">');image_hashes[f]=hashlib.sha256((output/f).read_bytes()).hexdigest()
        for v in VIEWS:
            content=[]
            for k in ('before','after'):content += [{'type':'text','text':k.upper()},encode(Image.open(output/f'{c["name"]}-{k}-{v}.png').convert('RGB'))]
            for official in (False,True):
                for rep in range(3):
                    p=request(content,official,rep);jobs.append({'id':len(jobs),'case':c['name'],'kind':c['kind'],'view':v,
                        'arm':v+'_'+('official' if official else 'greedy'),'repeat':rep,'payload':p,'payload_digest':digest(p)})
    for control in ('blank_shell','no_image'):
        content=[]
        if control=='blank_shell':
            for k in ('BEFORE','AFTER'):content += [{'type':'text','text':k},encode(Image.open(output/'blank-shell.png').convert('RGB'))]
        for official in (False,True):
            for rep in range(3):
                p=request(content,official,rep);jobs.append({'id':len(jobs),'case':control,'kind':'control','view':control,
                    'arm':control+'_'+('official' if official else 'greedy'),'repeat':rep,'payload':p,'payload_digest':digest(p)})
    assert len(jobs)==138
    save_json(output/'jobs.json',jobs)
    sources={}
    for file in [Path(__file__),Path('scripts/render_retro_ui.cjs'),Path('scripts/benchmark_object_recognition.py')]:
        sources[str(file)]=hashlib.sha256(file.read_bytes()).hexdigest();(output/file.name).write_bytes(file.read_bytes())
    save_json(output/'plan.json',{'created_at':now(),'jobs_digest':digest(jobs),'cases_digest':digest(cases),'sources':sources,'image_hashes':image_hashes,
        'source_url':'https://arcprize.org/tasks/ls20','main_requests':138,'arms':ARMS,'wall_budget_seconds':20,'max_tokens':256,'rubric':RUBRIC,
        'scope':'Same seven prior cases, three repeats, two samplers, three views = 126; blank-shell and no-image controls add 12. Fourteen original frames rendered by local Chrome without network or game actions.',
        'isolation':'Plain and shell board RGB are byte-identical at the same position and 512x512 scale in 640x820 images. All shell pixels outside the display identical across both times, cases and shell/grid conditions. Grid case adds only black alpha .15 cell lines; centers retain RGB.',
        'ui':'Public static HTML/CSS and font/texture, with display size/layout overrides. Title GAME replaces task ID; decorative LEVEL 1 / 7 held constant, not observed synthetic-case progress. No action press/highlight. This is a reconstruction, not actual gameplay screenshots.',
        'secondary':'Does the model describe the large background as one robot/person? Are independent small shapes identified? Does it confuse the surrounding UI with the game board? Semantic review, not keyword scoring.',
        'controls':'Blank-shell has two identical blank displays, should report no mover and not invent gameplay objects. No-image should acknowledge missing evidence.',
        'warmup':'one forward request per six main arms, excluded','comparison_limit':'Old tests used smaller boards; use newly measured equal-sized plain controls for UI-effect claims. No claim about human perception or other games.'})
    save_json(output/'asset-hashes.json',{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in (output/'assets').iterdir() if f.is_file()})
    (output/'gallery.html').write_text('\n'.join(gallery))
    print('Prepared 138 requests; RGB, scale, static controls, grid centers and invariant UI checked')


def run(output,base):
    jobs=json.loads((output/'jobs.json').read_text());plan=json.loads((output/'plan.json').read_text())
    assert digest(jobs)==plan['jobs_digest']
    assert digest(json.loads((output/'cases.json').read_text()))==plan['cases_digest']
    for name,h in plan['sources'].items():assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==h
    save_json(output/'server-before.json',http(base.removesuffix('/v1')+'/props'))
    warm=[next(j for j in jobs if j['case']=='forward' and j['arm']==a) for a in ARMS]
    run_jobs(output,base,jobs,'measurements.jsonl',warm)
    save_json(output/'server-after.json',http(base.removesuffix('/v1')+'/props'))
    rows=[r for r in read_lines(output/'measurements.jsonl') if not r['warmup']];random.Random(98237).shuffle(rows)
    save_json(output/'review-key.json',{str(i):r['id'] for i,r in enumerate(rows)})
    groups={}
    for i,r in enumerate(rows):groups.setdefault((r['case'],r.get('answer',''),r['valid'] and r['within_budget']),[]).append(i)
    save_json(output/'review-groups.json',[{'group':i,'case':k[0],'answer':k[1],'valid':k[2],'review_ids':v} for i,(k,v) in enumerate(groups.items())])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['prepare','run']);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--base',default='http://127.0.0.1:8080/v1');a=p.parse_args()
    if a.command=='prepare':prepare(a.output)
    else:run(a.output,a.base)
