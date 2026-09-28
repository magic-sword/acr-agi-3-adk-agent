"""Compare three temporal overlays to repeated AFTER, with equal image count."""
import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random
import shutil
import sys
import time
from PIL import Image

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.benchmark_attention_selection import http,now
from scripts.benchmark_parallel import digest,read_lines
from scripts.benchmark_recognition import encode,save_json
from scripts.benchmark_object_gate import payload,inventory_text

ARMS=['repeat','blend','outline','split']
LEGEND={
 'repeat':'This third image repeats the original AFTER image. It is not another time step.',
 'blend':'This third image is a derived overlay, not an observation: 50% BEFORE + 50% AFTER at the same pixel positions. Unchanged context is dimmed to 70%, except within one source pixel of a difference. Blended colors are artificial. A double image is not two objects. This overlay alone does not encode time direction.',
 'outline':'This third image is a derived overlay, not an observation: 50% BEFORE + 50% AFTER. Unchanged context is dimmed to 70%, except within one source pixel of a difference. Magenta dashed lines show color boundaries present only in BEFORE; cyan solid lines show boundaries present only in AFTER. Lines are annotations, not game objects. Shared boundaries are not highlighted. No object identity or motion is inferred by this rendering.',
 'split':'This third image is a derived overlay, not an observation. Within each changed source-pixel cell, the LEFT half shows its original BEFORE color and the RIGHT half its original AFTER color. Unchanged context is dimmed to 70%, except within one source pixel of a difference. Stripes are a display encoding, not new shapes or objects. Half-cell positions do not represent actual displacement.'}
PROMPT='''Compare the original BEFORE and AFTER images, using the third image as supplementary evidence.
The inventory below is an imperfect BEFORE-only description, not a game rule.
Identify visible changes to specific objects. Position changes count even if shape and color are identical.
Distinguish movement, shape change, separation, size change, color change, appearance, and disappearance.
Multiple objects can move in different directions. Do not infer causes or roles. Do not require coordinates, distances, or exact direction.
For each listed entry report a single most informative kind: unchanged, moved, shape_changed, separated, resized, color_changed, appeared, disappeared, or uncertain.
Broad regions may contain changed objects, but naming only a region does not identify a changed object.
Also report X for any changed or new object not individually listed; a broad region is not an individual listing. Use unchanged for X if none.
Return one JSON object with an observations array, one row per listed ID and X: {"id":"A","kind":"unchanged","description":"brief visual evidence"}.
Each description must be at most 12 words. Mention the actual object, not annotation marks. Use only the original images to establish actual state.
BEFORE inventory:
'''

def prepare(out):
    source=Path('outputs/temporal-overlay-preview-20260928')
    cases=json.loads((source/'cases.json').read_text());old=json.loads(Path('outputs/object-gate-20260928/catalogs.json').read_text())
    cats=[];jobs=[];gallery=['<!doctype html><meta charset="utf-8"><title>Overlay benchmark inputs</title><style>body{font:16px system-ui}img{width:19%;max-width:320px}</style>']
    for c in cases:
        name=c['name'];gallery.append('<h2>'+name+'</h2>')
        media={m:encode(Image.open(out/f'{name}-{m}.png').convert('RGB')) for m in ['before','after','blend','outline','split']}
        for m in media:gallery.append(f'<img title="{m}" src="{name}-{m}.png">')
        for rep in range(3):
            cat=deepcopy(next(x for x in old if x['case']==('ls20_up' if name=='ls20_up' else 'synthetic_static') and x['repeat']==rep))
            cat.update(case=name,key=f'{name}:{rep}');truth={x['id']:'unchanged' for x in cat['items']};truth['X']='unchanged';movers=[]
            if name=='ls20_up':
                truth['E']='resized';mover=['D','C','X'][rep];truth[mover]='moved';movers=[mover]
            elif name.startswith('block_'):truth['D']='moved';movers=['D']
            elif name.startswith('cross_'):truth['B']='moved';movers=['B']
            elif name=='opposite_1px':truth['B']=truth['D']='moved';movers=['B','D']
            elif name=='synthetic_reshape':truth['D']='shape_changed'
            elif name=='synthetic_split':truth['D']='separated'
            elif name=='synthetic_new':truth['X']='appeared'
            elif name=='color_only':truth['B']='color_changed'
            cat.update(truth=truth,mover_ids=movers);cats.append(cat)
            for arm in ARMS:
                content=[{'type':'text','text':'BEFORE'},media['before'],{'type':'text','text':'AFTER'},media['after'],
                         {'type':'text','text':LEGEND[arm]},media['after' if arm=='repeat' else arm]]
                prompt=PROMPT+inventory_text(cat)+'\nX: any changed or new object NOT individually listed above; use unchanged if none.\nInclude every ID above, including X, exactly once.'
                p=payload(content,prompt,rep,512);p['response_format']={'type':'json_object'}
                jobs.append({'id':len(jobs),'case':name,'repeat':rep,'arm':arm,'catalog_key':cat['key'],'payload':p,'payload_digest':digest(p)})
    save_json(out/'cases.json',cases);save_json(out/'catalogs.json',cats);save_json(out/'jobs.json',jobs);(out/'gallery.html').write_text('\n'.join(gallery))
    sources=[Path(__file__),Path('scripts/render_temporal_benchmark.cjs'),Path('scripts/preview_temporal_overlay.html'),Path('scripts/benchmark_object_gate.py'),Path('scripts/benchmark_attention_selection.py'),Path('scripts/benchmark_parallel.py'),Path('scripts/benchmark_recognition.py')]
    (out/'sources').mkdir();hashes={}
    for f in sources:hashes[str(f)]=hashlib.sha256(f.read_bytes()).hexdigest();shutil.copyfile(f,out/'sources'/f.name)
    save_json(out/'plan.json',{'created_at':now(),'jobs_digest':digest(jobs),'cases_digest':digest(cases),'catalog_digest':digest(cats),'source_hashes':hashes,'requests':len(jobs),'warmups':4,
        'design':'11 image pairs, 4 arms, 3 frozen model-generated inventories; original BEFORE/AFTER + one supplementary image. Same 640x820 retro UI and 512x512 board in every image. No title, truth, motion magnitude or action supplied.',
        'generation':'Same Qwen3-VL-4B-Instruct Q4_K_M, greedy temperature0, top_p1 top_k0 min_p0 repeat_penalty1 presence_penalty0; 512tokens,20seconds,serial,cache_prompt false. No native fast/slow switch. Replicates use different frozen inventories, not independent stochastic trials.',
        'primary':'Actual moving target identified as moved, with matching appearance description. Exact direction/distance/coordinates not graded. All movers needed for case success. X can recover missing inventory target.',
        'secondary':'Changed foreground detection, correct change kind for non-motion cases, unchanged-object false positives, annotation-as-object hallucinations, latency and invalid outputs.',
        'review':'Classification scored against frozen truth, then arm-hidden shuffled semantic audit of every response. Implementing assistant, not independent human review.',
        'limitations':'One real pair, ten synthetic pairs, no gameplay, no inferred causality; inventory acquisition excluded. Compare only current arms, not earlier benchmark rates.'})
    print('Prepared',len(jobs))

def run(out,base):
    jobs=json.loads((out/'jobs.json').read_text());plan=json.loads((out/'plan.json').read_text());assert digest(jobs)==plan['jobs_digest']
    for f,h in plan['source_hashes'].items():assert hashlib.sha256(Path(f).read_bytes()).hexdigest()==h
    save_json(out/'server-before.json',http(base.removesuffix('/v1')+'/props'));assert not (out/'measurements.jsonl').exists()
    shuffled=list(jobs);random.Random(280937).shuffle(shuffled)
    warm=[next(j for j in jobs if j['arm']==a) for a in ARMS]
    allowed={'unchanged','moved','shape_changed','separated','resized','color_changed','appeared','disappeared','uncertain'}
    cats={c['key']:c for c in json.loads((out/'catalogs.json').read_text())}
    with (out/'measurements.jsonl').open('w') as file:
        for i,(j,iswarm) in enumerate([(x,True) for x in warm]+[(x,False) for x in shuffled]):
            r={k:v for k,v in j.items() if k!='payload'};r.update(warmup=iswarm,started_at=now());start=time.monotonic()
            try:
                result=http(base+'/chat/completions',j['payload'],20);r['response']=result
                choice=result['choices'][0];r['answer']=choice['message']['content'];parsed=json.loads(r['answer']);r['parsed']=parsed
                rows=parsed['observations'];ids=[x['id'] for x in rows];expected=[x['id'] for x in cats[j['catalog_key']]['items']]+['X']
                r['valid']=choice['finish_reason']=='stop' and sorted(ids)==sorted(expected) and all(x['kind'] in allowed and isinstance(x['description'],str) for x in rows)
            except Exception as e:r.update(valid=False,error=f'{type(e).__name__}: {e}')
            r['seconds']=time.monotonic()-start;r['within_budget']=r['seconds']<=20
            file.write(json.dumps(r,ensure_ascii=False)+'\n');file.flush();print(i,j['case'],j['arm'],r['valid'],round(r['seconds'],2),flush=True)
            if iswarm and not r['valid']:raise RuntimeError('Warmup format invalid; inspect before collecting main measurements')
    save_json(out/'server-after.json',http(base.removesuffix('/v1')+'/props'))
    rows=[r for r in read_lines(out/'measurements.jsonl') if not r['warmup']];random.Random(33711).shuffle(rows)
    save_json(out/'review-key.json',{str(i):r['id'] for i,r in enumerate(rows)})
    save_json(out/'review-candidates.json',[{'review_id':i,'case':r['case'],'answer':r.get('answer',''),'valid':r['valid']} for i,r in enumerate(rows)])
    print('Completed',len(rows),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('command',choices=['prepare','run']);ap.add_argument('--output',required=True,type=Path);ap.add_argument('--base',default='http://127.0.0.1:8080/v1');a=ap.parse_args()
    if a.command=='prepare':prepare(a.output)
    else:run(a.output,a.base)
