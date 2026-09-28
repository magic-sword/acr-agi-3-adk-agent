"""Predeclared EdgeBoxes recall/latency comparison with frozen SAM inputs.

Inference sees RGB pixels only. Gold is accessed only by analyze/audit.
Uses isolated opencv-contrib-python-headless, never changes the game runtime.
"""
import argparse
from datetime import datetime, timezone
import html
import json
from pathlib import Path
import random
import shutil
import statistics
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from PIL import Image,ImageDraw
from scripts.benchmark_iterative_proposals import score,sha,save

SOURCE=Path('outputs/sam-proposals-20260928')
DEPS=Path('outputs/edgeboxes-deps')
REPEATS=5
CONFIGS={'default':dict(minBoxArea=1000.,maxAspectRatio=3.),
         'small':dict(minBoxArea=36.,maxAspectRatio=3.),
         'small_wide':dict(minBoxArea=36.,maxAspectRatio=32.)}
COMMON=dict(alpha=.65,beta=.75,eta=1.,minScore=.01,edgeMinMag=.1,
            edgeMergeThr=.5,clusterMinMag=.5,gamma=2.,kappa=1.5)
BUDGETS=(128,1000)


def normalize_xywh(box,width,height):
    x,y,w,h=map(int,box)
    if not (0<=x<x+w<=width and 0<=y<y+h<=height):
        raise ValueError('invalid OpenCV rectangle')
    return [x/width*1000,y/height*1000,(x+w)/width*1000,(y+h)/height*1000]


def prepare(out):
    out.mkdir(parents=True,exist_ok=False);(out/'sources').mkdir()
    cases=json.loads((SOURCE/'cases.json').read_text())
    for name in ['cases.json','sam-results.jsonl','qwen-results.jsonl','sam-environment.json']:
        shutil.copyfile(SOURCE/name,out/name)
    for c in cases:shutil.copyfile(SOURCE/(c['name']+'.png'),out/(c['name']+'.png'))
    jobs=[dict(case=c['name'],arm=a,budget=b,repeat=r)
          for c in cases for a in CONFIGS for b in BUDGETS for r in range(REPEATS)]
    random.Random(933101).shuffle(jobs);save(out/'jobs.json',jobs)
    sources=list(Path('scripts').glob('benchmark_*.py'))
    for p in sources:shutil.copyfile(p,out/'sources'/p.name)
    external=[DEPS/'model.yml.gz',DEPS/'edgeboxes.cpp',DEPS/'edgeboxes.hpp',DEPS/'edgeboxes_demo.py',
              DEPS/'python/cv2/cv2.abi3.so']
    save(out/'plan.json',dict(created=datetime.now(timezone.utc).isoformat(),configs=CONFIGS,common=COMMON,
        budgets=BUDGETS,repeats=REPEATS,cv_threads=4,
        purpose='Same 23 images and relaxed retrieval metric as SAM, no category/count/gold prompts.',
        limits='20 synthetic/80 objects; 3 real partial/17 targets. SAM/Qwen outputs and single-repeat timings reused from prior measurement, not rerun.',
        pipeline='RGB float32/255 -> official StructuredEdgeDetection -> computeOrientation -> edgesNms defaults -> EdgeBoxes. No Canny substitution, no label-dependent filtering.',
        coordinate_rule='Use returned OpenCV Rect x,y,w,h as [x,y,x+w,y+h] outer edges, normalized to0..1000, no oracle correction or clipping.',
        parameters='Official defaults except maxBoxes; small lowers initial search minBoxArea to36 (one original board pixel at6x display scale); small_wide additionally allows aspect32 for thin regions. Fixed before any dataset inference.',
        caps='Native generation budgets128 and1000 independently measured. Within128: top16/32/64/128 post-selection, same generation time.1000 is a separate larger-budget diagnostic.',
        timing='5 repeats/image/config/budget, random interleaving. End-to-end includes PNG load, RGB conversion, edge detection, orientation/NMS, box search, ranking, normalization. Excludes model load, warmup, saving and scoring. CPU only. Dataset latency=median of per-image medians.',
        warmup='Unscored gray384x384 image with red square, each configuration/budget once.',
        frozen={p.name:sha(p) for p in [out/'cases.json',out/'jobs.json',out/'sam-results.jsonl',out/'qwen-results.jsonl',out/'sam-environment.json',*out.glob('*.png')]},
        sources={str(p):sha(p) for p in sources},external={str(p):sha(p) for p in external}))
    print('Prepared',len(jobs),'timed requests',flush=True)


def verify(out):
    p=json.loads((out/'plan.json').read_text())
    for name,h in p['frozen'].items():assert sha(out/name)==h,name
    for name,h in p['sources'].items():assert sha(name)==h and sha(out/'sources'/Path(name).name)==h,name
    for name,h in p['external'].items():assert sha(name)==h,name
    return p


def run(out):
    plan=verify(out)
    sys.path.insert(0,str((DEPS/'python').resolve()))
    import cv2
    import numpy as np
    assert str((DEPS/'python').resolve()) in cv2.__file__
    cv2.setNumThreads(plan['cv_threads']);cv2.ocl.setUseOpenCL(False)
    start=time.perf_counter()
    detector=cv2.ximgproc.createStructuredEdgeDetection(str(DEPS/'model.yml.gz'))
    load_seconds=time.perf_counter()-start
    generators={(a,b):cv2.ximgproc.createEdgeBoxes(**plan['common'],**cfg,maxBoxes=b)
                for a,cfg in plan['configs'].items() for b in plan['budgets']}
    warm=np.full((384,384,3),120,dtype=np.uint8);warm[100:160,100:160]=[230,20,20]
    def generate(im,a,b):
        marks=[time.perf_counter()]
        rgb=np.asarray(im,dtype=np.float32)/255.
        edges=detector.detectEdges(rgb);marks.append(time.perf_counter())
        orientation=detector.computeOrientation(edges)
        nms=detector.edgesNms(edges,orientation);marks.append(time.perf_counter())
        rects,values=generators[(a,b)].getBoundingBoxes(nms,orientation)
        marks.append(time.perf_counter())
        records=[dict(index=i,rect=list(map(int,rect)),score=float(value),
                      box=normalize_xywh(rect,im.shape[1],im.shape[0]))
                 for i,(rect,value) in enumerate(zip(rects,np.asarray(values).reshape(-1)))]
        records.sort(key=lambda r:(-r['score'],r['index']))
        return records,nms,dict(edge_seconds=marks[1]-marks[0],orientation_nms_seconds=marks[2]-marks[1],
                               box_seconds=marks[3]-marks[2])
    for a,b in generators:generate(warm,a,b)
    save(out/'environment.json',dict(python=sys.version,opencv=cv2.__version__,opencv_file=cv2.__file__,
        numpy=np.__version__,cv_threads=cv2.getNumThreads(),opencl=cv2.ocl.useOpenCL(),
        model_load_seconds=load_seconds,build_info=cv2.getBuildInformation(),
        cpuinfo=next(s.split(':',1)[1].strip() for s in Path('/proc/cpuinfo').read_text().splitlines() if s.startswith('model name'))))
    (out/'edges').mkdir()
    jobs=json.loads((out/'jobs.json').read_text())
    with (out/'results.jsonl').open('x') as log:
        for i,j in enumerate(jobs):
            start=time.perf_counter()
            im=np.array(Image.open(out/(j['case']+'.png')).convert('RGB'))
            records,nms,stages=generate(im,j['arm'],j['budget'])
            seconds=time.perf_counter()-start
            r=dict(**j,seconds=seconds,stages=stages,records=records,boxes=[v['box'] for v in records])
            log.write(json.dumps(r)+'\n');log.flush()
            if j['repeat']==0 and j['arm']=='default' and j['budget']==128:
                np.save(out/'edges'/(j['case']+'.npy'),nms)
                Image.fromarray(np.uint8(np.clip(nms*255,0,255))).save(out/'edges'/(j['case']+'.png'))
            if (i+1)%23==0:print(i+1,'/',len(jobs),j['arm'],len(records),round(seconds,4),flush=True)


def read_rows(out,name):return [json.loads(line) for line in (out/name).read_text().splitlines()]


def analyze(out):
    plan=verify(out);cases={c['name']:c for c in json.loads((out/'cases.json').read_text())}
    rows=read_rows(out,'results.jsonl');jobs=json.loads((out/'jobs.json').read_text())
    assert len(rows)==len(jobs)
    assert [{k:r[k] for k in ('case','arm','budget','repeat')} for r in rows]==jobs
    selected=[];repeat_agreement=0
    for name in cases:
        for arm in CONFIGS:
            for budget in BUDGETS:
                rr=[r for r in rows if (r['case'],r['arm'],r['budget'])==(name,arm,budget)]
                assert len(rr)==REPEATS and {r['repeat'] for r in rr}==set(range(REPEATS))
                first=next(r for r in rr if r['repeat']==0)
                for r in rr:assert r['records']==first['records'];repeat_agreement+=1
                selected.append(dict(case=name,arm=arm,budget=budget,boxes=first['boxes'],
                    seconds=statistics.median(r['seconds'] for r in rr),
                    stage_medians={k:statistics.median(r['stages'][k] for r in rr) for k in first['stages']},
                    min_seconds=min(r['seconds'] for r in rr),max_seconds=max(r['seconds'] for r in rr)))
    for file in ('sam-results.jsonl','qwen-results.jsonl'):
        for r in read_rows(out,file):selected.append(dict(case=r['case'],arm=r['arm'],budget=128,boxes=r['boxes'],seconds=r['seconds']))
    scored=[]
    for r in selected:
        c=cases[r['case']]
        for cap in ([16,32,64,128] if r['budget']==128 else [1000]):
            boxes=r['boxes'][:cap]
            scored.append(dict(case=r['case'],arm=r['arm'],budget=r['budget'],cap=cap,group=c['group'],
                fully_annotated=c['fully_annotated'],pool=len(r['boxes']),seconds=r['seconds'],
                main=score(boxes,c['gold']),raw=score(boxes,c['gold'],padding=0),cap4=score(boxes,c['gold'],cap=4)))
    summary=[]
    for arm,budget,cap in sorted({(r['arm'],r['budget'],r['cap']) for r in scored}):
        for full in (True,False):
            rr=[r for r in scored if (r['arm'],r['budget'],r['cap'],r['fully_annotated'])==(arm,budget,cap,full)]
            summary.append(dict(arm=arm,budget=budget,cap=cap,dataset='synthetic' if full else 'real_partial',
                tp=sum(r['main']['tp'] for r in rr),gold=sum(r['main']['gold'] for r in rr),
                raw_tp=sum(r['raw']['tp'] for r in rr),cap4_tp=sum(r['cap4']['tp'] for r in rr),
                proposals=sum(r['main']['predicted'] for r in rr),median_seconds=statistics.median(r['seconds'] for r in rr),
                duplicate=sum(r['main']['duplicate_eligible'] for r in rr),mixed=sum(r['main']['mixed_boxes'] for r in rr)))
    save(out/'selected.json',selected);save(out/'scored.json',scored);save(out/'summary.json',summary)
    save(out/'repeat-audit.json',dict(equal_records=repeat_agreement,expected=len(jobs)))
    (out/'review').mkdir()
    page=['<meta charset="utf-8"><title>EdgeBoxes comparison</title><style>body{font:16px sans-serif}section{display:flex;flex-wrap:wrap}figure{margin:6px}img{width:320px}</style>',
          '<h1>Top 128 boxes; edges and gold shown separately</h1>']
    for name,c in cases.items():
        page.append('<h2>'+html.escape(name)+'</h2><section>')
        entries=[('gold',[[v/64*1000 for v in (g['bbox'][0],g['bbox'][1],g['bbox'][2]+1,g['bbox'][3]+1)] for g in c['gold']])]
        entries += [(r['arm'],r['boxes'][:128]) for r in selected if r['case']==name and r['budget']==128]
        for arm,boxes in entries:
            im=Image.open(out/(name+'.png')).convert('RGB');draw=ImageDraw.Draw(im)
            for i,b in enumerate(boxes):
                xy=[v*384/1000 for v in b];draw.rectangle(xy,outline='#ff00ff',width=1)
                draw.text((xy[0],xy[1]),str(i+1),fill='#ff00ff')
            filename='review/'+name+'-'+arm+'.png';im.save(out/filename)
            page.append(f'<figure><figcaption>{arm}: {len(boxes)} boxes</figcaption><img src="{filename}"></figure>')
        page.append(f'<figure><figcaption>Structured edges (fixed scale)</figcaption><img src="edges/{name}.png"></figure></section>')
    (out/'gallery.html').write_text('\n'.join(page))
    print('Analyzed',len(rows),'requests;',len(scored),'scores',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['prepare','run','analyze'])
    parser.add_argument('--output',type=Path,default=Path('outputs/edgeboxes-20260928'))
    a=parser.parse_args();{'prepare':prepare,'run':run,'analyze':analyze}[a.command](a.output)
