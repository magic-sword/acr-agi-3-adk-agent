"""Frozen temporal ablation: program, SAM once, adaptive SAM, SAM every frame."""
import argparse
from copy import deepcopy
from datetime import datetime,timezone
import json
from pathlib import Path
import random
import shutil
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from PIL import Image,ImageColor
from scripts.benchmark_iterative_proposals import save,sha
from scripts.temporal_proposal_tracker import PixelTracker,native_boxes

DEPS=Path('outputs/sam-deps')
BASE=Path('outputs/sam-proposals-20260928')
ARMS=('program','sam_initial','sam_adaptive','sam_every')


def bounds(points):return [min(p[0] for p in points),min(p[1] for p in points),max(p[0] for p in points),max(p[1] for p in points)]


def synthetic():
    cross=[[9 if x==3 else 12 if y==3 else None for x in range(7)] for y in range(7)]
    ell=[[9 if x<2 else 12 if y>=5 else None for x in range(7)] for y in range(7)]
    rect=[[9]*6 for _ in range(3)]+[[12]*6 for _ in range(3)]
    square=[[8]*5 for _ in range(5)]
    tiny=[[None,0,None],[1,0,0],[None,1,None]]
    result=[]
    path=[0,1,2,3,4,5,4,3,2,1,0]
    for name in ['rigid_multicolor','static_and_moving','contact','appearance_change','birth_death','identical_copies']:
        frames=[]
        for t in range(11):
            objects=[('g1',8,8,deepcopy(cross)),('g2',38,8,deepcopy(ell)),('g3',8,38,deepcopy(rect)),('g4',38,38,deepcopy(square))]
            if name=='rigid_multicolor':objects=[(i,x+path[t],y+path[t]//2,p) for i,x,y,p in objects]
            elif name=='static_and_moving':objects=[(i,x+(path[t] if i in ['g3','g4'] else 0),y,p) for i,x,y,p in objects]
            elif name=='contact':objects=[('g1',10+path[t],10,deepcopy(square)),('g2',20,10,deepcopy(square)),('g3',8,38,deepcopy(cross)),('g4',38,38,deepcopy(ell))]
            elif name=='appearance_change':
                objects=[(i,x+path[t],y,[[14 if v==9 and t>=4 else v for v in row] for row in p]) for i,x,y,p in objects]
            elif name=='birth_death' and t>=4:objects=[o for o in objects if o[0]!='g2']+[('g5',38,8,deepcopy(cross))]
            elif name=='identical_copies':
                d=[0,1,2,3,4,5,6,7,6,5,4][t]
                objects=[('g1',12+d,10,deepcopy(tiny)),('g2',32-d,10,deepcopy(tiny)),('g3',8,38,deepcopy(ell)),('g4',38,38,deepcopy(rect))]
            grid=[[5]*64 for _ in range(64)];gold=[]
            for ident,x,y,pattern in objects:
                support=[]
                for yy,row in enumerate(pattern):
                    for xx,v in enumerate(row):
                        if v is None:continue
                        assert grid[y+yy][x+xx]==5
                        grid[y+yy][x+xx]=v;support.append([x+xx,y+yy])
                gold.append(dict(id=ident,support=support,bbox=bounds(support)))
            frames.append(dict(index=t,grid=grid,gold=gold))
        result.append(dict(name=name,kind='synthetic',fully_annotated=True,frames=frames,
            note='Constructed 11-frame diagnostic. Persistent IDs from renderer; no game rules or labels passed to tracker.'))
    return result


def real():
    initial={c['name']:c for c in json.loads((BASE/'cases.json').read_text())}
    result=[]
    for p in sorted(Path('outputs/action-cue-20260927/source-logs').glob('*.observations.jsonl')):
        rows=[json.loads(line) for line in p.read_text().splitlines()]
        name=rows[0]['game_id'].split('-')[0];c=initial[name]
        assert rows[0]['grid']==c['grid']
        frames=[]
        for t,r in enumerate(rows):
            gold=deepcopy(c['gold'])
            for g in gold:
                if name=='ls20' and g['id']=='g2':g['support']=[[x,y-5*min(t,6)] for x,y in g['support']]
                elif (name=='ls20' and g['id']=='g3') or (name=='vc33' and g['id']=='g1'):
                    color=11 if name=='ls20' else 7
                    g['support']=[[x,y] for x,y in g['support'] if r['grid'][y][x]==color]
                elif name=='ls20' and g['id']=='g1':
                    # At steps6..8 the moving rectangle occludes part of the fixed top square.
                    g['support']=[[x,y] for x,y in g['support'] if not (34<=x<=38 and 45-5*min(t,6)<=y<=49-5*min(t,6))]
                g['bbox']=bounds(g['support'])
                # Manual geometry above must account for every changed annotated pixel.
                for x,y in g['support']:
                    yy=y+5*min(t,6) if name=='ls20' and g['id']=='g2' else y
                    assert r['grid'][y][x]==c['grid'][yy][x],(name,t,g['id'],x,y)
            frames.append(dict(index=t,grid=r['grid'],gold=gold,observation_id=r['observation_id'],
                               levels_completed=r['levels_completed']))
        result.append(dict(name=name,kind='real_partial',fully_annotated=False,frames=frames,source=str(p),source_sha256=sha(p),
            note='Manually reviewed contact sheets before inference: ls20 rectangle moves up5 pixels for6 transitions then stays; yellow bar shrinks. vc33 top pink line shrinks. ft09 unchanged. Preserve previous partial object annotations; remove visible occlusion from fixed top-square support.'))
    return result


def prepare(out):
    out.mkdir(parents=True,exist_ok=False);(out/'images').mkdir();(out/'sources').mkdir()
    seqs=synthetic()+real()
    palette=json.loads(Path('outputs/instance-state-20260928-final/cases.json').read_text())[0]['palette']
    save(out/'sequences.json',seqs);save(out/'palette.json',palette)
    jobs=[]
    for s in seqs:
        for f in s['frames']:
            name=s['name']+'-'+str(f['index'])
            im=Image.new('RGB',(64,64));im.putdata([ImageColor.getrgb(palette[str(c)]) for row in f['grid'] for c in row])
            im.resize((384,384),getattr(Image,'Resampling',Image).NEAREST).save(out/'images'/(name+'.png'))
            jobs.append(dict(sequence=s['name'],frame=f['index'],image='images/'+name+'.png'))
    random.Random(934101).shuffle(jobs);save(out/'jobs.json',jobs)
    sources=[Path('scripts/benchmark_temporal_proposals.py'),Path('scripts/temporal_proposal_tracker.py'),Path('agent/cognition/geometry.py'),Path('scripts/benchmark_iterative_proposals.py'),Path('scripts/benchmark_sam_proposals.py')]
    for p in sources:shutil.copyfile(p,out/'sources'/p.name)
    model=DEPS/'sam_vit_b_01ec64.pth'
    save(out/'plan.json',dict(created=datetime.now(timezone.utc).isoformat(),arms=ARMS,frames=len(jobs),
        sam=dict(model='vit_b',checkpoint=str(model),points_per_side=32,points_per_batch=32,pred_iou_thresh=.88,stability_score_thresh=.95,box_nms_thresh=.7,crop_n_layers=0),
        sam_cap=128,cpu_repeats=3,tracker_radius=8,
        comparison='Same program extraction + exact-pixel ROI tracking in all4 arms. Initial SAM adds proposals at frame0; adaptive also calls SAM when current program/tracking reports local unresolved tracks or unexplained changed pixels; every adds SAM every frame. Exact integer-bbox dedup only. No GT online.',
        tracker='Match full integer-grid ROI template within radius8. Unique exact match preserves ID; none or multiple -> abstain/drop. Add fresh program boxes each frame. On fully unchanged frame preserve observed state. No learned tracking, color tolerance, optical flow or persistent occlusion memory. Background/large boxes>256 cells excluded only from refresh triggers and change coverage, not proposals or tracking.',
        evaluation='Frame recall: same padded90% support,9xarea,one-to-one. Raw and4xarea sensitivity. Correct observed displacement links on continuing gold objects; wrong unique-object links/ID switches; initial-target uninterrupted same-ID coverage; births, dropped tracks, ambiguity and refresh frequency. Do not interpret local object labels as game roles.',
        timing='SAM each image once on GPU with warmup; all arms share these saved detector outputs and measured per-frame detector times. CPU pipeline3 repeats/arm/sequence, no parallel inference; sum its wall time and actually required detector times. These are shared-component replay totals, not4 independent end-to-end service runs. Include first detector cost. Model load excluded and reported. No re-detection cost hidden in amortized figures.',
        scope='6 synthetic11-frame sequences, plus23 recorded real frames (9ls20,8vc33,6ft09). Real labels partial; ft09 static; report moving transitions separately. Known geometry diagnostics, not unseen-game validation.',
        frozen={str(p.relative_to(out)):sha(p) for p in [out/'sequences.json',out/'palette.json',out/'jobs.json',*sorted((out/'images').glob('*.png'))]},
        source_hashes={str(p):sha(p) for p in sources},checkpoint_sha256=sha(model),
        sam_source_hashes={str(p):sha(p) for p in (DEPS/'segment-anything/segment_anything').rglob('*.py')}))
    print('Prepared',len(seqs),'sequences,',len(jobs),'frames',flush=True)


def verify(out):
    plan=json.loads((out/'plan.json').read_text())
    for p,h in plan['frozen'].items():assert sha(out/p)==h,p
    for p,h in plan['source_hashes'].items():assert sha(p)==h and sha(out/'sources'/Path(p).name)==h,p
    for p,h in plan['sam_source_hashes'].items():assert sha(p)==h,p
    assert sha(plan['sam']['checkpoint'])==plan['checkpoint_sha256']
    return plan


def run_sam(out):
    plan=verify(out)
    import numpy as np
    import torch
    import torchvision
    sys.path.insert(0,str((DEPS/'segment-anything').resolve()))
    from segment_anything import sam_model_registry,SamAutomaticMaskGenerator
    from scripts.benchmark_sam_proposals import normalized_mask_box
    torch.set_num_threads(4);torch.manual_seed(934101)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    cfg=dict(plan['sam']);kind=cfg.pop('model');checkpoint=cfg.pop('checkpoint')
    start=time.perf_counter();model=sam_model_registry[kind](checkpoint=checkpoint).to('cuda').eval();torch.cuda.synchronize()
    load=time.perf_counter()-start;gen=SamAutomaticMaskGenerator(model,**cfg)
    warm=np.full((384,384,3),120,dtype=np.uint8);warm[100:160,100:160]=[230,20,20]
    with torch.inference_mode():gen.generate(warm)
    torch.cuda.synchronize()
    save(out/'sam-environment.json',dict(torch=torch.__version__,torchvision=torchvision.__version__,numpy=np.__version__,gpu=torch.cuda.get_device_name(),model_load_seconds=load))
    with (out/'sam-results.jsonl').open('x') as log:
        jobs=json.loads((out/'jobs.json').read_text())
        for i,j in enumerate(jobs):
            torch.cuda.synchronize();start=time.perf_counter()
            im=np.array(Image.open(out/j['image']).convert('RGB'))
            with torch.inference_mode():anns=gen.generate(im)
            records=[dict(index=k,box=normalized_mask_box(a['segmentation']),quality=float(a['predicted_iou']),stability=float(a['stability_score'])) for k,a in enumerate(anns)]
            records.sort(key=lambda a:(-a['quality'],-a['stability'],a['index']))
            torch.cuda.synchronize();seconds=time.perf_counter()-start
            row=dict(**j,seconds=seconds,records=records,boxes=[r['box'] for r in records[:plan['sam_cap']]])
            log.write(json.dumps(row)+'\n');log.flush()
            print(i+1,'/',len(jobs),j['sequence'],j['frame'],len(records),round(seconds,3),flush=True)


def run_cpu(out):
    plan=verify(out)
    import numpy as np
    sam={(r['sequence'],r['frame']):r for r in [json.loads(x) for x in (out/'sam-results.jsonl').read_text().splitlines()]}
    assert len(sam)==plan['frames']
    seqs=json.loads((out/'sequences.json').read_text())
    with (out/'results.jsonl').open('x') as log:
        for repeat in range(plan['cpu_repeats']):
            jobs=[(s,a) for s in seqs for a in ARMS];random.Random(934102+repeat).shuffle(jobs)
            for s,arm in jobs:
                tracker=PixelTracker(plan['tracker_radius'])
                for f in s['frames']:
                    start=time.perf_counter();events=tracker.advance(f['grid'])
                    call=arm=='sam_every' or (arm!='program' and f['index']==0) or (arm=='sam_adaptive' and events['refresh_requested'])
                    if call:tracker.add(native_boxes(sam[s['name'],f['index']]['boxes']),'sam')
                    predictions=tracker.output();cpu_seconds=time.perf_counter()-start
                    r=dict(sequence=s['name'],frame=f['index'],arm=arm,repeat=repeat,events=events,tracks=predictions,
                        sam_called=call,cpu_seconds=cpu_seconds,sam_seconds=sam[s['name'],f['index']]['seconds'] if call else 0)
                    log.write(json.dumps(r)+'\n');log.flush()
            print('CPU repeat',repeat+1,'complete',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['prepare','sam','cpu'])
    parser.add_argument('--output',type=Path,default=Path('outputs/temporal-proposals-20260928'))
    a=parser.parse_args();{'prepare':prepare,'sam':run_sam,'cpu':run_cpu}[a.command](a.output)
