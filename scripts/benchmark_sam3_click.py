"""Frozen SAM 3 text-to-mask experiment. Requires authorized local weights and SAM 3."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
from PIL import Image
from scripts import benchmark_click_choices as b

SHORT = [
 'blue angular pattern','orange and blue rectangle','white and gray cross','yellow horizontal bar',
 'blue angular pattern','green triangle','blue square','blue square','yellow vertical line',
 'black horizontal bar with yellow segment','blue square','red circular button','blue square',
 'white and gray patterned square','white and gray patterned square with red center','blue square',
 'red square','green square','white square','white square','yellow C-shaped object',
 'yellow hollow square','yellow horizontal line','yellow and white rectangle','green square','winning button']


def prepare(out):
    cases=json.loads(Path('outputs/click-choices-20260929-v2/cases.json').read_text())
    assert len(cases)==len(SHORT)==26
    out.mkdir(parents=True,exist_ok=False)
    for c,prompt in zip(cases,SHORT):c['short_prompt']=prompt
    b.save(out/'cases.json',cases)
    b.save(out/'plan.json',dict(cases_hash=b.digest(cases),threshold=.5,input_scale=6,
        arms=['original_query','short_concept'],
        protocol='Original frozen query is primary; manually shortened concept is diagnostic and omits relations. '
        'Every query uses a fresh set_image state, no box/point/SAM1/gold inputs. '
        'Top score mask selection primary; also report conservative single-mask-only choice. '
        'Any-candidate correctness is oracle availability, not selected accuracy. No prompt or threshold tuning.',
        scoring='80% purity and recall; native 64x64 annotation. Resize boolean output with nearest neighbor. '
        'No-mask abstention differs from inference errors. Negatives scored separately. No gameplay.'))
    for path in (Path(__file__).resolve(),Path(b.__file__).resolve()):
        (out/path.name).write_bytes(path.read_bytes())
    print('Prepared 26 frozen cases, 52 queries; inference not executed.')


def masks_to_points(masks):
    masks=np.asarray(masks)
    if masks.ndim==4:
        assert masks.shape[1]==1
        masks=masks[:,0]
    assert masks.ndim==3
    result=[]
    for mask in masks:
        assert mask.dtype==bool, 'Expected thresholded SAM 3 masks'
        image=Image.fromarray(mask.astype(np.uint8)).resize((64,64),Image.Resampling.NEAREST)
        ys,xs=np.where(np.asarray(image)>0)
        result.append(set(zip(map(int,xs),map(int,ys))))
    return result


def score(points,scores,gold):
    assert len(points)==len(scores)
    assert all(np.isfinite(s) for s in scores)
    order=sorted(range(len(scores)),key=lambda i:(-scores[i],i))
    selected=points[order[0]] if order else set()
    point=b.point_for(selected,'centroid')
    unique=selected if len(points)==1 else set()
    return dict(candidates=len(points),available=any(b.region_score(p,gold)['correct'] for p in points),
        selected_index=order[0] if order else None,region=b.region_score(selected,gold)['correct'],
        point=point,hit=point is not None and point in gold,
        abstention=not gold and not selected,false_click=not gold and bool(selected),
        single_only_hit=bool(unique) and b.point_for(unique,'centroid') in gold,
        single_only_abstention=not gold and not unique)


def run(out,checkpoint,device):
    import torch
    if device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; no inference executed')
    assert checkpoint.is_file(), 'Provide an authorized local SAM 3 checkpoint'
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    import sam3
    plan=json.loads((out/'plan.json').read_text());cases=json.loads((out/'cases.json').read_text())
    assert b.digest(cases)==plan['cases_hash']
    assert b.sha(Path(__file__))==b.sha(out/Path(__file__).name),'Prepare a new output after changing experiment code'
    assert not (out/'predictions.jsonl').exists(),'Use a fresh output; no silent retries'
    torch.set_num_threads(4);torch.manual_seed(b.SEED)
    begin=time.monotonic()
    model=build_sam3_image_model(checkpoint_path=str(checkpoint),device=device,load_from_HF=False)
    processor=Sam3Processor(model,device=device,confidence_threshold=plan['threshold'])
    def sync():
        if device=='cuda':torch.cuda.synchronize()
    sync()
    b.save(out/'environment.json',dict(torch=torch.__version__,sam3_source=str(sam3.__file__),
        checkpoint_hash=b.sha(checkpoint),device=device,load_seconds=time.monotonic()-begin))
    rows=[]
    with (out/'predictions.jsonl').open('w') as log:
        for c in cases:
            for arm in plan['arms']:
                prompt=c['query'] if arm=='original_query' else c['short_prompt']
                image=b.frame_image(c['grid']).resize((384,384),Image.Resampling.NEAREST)
                sync();start=time.monotonic()
                with torch.inference_mode():
                    state=processor.set_image(image)
                    prediction=processor.set_text_prompt(state=state,prompt=prompt)
                sync();elapsed=time.monotonic()-start
                masks=prediction['masks'].detach().cpu().numpy();scores=prediction['scores'].detach().float().cpu().tolist()
                points=masks_to_points(masks);gold=b.decode(c['gold_runs'])
                np.savez_compressed(out/f'{c["name"]}-{arm}.npz',masks=masks,scores=scores)
                row=dict(case=c['name'],origin=c['origin'],positive=bool(gold),arm=arm,prompt=prompt,
                    seconds=elapsed,scores=scores,**score(points,scores,gold))
                rows.append(row);log.write(json.dumps(row)+'\n');log.flush()
                print(c['name'],arm,'masks',len(points),'hit',row['hit'],flush=True)
    import statistics
    summary={}
    for arm in plan['arms']:
        summary[arm]={}
        for origin in ('all','real','synthetic'):
            group=[r for r in rows if r['arm']==arm and (origin=='all' or r['origin']==origin)]
            summary[arm][origin]=dict(positive=sum(r['positive'] for r in group),negative=sum(not r['positive'] for r in group),
                **{key:sum(r[key] for r in group) for key in ('available','region','hit','abstention','false_click','single_only_hit','single_only_abstention')},
                median_seconds=statistics.median(r['seconds'] for r in group))
    b.save(out/'summary.json',summary)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','run']);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path);p.add_argument('--device',choices=['cuda','cpu'],default='cuda');a=p.parse_args()
    if a.command=='prepare':prepare(a.output)
    else:
        if a.checkpoint is None:p.error('--checkpoint is required for run')
        run(a.output,a.checkpoint,a.device)
