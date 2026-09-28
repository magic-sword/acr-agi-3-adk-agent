"""Frozen category-free SAM/Qwen proposal comparison; no game actions.

SAM code/checkpoint live in outputs/sam-deps. Inference never reads gold.
Candidate caps are post-generation budgets, not inference-time early stopping.
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from PIL import Image, ImageDraw
from scripts.benchmark_iterative_proposals import score, sha, save, http, native_parse

SOURCE = Path('outputs/iterative-proposals-20260928')
DEPS = Path('outputs/sam-deps')
CAPS = (16, 32, 64, 128)
CONFIGS = {
    'sam_standard': dict(points_per_side=32, points_per_batch=32,
                         crop_n_layers=0, crop_n_points_downscale_factor=1),
    'sam_crops': dict(points_per_side=32, points_per_batch=32,
                      crop_n_layers=1, crop_n_points_downscale_factor=2),
}
COMMON = dict(pred_iou_thresh=.88, stability_score_thresh=.95,
              box_nms_thresh=.7, crop_nms_thresh=.7, crop_overlap_ratio=512/1500,
              min_mask_region_area=0, output_mode='binary_mask')


def normalized_mask_box(mask):
    """Use outer pixel edges, including the rightmost/bottommost mask pixel."""
    yy, xx = mask.nonzero()
    if len(xx) == 0:
        raise ValueError('empty mask')
    h, w = mask.shape
    return [float(xx.min())/w*1000, float(yy.min())/h*1000,
            float(xx.max()+1)/w*1000, float(yy.max()+1)/h*1000]


def ranked(records):
    # SAM quality is predicted mask IoU, not probability of semantic objecthood.
    return sorted(records, key=lambda r: (-r['predicted_iou'], -r['stability_score'], r['index']))


def prepare(out):
    out.mkdir(parents=True, exist_ok=False)
    (out/'sources').mkdir()
    cases = json.loads((SOURCE/'cases.json').read_text())
    for name in ('cases.json', 'native-payloads.json'):
        shutil.copyfile(SOURCE/name, out/name)
    for c in cases:
        shutil.copyfile(SOURCE/(c['name']+'.png'), out/(c['name']+'.png'))
    jobs = [dict(case=c['name'], arm=a) for c in cases for a in CONFIGS]
    random.Random(932101).shuffle(jobs)
    save(out/'jobs.json', jobs)
    sources = list(Path('scripts').glob('benchmark_*.py'))
    for p in sources:
        shutil.copyfile(p, out/'sources'/p.name)
    model = DEPS/'sam_vit_b_01ec64.pth'
    external = {str(p):sha(p) for p in (DEPS/'segment-anything'/'segment_anything').rglob('*.py')}
    save(out/'plan.json', dict(created=datetime.now(timezone.utc).isoformat(),
        model='SAM ViT-B, official sam_vit_b_01ec64.pth, float32 CUDA, eval, no AMP',
        checkpoint=str(model), checkpoint_sha256=sha(model), configs=CONFIGS, common=COMMON,
        caps=CAPS, rank='predicted_iou descending, stability descending, original index ascending',
        coordinate_rule='outer pixel edges from mask support, normalized 0..1000',
        primary='Existing score: +1 board pixel padding, >=90% support coverage, <=9x gold bbox area, one-to-one matching.',
        secondary='No padding; area cap4; group/case recall; duplicate and mixed boxes; program baseline.',
        timing='Wall time including input decoding, image embedding, point decoding, masks, ranking and box conversion; excludes model load, warmup, saving, scoring. CUDA synchronize at both ends. Qwen HTTP includes server processing and response transfer; identical frozen prompts. One repeat. No simultaneous model requests.',
        scope='20 synthetic boards/80 objects; 3 partially annotated real boards/17 targets. No category/count/gold supplied to models. Candidate cap applied AFTER full generation, same latency at all caps.',
        warmup='Unscored fixed RGB image with red square, once per SAM configuration; Qwen first frozen request once, response excluded.',
        frozen={p.name:sha(p) for p in [out/'cases.json',out/'jobs.json',out/'native-payloads.json',*out.glob('*.png')]},
        source_hashes={str(p):sha(p) for p in sources}, external_source_hashes=external))
    print('Prepared', len(jobs), 'SAM jobs and 23 Qwen jobs', flush=True)


def verify(out, model=False):
    plan = json.loads((out/'plan.json').read_text())
    for p, h in plan['frozen'].items():
        assert sha(out/p) == h, p
    for p, h in plan['source_hashes'].items():
        assert sha(p) == h and sha(out/'sources'/Path(p).name) == h, p
    for p, h in plan['external_source_hashes'].items():
        assert sha(p) == h, p
    if model:
        assert sha(plan['checkpoint']) == plan['checkpoint_sha256']
    return plan


def run_sam(out):
    plan = verify(out, model=True)
    import numpy as np
    import torch
    import torchvision
    sys.path.insert(0, str((DEPS/'segment-anything').resolve()))
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
    torch.set_num_threads(4)
    torch.manual_seed(932101)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    started = time.monotonic()
    sam = sam_model_registry['vit_b'](checkpoint=plan['checkpoint']).to('cuda').eval()
    torch.cuda.synchronize()
    meta = dict(torch=torch.__version__, torchvision=torchvision.__version__, numpy=np.__version__,
                python=sys.version, gpu=torch.cuda.get_device_name(), dtype='float32',
                threads=torch.get_num_threads(), model_load_seconds=time.monotonic()-started)
    generators = {a:SamAutomaticMaskGenerator(sam, **cfg, **plan['common'])
                  for a, cfg in plan['configs'].items()}
    warm = np.full((384,384,3), 120, dtype=np.uint8)
    warm[100:160,100:160] = [230,20,20]
    meta['warmups'] = {}
    for a, gen in generators.items():
        t = time.monotonic()
        with torch.inference_mode(): gen.generate(warm)
        torch.cuda.synchronize()
        meta['warmups'][a] = time.monotonic()-t
    save(out/'sam-environment.json', meta)
    (out/'masks').mkdir(exist_ok=True)
    with (out/'sam-results.jsonl').open('x') as log:
        for i, j in enumerate(json.loads((out/'jobs.json').read_text())):
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start = time.monotonic()
            im = np.array(Image.open(out/(j['case']+'.png')).convert('RGB'))
            with torch.inference_mode(): anns = generators[j['arm']].generate(im)
            records = []
            for k, a in enumerate(anns):
                record = {key:value for key,value in a.items() if key != 'segmentation'}
                record.update(index=k, box=normalized_mask_box(a['segmentation']))
                records.append(record)
            records = ranked(records)
            torch.cuda.synchronize()
            seconds = time.monotonic()-start
            masks_path = 'masks/'+j['case']+'-'+j['arm']+'.npz'
            np.savez_compressed(out/masks_path, **{str(k):a['segmentation'] for k,a in enumerate(anns)})
            result = dict(**j, seconds=seconds, records=records, boxes=[r['box'] for r in records],
                          masks=masks_path, masks_sha256=sha(out/masks_path),
                          peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                          peak_reserved_bytes=torch.cuda.max_memory_reserved())
            log.write(json.dumps(result)+'\n'); log.flush()
            print(i+1, '/', 46, j['case'], j['arm'], len(records), round(seconds,3), flush=True)


def run_qwen(out):
    verify(out)
    base = 'http://127.0.0.1:8080'
    controls = json.loads((out/'native-payloads.json').read_text())
    save(out/'qwen-server-before.json', http(base+'/props'))
    save(out/'qwen-warmup.json', http(base+'/v1/chat/completions', next(iter(controls.values()))))
    names = list(controls); random.Random(932102).shuffle(names)
    with (out/'qwen-results.jsonl').open('x') as log:
        for name in names:
            start = time.monotonic()
            response = http(base+'/v1/chat/completions', controls[name])
            choice = response['choices'][0]
            assert choice['finish_reason'] == 'stop'
            boxes = [o['bbox'] for o in native_parse(choice['message']['content'], 'normalized_board')]
            r = dict(case=name, arm='qwen', boxes=boxes, response=response, seconds=time.monotonic()-start)
            log.write(json.dumps(r)+'\n'); log.flush()
            print(name, len(boxes), round(r['seconds'],3), flush=True)
    save(out/'qwen-server-after.json', http(base+'/props'))


def analyze(out):
    verify(out)
    cases = {c['name']:c for c in json.loads((out/'cases.json').read_text())}
    rows = [json.loads(line) for f in ('sam-results.jsonl','qwen-results.jsonl')
            for line in (out/f).read_text().splitlines()]
    assert len(rows) == 69 and len({(r['case'],r['arm']) for r in rows}) == 69
    for c in cases.values():
        rows.append(dict(case=c['name'],arm='program',seconds=c['program_seconds'],
            boxes=[[v/64*1000 for v in (o['bbox'][0],o['bbox'][1],o['bbox'][2]+1,o['bbox'][3]+1)]
                   for o in c['program']['instances']]))
    scored = []
    for r in rows:
        c = cases[r['case']]
        for cap in CAPS:
            boxes = r['boxes'][:cap]
            scored.append(dict(case=r['case'], arm=r['arm'], cap=cap, group=c['group'],
                fully_annotated=c['fully_annotated'], seconds=r['seconds'], pool=len(r['boxes']),
                main=score(boxes,c['gold']), raw=score(boxes,c['gold'],padding=0),
                cap4=score(boxes,c['gold'],cap=4)))
    save(out/'scored.json', scored)
    summary = []
    for arm in (*CONFIGS,'qwen','program'):
        for cap in CAPS:
            for full in (True, False):
                rr = [r for r in scored if r['arm']==arm and r['cap']==cap and r['fully_annotated']==full]
                summary.append(dict(arm=arm,cap=cap,dataset='synthetic' if full else 'real_partial',
                    images=len(rr), gold=sum(r['main']['gold'] for r in rr),
                    tp=sum(r['main']['tp'] for r in rr), raw_tp=sum(r['raw']['tp'] for r in rr),
                    cap4_tp=sum(r['cap4']['tp'] for r in rr),
                    proposals=sum(r['main']['predicted'] for r in rr),
                    pool=sum(r['pool'] for r in rr), median_seconds=statistics.median(r['seconds'] for r in rr),
                    min_seconds=min(r['seconds'] for r in rr), max_seconds=max(r['seconds'] for r in rr),
                    mixed=sum(r['main']['mixed_boxes'] for r in rr),
                    duplicate=sum(r['main']['duplicate_eligible'] for r in rr)))
    save(out/'summary.json',summary)
    # No gold overlay on inference inputs; reference gold only in review gallery.
    (out/'review').mkdir(exist_ok=True)
    page=['<meta charset="utf-8"><title>SAM/Qwen candidate comparison</title>',
          '<style>body{font:16px sans-serif}section{display:flex;flex-wrap:wrap}figure{margin:8px}img{width:300px}</style>',
          '<h1>Candidate boxes, top 128 (gold shown separately)</h1>']
    for name,c in cases.items():
        page.append('<h2>'+html.escape(name)+'</h2><section>')
        entries=[('gold',[[v/64*1000 for v in (g['bbox'][0],g['bbox'][1],g['bbox'][2]+1,g['bbox'][3]+1)] for g in c['gold']])]
        entries += [(r['arm'],r['boxes'][:128]) for r in rows if r['case']==name]
        for arm,boxes in entries:
            im=Image.open(out/(name+'.png')).convert('RGB'); draw=ImageDraw.Draw(im)
            for i,b in enumerate(boxes):
                xy=[v*384/1000 for v in b]
                draw.rectangle(xy,outline='#ff00ff',width=1)
                draw.text((xy[0],xy[1]),str(i+1),fill='#ff00ff')
            filename='review/'+name+'-'+arm+'.png'; im.save(out/filename)
            page.append(f'<figure><figcaption>{arm}: {len(boxes)} boxes</figcaption><img src="{filename}"></figure>')
        page.append('</section>')
    (out/'gallery.html').write_text('\n'.join(page))
    print(json.dumps(summary,indent=2))


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=['prepare','sam','qwen','analyze'])
    parser.add_argument('--output',type=Path,default=Path('outputs/sam-proposals-20260928'))
    args=parser.parse_args()
    {'prepare':prepare,'sam':run_sam,'qwen':run_qwen,'analyze':analyze}[args.command](args.output)
