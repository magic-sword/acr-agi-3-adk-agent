"""Paired ablation of background dimming and RGB mixing; no object tracking."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import sys
from PIL import Image

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts import benchmark_temporal_overlay as original
from scripts.benchmark_parallel import digest
from scripts.benchmark_recognition import encode,save_json

ARMS=['repeat','outline','outline_bright','after_edges']
SOURCE=Path('outputs/temporal-overlay-comparison-20260928-final')
LEGENDS={
 'outline_bright':'This third image is a derived overlay, not an observation: 50% BEFORE + 50% AFTER. Unchanged context retains its original brightness. Magenta dashed lines show color boundaries present only in BEFORE; cyan solid lines show boundaries present only in AFTER. Lines are annotations, not game objects. Shared boundaries are not highlighted. No object identity or motion is inferred by this rendering.',
 'after_edges':'This third image is a derived overlay, not an observation: original AFTER colors, with no RGB blending or background dimming. Magenta dashed lines show color boundaries present only in BEFORE; cyan solid lines show boundaries present only in AFTER. Lines are annotations, not game objects. Shared boundaries are not highlighted. No object identity or motion is inferred by this rendering.'}

def prepare(out):
    assert not (out/'jobs.json').exists()
    prior=json.loads((SOURCE/'jobs.json').read_text());jobs=[]
    cases=json.loads((SOURCE/'cases.json').read_text())
    for c in cases:
        for mode in ['before','after','outline']:
            shutil.copyfile(SOURCE/f'{c["name"]}-{mode}.png',out/f'{c["name"]}-{mode}.png')
    for j in prior:
        if j['arm'] not in ['repeat','outline']:continue
        for arm in (['repeat'] if j['arm']=='repeat' else ARMS[1:]):
            row=deepcopy(j);row.update(id=len(jobs),arm=arm)
            if arm in LEGENDS:
                content=row['payload']['messages'][1]['content']
                content[4]['text']=LEGENDS[arm]
                content[5]=encode(Image.open(out/f'{j["case"]}-{arm}.png').convert('RGB'))
            row['payload_digest']=digest(row['payload']);jobs.append(row)
    save_json(out/'jobs.json',jobs)
    for name in ['cases.json','catalogs.json']:shutil.copyfile(SOURCE/name,out/name)
    files=[Path(__file__),Path('scripts/benchmark_temporal_overlay.py'),Path('scripts/render_overlay_ablation.cjs'),Path('scripts/summarize_temporal_overlay.py'),Path('outputs/temporal-overlay-preview-20260928/preview.html')]
    (out/'sources').mkdir();hashes={}
    for f in files:
        hashes[str(f)]=hashlib.sha256(f.read_bytes()).hexdigest();shutil.copyfile(f,out/'sources'/f.name)
    plan=json.loads((SOURCE/'plan.json').read_text())
    plan.update(created_at=original.now(),arms=ARMS,source_hashes=hashes,jobs_digest=digest(jobs),
        design='11 frozen pairs x 3 frozen BEFORE inventories x 4 arms. Repeat and outline payloads identical to previous experiment. outline_bright removes background dimming only; after_edges additionally replaces RGB blend with original AFTER. Same boundary lines, resolution, UI, image count, main prompt and generation settings.',
        comparison_source=str(SOURCE),requests=len(jobs),warmups=4)
    save_json(out/'plan.json',plan)
    manifest={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(out.glob('*.png'))}
    manifest.update({str(SOURCE/n):hashlib.sha256((SOURCE/n).read_bytes()).hexdigest() for n in ['jobs.json','cases.json','catalogs.json','input-manifest.json']})
    save_json(out/'input-manifest.json',manifest)
    gallery=['<!doctype html><meta charset="utf-8"><title>重ね合わせ加工の比較</title><style>body{font:16px system-ui;background:#eef1f5;margin:24px}.row{display:flex;gap:12px;overflow:auto}figure{margin:0;min-width:240px}img{width:280px;image-rendering:pixelated}figcaption{font-weight:bold}</style><h1>背景の暗色化・色混合の比較</h1><p>同じ元画像に補助画像を1枚追加。境界は色の境界であり、物体の対応付けではありません。画像をクリックすると原寸表示。</p>']
    labels={'before':'元のBEFORE','after':'元のAFTER／再提示','outline':'従来：混合＋境界／背景70%','outline_bright':'混合＋境界／背景100%','after_edges':'原色AFTER＋前後境界'}
    for c in cases:
        gallery.append('<h2>'+c['name']+'</h2><div class="row">')
        for m,label in labels.items():
            f=f'{c["name"]}-{m}.png';gallery.append(f'<figure><figcaption>{label}</figcaption><a href="{f}"><img src="{f}"></a></figure>')
        gallery.append('</div>')
    (out/'gallery.html').write_text('\n'.join(gallery))
    print('Prepared',len(jobs),'requests')

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('command',choices=['prepare','run']);ap.add_argument('--output',required=True,type=Path);ap.add_argument('--base',default='http://127.0.0.1:8080/v1');a=ap.parse_args()
    if a.command=='prepare':prepare(a.output)
    else:
        for f,h in json.loads((a.output/'input-manifest.json').read_text()).items():assert hashlib.sha256(Path(f).read_bytes()).hexdigest()==h
        original.ARMS=ARMS;original.run(a.output,a.base)
