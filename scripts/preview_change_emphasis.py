"""Render raw pixel data as SVG/HTML diagrams of temporal change emphasis.

No generated pixels, game actions, or model calls. These are derived evidence
views: unchanged pixels are composited onto white, not new observations.
"""
import argparse
import html
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.motion_evidence import components
from scripts.benchmark_parallel import digest
from scripts.benchmark_recognition import save_json


def masks(before, after, max_component_pixels=256):
    if not before or not before[0] or len(before) != len(after):
        raise ValueError('incompatible grids')
    width = len(before[0])
    if any(len(row) != width for grid in (before, after) for row in grid):
        raise ValueError('incompatible grids')
    changed = {(x, y) for y, row in enumerate(before) for x, color in enumerate(row)
               if color != after[y][x]}
    context = set(changed)
    # Both views use the SAME mask in original coordinates. Include the support
    # of small changed components from either time, preserving the extent of a
    # shrinking part. Large backgrounds do not flood the mask.
    for grid in (before, after):
        for c in components(grid):
            if c['area'] <= max_component_pixels and c['pixels'] & changed:
                context.update(c['pixels'])
    return changed, context


def rgb(hex_color):
    value = hex_color.lstrip('#')
    return tuple(int(value[i:i+2], 16) for i in (0, 2, 4))


def displayed_color(color, retained, opacity):
    if not 0 <= opacity <= 1:
        raise ValueError('opacity must be between zero and one')
    source = rgb(color)
    return source if retained else tuple(int(c*opacity+255*(1-opacity)+.5) for c in source)


def panel(frame, palette, mask, opacity, title, origin_x=0, origin_y=0):
    scale = 6; ox = origin_x+32; oy = origin_y+44
    parts = [f'<text x="{origin_x+12}" y="{origin_y+18}" fill="white">{html.escape(title)}</text>']
    for y, row in enumerate(frame):
        colors = [displayed_color(palette[str(c)], mask is None or (x, y) in mask, opacity)
                  for x, c in enumerate(row)]
        x = 0
        while x < len(row):
            end = x+1
            while end < len(row) and colors[end] == colors[x]:
                end += 1
            color = '#%02x%02x%02x' % colors[x]
            parts.append(f'<rect x="{ox+x*scale}" y="{oy+y*scale}" width="{(end-x)*scale}" height="{scale}" fill="{color}"/>')
            x = end
    for x in sorted(set(range(0, len(frame[0]), 8)) | {len(frame[0])-1}):
        parts.append(f'<text x="{ox+x*scale}" y="{oy-8}" fill="white">{x}</text>')
    for y in sorted(set(range(0, len(frame), 8)) | {len(frame)-1}):
        parts.append(f'<text x="{origin_x+3}" y="{oy+y*scale+8}" fill="white">{y}</text>')
    return '\n'.join(parts)


def svg(body, width=856, height=452):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
            'style="font:12px sans-serif;shape-rendering:crispEdges">'
            f'<rect width="{width}" height="{height}" fill="#18202b"/>{body}</svg>')


def prepare(source, output, opacity):
    output.mkdir(parents=True, exist_ok=False)
    frames = json.loads((source/'frames.json').read_text())
    case = next(c for c in json.loads((source/'cases.json').read_text()) if c['name'] == 'up')
    palette = json.loads(case['payload']['messages'][1]['content'][-1]['text'])['palette']
    before, after = frames['before'], frames['after']
    if (len(before), len(before[0])) != (64, 64):
        raise ValueError('preview layout expects 64x64 boards')
    changed, context = masks(before, after)
    config = {'frames': frames, 'palette': palette, 'opacity': opacity,
              'changed': sorted(changed), 'context': sorted(context)}
    save_json(output/'evidence.json', config)
    body = []
    variants = [('original', None, 'ORIGINAL'), ('pixels', changed, 'CHANGED PIXELS'),
                ('components', context, 'CHANGED SMALL COMPONENTS')]
    for i, (name, mask, label) in enumerate(variants):
        pair = panel(before, palette, mask, opacity, f'{label}: BEFORE')
        pair += panel(after, palette, mask, opacity, f'{label}: AFTER', 428)
        (output/f'{name}.svg').write_text(svg(pair))
        body.append(panel(before, palette, mask, opacity, f'{label}: BEFORE', 0, 452*i))
        body.append(panel(after, palette, mask, opacity, f'{label}: AFTER', 428, 452*i))
    (output/'comparison.svg').write_text(svg(''.join(body), height=452*3))
    template = '''<!doctype html><html lang="ja"><meta charset="utf-8">
<title>差分強調の見本</title><style>
body{font:16px system-ui;max-width:1000px;margin:24px auto;padding:0 16px;color:#253044;background:#fafafa}
section{margin:28px 0}svg{width:100%;height:auto;display:block}label{display:block;padding:12px;background:#e8eef5}
.note{max-width:850px;line-height:1.7}input{vertical-align:middle;width:250px}
</style><h1>差分強調の見本</h1>
<p class="note">ls20の実際のUP操作前後。同じ画素データから描画した補助表示で、新しい観測ではありません。
左が操作前、右が操作後です。前後で同じ強調マスク・座標・縮尺を使います。</p>
<label>不変部分に残す元の色：<input id="alpha" type="range" min="0" max="1" step="0.05"><output id="amount"></output></label>
<section><h2>1. 元の画像</h2><div id="original"></div></section>
<section><h2>2. 変わった画素だけを原色で残す</h2>
<p class="note">変化した52画素の位置を強調。不変部分は白と混合します。移動前の位置と移動後の位置が見えますが、黄色い帯では減った左端2画素だけが強調されます。</p><div id="pixels"></div></section>
<section><h2>3. 変化が触れた小さな部品の範囲まで残す</h2>
<p class="note">変化画素に接する256画素以下の単色連結領域を、前後両方から加えたマスクです。
黄色い帯の全体形も残します。部品の役割や、一つの物体であることを確定する処理ではありません。</p><div id="components"></div></section>
<p class="note">白い十字など不変の参照対象は薄くなるため、元画像も併用します。
操作対象・出口・体力などの役割は、この表示だけでは確定できません。モデルの認識精度への効果は未計測です。</p>
<script>const data = DATA;
const alpha=document.querySelector('#alpha');alpha.value=data.opacity;
function panel(frame,mask,a,title,ox){let out=`<text x="${ox+12}" y="18" fill="white">${title}</text>`;
 for(let y=0;y<64;y++)for(let x=0;x<64;x++){let c=data.palette[frame[y][x]].slice(1).match(/../g).map(v=>parseInt(v,16));
  if(mask&&!mask.has(`${x},${y}`))c=c.map(v=>Math.round(v*a+255*(1-a)));
  out+=`<rect x="${ox+32+x*6}" y="${44+y*6}" width="6" height="6" fill="rgb(${c.join(',')})"/>`;}
 for(const x of [0,8,16,24,32,40,48,56,63])out+=`<text x="${ox+32+x*6}" y="36" fill="white">${x}</text>`;
 for(const y of [0,8,16,24,32,40,48,56,63])out+=`<text x="${ox+3}" y="${52+y*6}" fill="white">${y}</text>`;return out;}
function draw(){const a=Number(alpha.value);document.querySelector('#amount').textContent=Math.round(a*100)+'%';
 for(const [id,coords] of [['original',null],['pixels',data.changed],['components',data.context]]){
 const mask=coords?new Set(coords.map(p=>p.join(','))):null;
 document.getElementById(id).innerHTML='<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 856 452" style="font:12px sans-serif;shape-rendering:crispEdges"><rect width="856" height="452" fill="#18202b"/>'+panel(data.frames.before,mask,a,'BEFORE',0)+panel(data.frames.after,mask,a,'AFTER',428)+'</svg>';}}
alpha.addEventListener('input',draw);draw();</script></html>'''
    (output/'preview.html').write_text(template.replace('DATA', json.dumps(config, separators=(',', ':'))))
    checks = {'source': str(source), 'source_frames_digest': digest(frames), 'changed_pixels': len(changed),
              'context_mask_pixels': len(context), 'unchanged_opacity': opacity,
              'formula': 'changed/masked: original RGB; otherwise round(alpha*RGB+(1-alpha)*255)',
              'same_mask_for_before_after': True, 'background': 'opaque white compositing, not transparent file pixels',
              'model_evaluation': 'not performed', 'method': 'direct raw-grid SVG rendering; no image synthesis'}
    save_json(output/'manifest.json', checks)
    print(json.dumps(checks, ensure_ascii=False, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, default=Path('outputs/motion-comparison-20260927'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--opacity', type=float, default=.2)
    a = p.parse_args()
    if not 0 <= a.opacity <= 1: p.error('--opacity must be between zero and one')
    prepare(a.source, a.output, a.opacity)


if __name__ == '__main__':
    main()
