"""Compare deterministic visual emphasis on fixed raw game pixels.

Original pair plus two equal-sized auxiliary views in every arm. Annotation
coordinates and scoring references never enter the textual model context.
"""
import argparse
from collections import deque
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path
import random
import statistics
import sys
import time

from PIL import Image, ImageDraw, ImageFilter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.preview_change_emphasis import masks, displayed_color, rgb
from scripts.benchmark_parallel import digest, read_lines
from scripts.benchmark_recognition import encode, save_json, validate
from scripts.benchmark_attention_selection import http
from scripts.fonts import ui_font

ARMS = ('original', 'white', 'outline', 'blur', 'numbered')
TARGETS = ('orange_blue', 'white_cross', 'yellow_strip')
SCALE = 6
ORIGIN = (32, 44)
SIZE = (428, 452)
INSTRUCTION = '''Inspect the BEFORE and AFTER images. Four images show ONE pair:
the first two are original observations and the last two are auxiliary renderings
of those same observations. Auxiliary background may be softened or regions may
have outline/number annotations. Those marks are not game objects or new events.
Original images are authoritative for colors, shapes, and static reference parts.
For each requested appearance (orange-blue group, white cross, yellow strip),
locate it before and after and classify the observed change. Names specify visual
appearance only; they do not establish player, exit, path, health, or any function.
Use null boxes and uncertain if you cannot locate a requested appearance.
Use inclusive original-grid boxes [left,top,right,bottom] within 64x64.
Positive dx is right and positive dy is down. Translation must agree with both
boxes. For unchanged use dx=dy=0; for shape change or uncertain use null dx/dy.
Pixel delta is the change in the target's colored pixel count, or null if unknown.
Describe evidence concisely; do not infer roles from familiar shapes. State whether
these observations establish a functional player/exit/health role. Return one
submit_visual_comparison tool call. Do not infer intermediate animation frames.'''


def now():
    return datetime.now(timezone.utc).isoformat()


def mask_boxes(mask):
    remaining = set(mask); result = []
    while remaining:
        first = min(remaining, key=lambda p: (p[1], p[0])); remaining.remove(first)
        group = {first}; queue = deque([first])
        while queue:
            x, y = queue.popleft()
            for p in ((x-1,y), (x+1,y), (x,y-1), (x,y+1)):
                if p in remaining:
                    remaining.remove(p); group.add(p); queue.append(p)
        result.append([min(x for x,y in group), min(y for x,y in group),
                       max(x for x,y in group), max(y for x,y in group)])
    return result


def board_image(frame, palette, arm, mask):
    board = Image.new('RGB', (64, 64))
    board.putdata([rgb(palette[str(c)]) for row in frame for c in row])
    original = board.resize((384, 384), getattr(Image, 'Resampling', Image).NEAREST)
    if arm == 'white':
        board.putdata([displayed_color(palette[str(c)], (x,y) in mask, .2)
                       for y,row in enumerate(frame) for x,c in enumerate(row)])
        return board.resize((384, 384), getattr(Image, 'Resampling', Image).NEAREST)
    if arm == 'blur':
        # Radius 6 display pixels = one original cell. Keep protected pixels exact.
        result = original.filter(ImageFilter.GaussianBlur(radius=6))
        for x,y in mask:
            box = (x*SCALE,y*SCALE,(x+1)*SCALE,(y+1)*SCALE)
            result.paste(original.crop(box), box[:2])
        return result
    return original


def render(frame, palette, arm, mask, title):
    image = Image.new('RGB', SIZE, '#18202b'); ox, oy = ORIGIN
    image.paste(board_image(frame, palette, arm, mask), ORIGIN)
    draw = ImageDraw.Draw(image); draw.text((12,10), title, fill='white')
    for x in (0,8,16,24,32,40,48,56,63): draw.text((ox+x*SCALE,oy-16), str(x), fill='white')
    for y in (0,8,16,24,32,40,48,56,63): draw.text((3,oy+y*SCALE), str(y), fill='white')
    if arm in ('outline', 'numbered'):
        font = ui_font(13)
        for index, (l,t,r,b) in enumerate(mask_boxes(mask), 1):
            # Outer two display pixels, separated from preserved pixels by a gap.
            box = [ox+l*SCALE-3, oy+t*SCALE-3, ox+(r+1)*SCALE+2, oy+(b+1)*SCALE+2]
            draw.rectangle(box, outline='#ff0055', width=2)
            if arm == 'numbered':
                x = min(box[2]+3, SIZE[0]-22); y = max(oy, box[1]-19)
                draw.rectangle([x,y,x+17,y+17], fill='white', outline='#ff0055')
                draw.text((x+4,y), str(index), fill='black', font=font)
    return image


def schema():
    box = {'type': ['array', 'null'], 'minItems': 4, 'maxItems': 4,
           'items': {'type': 'integer', 'minimum': 0, 'maximum': 63}}
    part = {'target': {'type': 'string', 'enum': list(TARGETS)},
            'before_box': box, 'after_box': box,
            'change': {'type': 'string', 'enum': ['translation','unchanged','color_or_shape_change','uncertain']},
            'dx': {'type': ['integer','null']}, 'dy': {'type': ['integer','null']},
            'pixel_delta': {'type': ['integer','null']},
            'evidence': {'type': 'string', 'maxLength': 140}}
    fields = {'parts': {'type': 'array', 'minItems': 3, 'maxItems': 3,
                       'items': {'type': 'object', 'additionalProperties': False, 'properties': part, 'required': list(part)}},
              'roles_established': {'type': 'boolean'}, 'summary': {'type': 'string', 'maxLength': 200}}
    return {'type': 'object', 'additionalProperties': False, 'properties': fields, 'required': list(fields)}


def payload(images):
    content = []
    for title, img in zip(('ORIGINAL BEFORE','ORIGINAL AFTER','AUXILIARY BEFORE','AUXILIARY AFTER'), images):
        content += [{'type':'text','text':title}, encode(img)]
    content.append({'type':'text','text':'Locate and compare the orange-blue group, white cross, and yellow strip. Describe only evidence from this pair.'})
    return {'model':'qwen3-vl-4b-instruct','messages':[{'role':'system','content':INSTRUCTION}, {'role':'user','content':content}],
            'temperature':0,'seed':123,'max_tokens':750,'stream':False,'cache_prompt':True,'id_slot':0,
            'tools':[{'type':'function','function':{'name':'submit_visual_comparison','description':'Submit visual correspondence and changes.', 'parameters':schema()}}],
            'tool_choice':'required','parallel_tool_calls':False}


def parse(response):
    choice = response['choices'][0]
    if choice.get('finish_reason') == 'length': raise ValueError('truncated response')
    calls = choice['message'].get('tool_calls', [])
    if len(calls) != 1 or calls[0]['function']['name'] != 'submit_visual_comparison': raise ValueError('wrong tool call')
    v = calls[0]['function']['arguments']; v = json.loads(v) if isinstance(v,str) else v
    validate(v, schema())
    if type(v['roles_established']) is not bool: raise ValueError('expected boolean role status')
    if {p['target'] for p in v['parts']} != set(TARGETS): raise ValueError('missing or duplicate target')
    for part in v['parts']:
        for name in ('before_box','after_box'):
            b = part[name]
            if b is not None and (not isinstance(b,list) or len(b)!=4 or any(type(x) is not int or not 0<=x<=63 for x in b)
                                  or b[0]>b[2] or b[1]>b[3]): raise ValueError('invalid box')
        for name in ('dx','dy','pixel_delta'):
            if part[name] is not None and type(part[name]) is not int: raise ValueError('invalid numeric field')
    return v


def iou(a,b):
    if a is None or b is None: return 0
    inter = max(0,min(a[2],b[2])-max(a[0],b[0])+1)*max(0,min(a[3],b[3])-max(a[1],b[1])+1)
    area = lambda x:(x[2]-x[0]+1)*(x[3]-x[1]+1)
    return inter/(area(a)+area(b)-inter)


def references(name):
    before = {'orange_blue':[34,45,38,49], 'white_cross':[20,31,22,33], 'yellow_strip':[13,61,54,62]}
    after = {**before, 'orange_blue':[34,40,38,44], 'yellow_strip':[14,61,54,62]}
    if name == 'reverse': before,after = after,before
    if name == 'same_frame': after = deepcopy(before)
    return before,after


def score(v, name):
    parts = {p['target']:p for p in v.get('parts',[])}; before,after = references(name)
    result = {'motion':False,'cross':False,'bar_shape':False,'bar_pixels':False,
              'all_three':False,'field_conflicts':0,'no_change':False}
    for target in TARGETS:
        p = parts.get(target)
        if p is None: continue
        threshold = .4 if target == 'white_cross' else .6
        located = iou(p['before_box'],before[target])>=threshold and iou(p['after_box'],after[target])>=threshold
        result[target+'_located'] = located
        dx,dy = p['dx'],p['dy']; b,a = p['before_box'],p['after_box']
        shift_consistent = b is not None and a is not None and dx is not None and dy is not None and a == [c+(dx if j%2==0 else dy) for j,c in enumerate(b)]
        if p['change'] in ('translation','unchanged'):
            if not shift_consistent or (p['change']=='unchanged' and (dx,dy)!=(0,0)): result['field_conflicts']+=1
        elif dx is not None or dy is not None: result['field_conflicts']+=1
        unchanged = located and p['change']=='unchanged' and (dx,dy)==(0,0) and b==a
        if target == 'white_cross': result['cross'] = unchanged
        if target == 'orange_blue':
            result['motion'] = unchanged if name=='same_frame' else located and shift_consistent and p['change']=='translation' and (dx,dy)==(0,5 if name=='reverse' else -5)
        if target == 'yellow_strip':
            expected = 0 if name=='same_frame' else 2 if name=='reverse' else -2
            # Full bar bounds, correct growth/shrink sign, null translation fields.
            if name=='same_frame': result['bar_shape'] = unchanged
            elif located and b is not None and a is not None:
                area = lambda box:(box[2]-box[0]+1)*(box[3]-box[1]+1)
                result['bar_shape'] = p['change']=='color_or_shape_change' and dx is None and dy is None and (area(a)-area(b))*expected>0
            result['bar_pixels'] = bool(result['bar_shape'] and p['pixel_delta']==expected)
    result['no_change'] = len(parts)==3 and all(p['change']=='unchanged' and p['dx']==0 and p['dy']==0
                                                and p['before_box'] is not None and p['before_box']==p['after_box'] for p in parts.values())
    result['all_three'] = bool(result['motion'] and result['cross'] and result['bar_pixels'])
    result['roles_established'] = v.get('roles_established')
    return result


def views(case, arm):
    before,after,palette = case['before'],case['after'],case['palette']
    _, mask = masks(before,after)
    images = [render(before,palette,'original',mask,'ORIGINAL BEFORE'), render(after,palette,'original',mask,'ORIGINAL AFTER'),
              render(before,palette,arm,mask,'AUXILIARY BEFORE'), render(after,palette,arm,mask,'AUXILIARY AFTER')]
    return images,mask


def prepare(source,output):
    output.mkdir(parents=True,exist_ok=False)
    f=json.loads((source/'frames.json').read_text()); old=json.loads((source/'cases.json').read_text())[0]
    palette=json.loads(old['payload']['messages'][1]['content'][-1]['text'])['palette']
    cases=[]; planned=[]
    for name in ('forward','reverse','same_frame'):
        b,a=f['before'],f['after']
        if name=='reverse':b,a=a,b
        if name=='same_frame':a=b
        case={'name':name,'before':b,'after':a,'palette':palette};cases.append(case)
        for arm in ARMS:
            images,mask=views(case,arm); p=payload(images)
            planned.append({'name':name,'arm':arm,'payload_digest':digest(p)})
            for j,img in enumerate(images):img.save(output/f'{name}-{arm}-{j}.png')
    save_json(output/'cases.json',cases);save_json(output/'planned-requests.json',planned)
    save_json(output/'experiment.json',{'created_at':now(),'case_digest':digest(cases),'arms':ARMS,
        'source':str(source),'source_frames_digest':digest(f),'warmups_per_input':1,'repeats':3,
        'wall_budget_seconds':20,'max_tokens':750,'images_per_request':4,'image_size':SIZE,
        'white_opacity':.2,'blur_radius_display_pixels':6,'outline_width_display_pixels':2,'outline_color':'#ff0055',
        'mask':'same union of changed small components (<=256 cells) before and after; no semantic roles',
        'context':'appearance queries only; no action name, prior roles, text pixel differences, coordinates or displacements',
        'scope':'One real transition plus reversed-pair and same-frame diagnostics. Reverse is not a real action. Appearance-guided localization, not autonomous object discovery.',
        'order':'random order per repetition, seed 10300+rep; warm slot, mixed prefix caching'})
    save_json(output/'rubric.json',{'created_at':now(),'references':{n:references(n) for n in ('forward','reverse','same_frame')},
        'localization':'IoU >=0.6 in both frames; white cross >=0.4 (partial localization permitted)',
        'motion':'correct reference localization, matching before/after boxes and displacement (forward dy=-5, reverse dy=+5)',
        'cross':'localized and unchanged with zero displacement and identical boxes',
        'bar_shape':'localized, shape change, null dx/dy, correctly signed bounding-box area change',
        'bar_pixels':'bar_shape plus exact pixel_delta (-2 forward, +2 reverse, 0 same)',
        'strict':'valid within budget, all three including pixel_delta, no field conflicts, roles_established false',
        'semantic':'free text coherence and appearances reviewed separately by assistant after scoring'})
    for path in (Path(__file__),Path(__file__).with_name('preview_change_emphasis.py'),Path(__file__).with_name('motion_evidence.py')):
        (output/path.name).write_bytes(path.read_bytes())
    page=['<!doctype html><html lang="ja"><meta charset="utf-8"><title>強調表示の5条件</title><style>body{font:16px system-ui;margin:24px}img{max-width:48%;image-rendering:pixelated}</style><h1>補助表示の5条件</h1><p>全条件に原画像の前後2枚を別途渡す。以下は補助画像。</p>']
    for arm in ARMS:
        page.append(f'<h2>{arm}</h2><img src="forward-{arm}-2.png"><img src="forward-{arm}-3.png">')
    (output/'input-gallery.html').write_text('\n'.join(page)+'</html>')
    print('Prepared',len(planned),'inputs',flush=True)


def run(output,base):
    assert not (output/'measurements.jsonl').exists(),'use a new output directory'
    cases=json.loads((output/'cases.json').read_text()); planned=json.loads((output/'planned-requests.json').read_text())
    checks={(p['name'],p['arm']):p['payload_digest'] for p in planned}
    props=http(base.removesuffix('/v1')+'/props');assert props['total_slots']==1
    paths=[Path(__file__),Path(__file__).with_name('preview_change_emphasis.py'),Path(__file__).with_name('motion_evidence.py')]
    save_json(output/'environment.json',{'server':props,'case_digest':digest(cases),
        'source_hashes':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}})
    with (output/'measurements.jsonl').open('w') as log:
        for rep in (-1,0,1,2):
            jobs=[(case,arm) for case in cases for arm in ARMS];random.Random(10300+rep).shuffle(jobs)
            for case,arm in jobs:
                row={'name':case['name'],'arm':arm,'repeat':rep,'warmup':rep==-1,'started_at':now()}
                start=time.monotonic();images,mask=views(case,arm);p=payload(images)
                row.update(preprocess_seconds=time.monotonic()-start,mask_pixels=len(mask),payload_digest=digest(p),request=p)
                assert row['payload_digest']==checks[(case['name'],arm)]
                request_start=time.monotonic()
                try:
                    remaining=20-(request_start-start)
                    if remaining<=0:raise TimeoutError('rendering exhausted shared budget')
                    response=http(base+'/chat/completions',p,remaining);row['response']=response
                    row['parsed']=parse(response);row['valid']=True
                except Exception as exc:row.update(valid=False,error=f'{type(exc).__name__}: {exc}')
                row['model_seconds']=time.monotonic()-request_start;row['seconds']=time.monotonic()-start
                row['within_budget']=row['seconds']<=20;row['score']=score(row.get('parsed',{}),case['name'])
                s=row['score'];row['strict_success']=bool(row['valid'] and row['within_budget'] and s['all_three'] and s['field_conflicts']==0 and s['roles_established'] is False)
                log.write(json.dumps(row,ensure_ascii=False)+'\n');log.flush()
                print(rep,case['name'],arm,round(row['seconds'],2),row['valid'],s,row.get('error',''),flush=True)
    save_json(output/'server-after.json',http(base.removesuffix('/v1')+'/props'));report(output)


def report(output):
    rows=[r for r in read_lines(output/'measurements.jsonl') if not r['warmup']];summary=[];unique={}
    for name in ('forward','reverse','same_frame'):
        for arm in ARMS:
            rs=[r for r in rows if (r['name'],r['arm'])==(name,arm)]
            if not rs:continue
            summary.append({'case':name,'arm':arm,'n':len(rs),'valid':sum(r['valid'] for r in rs),
                **{k:sum(r['score'].get(k,0) for r in rs) for k in ('motion','cross','bar_shape','bar_pixels','field_conflicts','no_change')},
                'roles_established':sum(r['score'].get('roles_established') is True for r in rs),
                'strict_success':sum(r['strict_success'] for r in rs),'seconds_median':statistics.median(r['seconds'] for r in rs),
                'preprocess_ms_median':1000*statistics.median(r['preprocess_seconds'] for r in rs),
                'prompt_tokens_median':statistics.median(r.get('response',{}).get('usage',{}).get('prompt_tokens',0) for r in rs),
                'output_tokens_median':statistics.median(r.get('response',{}).get('usage',{}).get('completion_tokens',0) for r in rs)})
    for r in rows:
        v=r.get('parsed',{'error':r.get('error')});key=digest(v)
        unique.setdefault(key,{'id':key,'output':v,'members':[]})['members'].append({k:r[k] for k in ('name','arm','repeat','valid','score')})
    save_json(output/'summary.json',summary);save_json(output/'review-candidates.json',list(unique.values()))
    lines=['# 強調表示の比較','', '座標・数値の事前基準による採点。文章の意味は別途確認。逆順は合成診断。', '',
           '|ケース|条件|n|形式|移動|十字|帯の伸縮|帯の画素数|矛盾|役割確定|厳格成功|秒|',
           '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for s in summary:
        lines.append('|'+ '|'.join(str(s[k]) for k in ('case','arm','n','valid','motion','cross','bar_shape','bar_pixels','field_conflicts','roles_established','strict_success'))+f'|{s["seconds_median"]:.2f}|')
    (output/'report.md').write_text('\n'.join(lines)+'\n')
    page=['<!doctype html><html lang="ja"><meta charset="utf-8"><title>強調表示比較・回答</title><style>body{font:16px system-ui;margin:24px}pre{white-space:pre-wrap;background:#f3f4f6;padding:12px}article{border:1px solid #aaa;padding:12px;margin:12px}</style><h1>回答比較</h1><a href="input-gallery.html">入力画像</a>']
    for r in unique.values():page.append('<article><pre>'+html.escape(json.dumps(r,ensure_ascii=False,indent=2))+'</pre></article>')
    (output/'comparison.html').write_text('\n'.join(page)+'</html>')
    print('Summarized',len(rows),'measurements;',len(unique),'unique responses',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['prepare','run','report'])
    p.add_argument('--source',type=Path,default=Path('outputs/motion-comparison-20260927'))
    p.add_argument('--output',type=Path,required=True);p.add_argument('--base',default='http://127.0.0.1:8080/v1')
    a=p.parse_args()
    if a.command=='prepare':prepare(a.source,a.output)
    elif a.command=='run':run(a.output,a.base)
    else:report(a.output)


if __name__=='__main__':main()
