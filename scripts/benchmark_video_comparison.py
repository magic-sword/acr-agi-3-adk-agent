"""Equal-four-frame video comparison: grouped, alternating, and crossfade."""
import argparse
import base64
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
from PIL import Image

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts import benchmark_temporal_overlay as legacy
from scripts.benchmark_object_gate import payload,inventory_text
from scripts.benchmark_parallel import digest,read_lines
from scripts.benchmark_recognition import save_json

SOURCE=Path('outputs/overlay-ablation-20260928')
ARMS=['grouped','alternating','crossfade']
SEQUENCES={'grouped':['before','before','after','after'],
           'alternating':['before','after','before','after'],
           'crossfade':['before','mix1','mix2','after']}
LEGENDS={
 'grouped':'The four frames are BEFORE, BEFORE, AFTER, AFTER. Repeated frames repeat the same observations.',
 'alternating':'The four frames are BEFORE, AFTER, BEFORE, AFTER. This is an artificial comparison loop, NOT actual back-and-forth motion. Judge the single BEFORE-to-AFTER change.',
 'crossfade':'The four frames are BEFORE, a blend containing 1/3 AFTER, a blend containing 2/3 AFTER, AFTER. The middle frames are artificial RGB crossfades, NOT observed game states. Blended colors and double images do not prove color changes, shape changes, or additional objects.'}
PROMPT=legacy.PROMPT.replace('Compare the original BEFORE and AFTER images, using the third image as supplementary evidence.',
    'Compare the original BEFORE (first video frame) and AFTER (last video frame). The video is a synthetic presentation of just these two observations; its playback timing is not game time.').replace('Use only the original images to establish actual state.','Use only the first and last frames to establish actual state.')

def ffmpeg(args,data):
    return subprocess.run(['docker','compose','exec','-T','vlm','ffmpeg','-hide_banner','-loglevel','error',*args],input=data,capture_output=True,check=True,timeout=30).stdout

def prepare(out):
    assert not (out/'jobs.json').exists()
    cases=json.loads((SOURCE/'cases.json').read_text());cats=json.loads((SOURCE/'catalogs.json').read_text());jobs=[];clips=[]
    for c in cases:
        name=c['name']
        for fr in ['before','after']:shutil.copyfile(SOURCE/f'{name}-{fr}.png',out/f'{name}-{fr}.png')
        frames={fr:Image.open(out/f'{name}-{fr}.png').convert('RGB') for fr in ['before','after','mix1','mix2']}
        # Rendered crossfades must preserve every unchanged pixel, including the UI.
        b,f=frames['before'].tobytes(),frames['after'].tobytes()
        for mid,alpha in [('mix1',1/3),('mix2',2/3)]:
            expected=bytes(int(x*(1-alpha)+y*alpha+0.5) for x,y in zip(b,f))
            # Canvas preserves the frozen UI byte-for-byte; float rounding inside
            # blend() may differ by one at ties, but these integer thirds have no ties.
            assert frames[mid].tobytes()==expected,(name,mid)
        for arm in ARMS:
            data=b''.join(frames[k].tobytes() for k in SEQUENCES[arm])
            clip=ffmpeg(['-f','rawvideo','-pixel_format','rgb24','-video_size','640x820','-framerate','4','-i','pipe:0','-frames:v','4','-c:v','ffv1','-level','3','-f','matroska','pipe:1'],data)
            (out/f'{name}-{arm}.mkv').write_bytes(clip)
            # Match the current runtime's 4 fps filter and RGB24 decoder output.
            decoded=ffmpeg(['-i','pipe:0','-vf','fps=4','-f','rawvideo','-pix_fmt','rgb24','pipe:1'],clip)
            assert decoded==data,(name,arm,len(decoded),len(data))
            hashes=[hashlib.sha256(frames[k].tobytes()).hexdigest() for k in SEQUENCES[arm]]
            clips.append({'case':name,'arm':arm,'sequence':SEQUENCES[arm],'frames':4,'fps':4,'size':[640,820],
                'frame_sha256':hashes,'decoded_rgb_sha256':hashlib.sha256(decoded).hexdigest(),'video_sha256':hashlib.sha256(clip).hexdigest(),'lossless_decode_equal':True})
            media={'type':'input_video','input_video':{'data':'data:video/x-matroska;base64,'+base64.b64encode(clip).decode()}}
            for cat in [x for x in cats if x['case']==name]:
                prompt=PROMPT+inventory_text(cat)+'\nX: any changed or new object NOT individually listed above; use unchanged if none.\nInclude every ID above, including X, exactly once.'
                req=payload([{'type':'text','text':LEGENDS[arm]},media],prompt,cat['repeat'],512);req['response_format']={'type':'json_object'}
                jobs.append({'id':len(jobs),'case':name,'repeat':cat['repeat'],'arm':arm,'catalog_key':cat['key'],'payload':req,'payload_digest':digest(req)})
    for n,value in [('cases.json',cases),('catalogs.json',cats),('jobs.json',jobs),('video-manifest.json',clips)]:save_json(out/n,value)
    files=[Path(__file__),Path('scripts/render_video_comparison.cjs'),Path('scripts/benchmark_temporal_overlay.py'),Path('scripts/benchmark_object_gate.py'),Path('scripts/summarize_temporal_overlay.py'),Path('outputs/temporal-overlay-preview-20260928/preview.html')]
    (out/'sources').mkdir();hashes={}
    for f in files:hashes[str(f)]=hashlib.sha256(f.read_bytes()).hexdigest();shutil.copyfile(f,out/'sources'/f.name)
    for f in Path('outputs/motion-inputs-20260927/runtime-source').glob('*'):shutil.copyfile(f,out/'sources'/f.name)
    plan=json.loads((SOURCE/'plan.json').read_text());plan.update(created_at=legacy.now(),arms=ARMS,requests=len(jobs),warmups=3,source_hashes=hashes,jobs_digest=digest(jobs),
        design='11 frozen pairs x 3 frozen BEFORE inventories x 3 video arms. Each input: one lossless FFV1 video, four 640x820 frames at 4fps. Same originals, UI, inventory, main prompt and generation; arm legend differs. No object tracking, motion interpolation, annotations or dimming.',
        sampling='Decoded every clip with ffmpeg fps=4 and checked every RGB byte and all four frame hashes. Matches default sampling in pinned runtime source; internal feature tensors not directly inspected.',
        runtime_limitation='Pinned llama.cpp emits timestamp text after first frame; middle two bitmaps can temporal-merge, endpoint bitmaps cannot merge across that text. Not claimed identical to official Transformers video processing.',
        comparison_source=str(SOURCE),fairness='Equal video frame count, size, fps, question, inventory and generation across three arms. Prior static-overlay scores are historical references only, with different presentation and prompts.',
        limitations='One real pair, ten synthetic pairs. Three frozen inventories are not independent stochastic repeats. Four sampled crossfade steps, not an exhaustive test of continuous animation or animation speed. No actual additional temporal observations.')
    save_json(out/'plan.json',plan)
    manifest={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(out.iterdir()) if f.suffix in ('.png','.mkv')}
    manifest.update({str(SOURCE/n):hashlib.sha256((SOURCE/n).read_bytes()).hexdigest() for n in ['cases.json','catalogs.json','server-before.json']})
    save_json(out/'input-manifest.json',manifest)
    gallery='''<!doctype html><html lang="ja"><meta charset="utf-8"><title>4フレームの表示比較</title>
<style>body{font:16px/1.6 system-ui;background:#eef1f5;margin:24px}.row{display:flex;gap:20px;flex-wrap:wrap}figure{margin:0}img{width:320px;image-rendering:pixelated}button,select{font:inherit;padding:7px}figcaption{font-weight:bold}.status{font-variant-numeric:tabular-nums}</style>
<h1>前後のまとめ提示・交互表示・クロスフェード</h1><p>モデル入力と同じ4フレームを4fpsで表示します。実観測はBEFORE→AFTERの1回だけです。繰り返しと中間色は比較用の加工です。</p>
<select id="case"></select> <button id="play">再生</button> <button id="step">1フレーム進む</button> <span id="status"></span><div class="row" id="row"></div>
<p>白紙の挿入、境界線の追加、物体の追跡、推定移動経路の生成は行っていません。初期状態は停止しています。</p><script>
const cases=CASES, sequences=SEQUENCES, labels={grouped:'まとめ提示：A A B B',alternating:'交互表示：A B A B',crossfade:'徐々に混合：A 1/3 2/3 B'};
let frame=0,timer=null;const el=id=>document.getElementById(id);for(const c of cases){const o=document.createElement('option');o.value=c.name;o.textContent=c.title||c.name;el('case').append(o)}
for(const [arm,label] of Object.entries(labels)){const f=document.createElement('figure');f.innerHTML='<figcaption>'+label+'</figcaption><img id="'+arm+'"><p class="status" id="'+arm+'-state"></p>';el('row').append(f)}
function draw(){for(const [arm,seq] of Object.entries(sequences)){el(arm).src=el('case').value+'-'+seq[frame]+'.png';el(arm+'-state').textContent=seq[frame]}el('status').textContent='フレーム '+(frame+1)+'/4（4fps）'}
function stop(){clearInterval(timer);timer=null;el('play').textContent='再生'}
el('play').onclick=()=>{if(timer){stop();return}el('play').textContent='停止';timer=setInterval(()=>{frame=(frame+1)%4;draw()},250)};
el('step').onclick=()=>{stop();frame=(frame+1)%4;draw()};el('case').onchange=()=>{frame=0;draw()};draw();</script></html>'''
    gallery=gallery.replace('CASES',json.dumps([{'name':c['name'],'title':c.get('title',c['name'])} for c in cases],ensure_ascii=False)).replace('SEQUENCES',json.dumps(SEQUENCES))
    (out/'gallery.html').write_text(gallery)
    print('Prepared',len(jobs),'requests and',len(clips),'verified videos',flush=True)

def run(out,base):
    for f,h in json.loads((out/'input-manifest.json').read_text()).items():assert hashlib.sha256(Path(f).read_bytes()).hexdigest()==h
    legacy.ARMS=ARMS;legacy.run(out,base)

def summarize(out):
    from scripts.summarize_temporal_overlay import summarize as common
    common(out)
    s=json.loads((out/'summary.json').read_text());s['warmups']=sum(r['warmup'] for r in read_lines(out/'measurements.jsonl'))
    for arm in s['arms']:
        rows=[r for r in read_lines(out/'measurements.jsonl') if not r['warmup'] and r['arm']==arm['arm']]
        arm['prompt_tokens']=sorted(set(r.get('response',{}).get('usage',{}).get('prompt_tokens') for r in rows if r.get('response')))
    save_json(out/'summary.json',s)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','run','summarize']);p.add_argument('--output',required=True,type=Path);p.add_argument('--base',default='http://127.0.0.1:8080/v1');a=p.parse_args()
    if a.command=='prepare':prepare(a.output)
    elif a.command=='run':run(a.output,a.base)
    else:summarize(a.output)
