"""Frame-by-frame review of preserved proposal tracks and partial gold."""
import json
from pathlib import Path
import sys


def run(out):
    seqs=json.loads((out/'sequences.json').read_text())
    rows=[json.loads(x) for x in (out/'results.jsonl').read_text().splitlines() if json.loads(x)['repeat']==0]
    scores=json.loads((out/'frame-scores.json').read_text())
    data=[]
    for s in seqs:
        frames=[]
        for f in s['frames']:
            frames.append(dict(index=f['index'],gold=f['gold'],
                states=[r for r in rows if r['sequence']==s['name'] and r['frame']==f['index']],
                scores=[r for r in scores if r['sequence']==s['name'] and r['frame']==f['index']]))
        data.append(dict(name=s['name'],kind=s['kind'],frames=frames))
    page='''<!doctype html><meta charset="utf-8"><title>Temporal proposal comparison</title>
<style>body{font:16px system-ui;margin:24px;background:#f4f5f7}header{position:sticky;top:0;background:#f4f5f7;padding:12px}main{display:flex;flex-wrap:wrap;gap:12px}article{background:white;padding:12px}h2{font-size:18px}canvas{width:320px;height:320px}p{max-width:1050px}label{margin:12px}pre{max-width:320px;white-space:pre-wrap}</style>
<h1>候補領域の維持：プログラムとSAM</h1><p>全方式で同じ画素追跡を使用。枠の番号は方式ごとの追跡IDです。実画面は部分注釈。時間はCPU実測中央値＋共有するSAM実測時間で、モデル読込みは除外しています。</p>
<header><label>系列 <select id="seq"></select></label><button id="prev">前</button><label>フレーム <input id="frame" type="range" min="0" value="0"><b id="number"></b></label><button id="next">次</button><label><input id="boxes" type="checkbox" checked>候補枠を表示</label></header><main id="cards"></main>
<script>const data=__DATA__;const $=id=>document.getElementById(id);const names={program:'プログラムのみ',sam_initial:'初回SAM＋画素更新',sam_adaptive:'必要時SAM＋画素更新',sam_every:'毎手SAM＋画素更新'};data.forEach((s,i)=>{const o=document.createElement('option');o.value=i;o.textContent=s.name+' / '+s.kind;$('seq').append(o)});let version=0;
function draw(){const ticket=++version,s=data[+$('seq').value];$('frame').max=s.frames.length-1;const i=Math.min(+$('frame').value,s.frames.length-1);$('frame').value=i;$('number').textContent=i+' / '+(s.frames.length-1);const f=s.frames[i];$('cards').replaceChildren();const entries=[{name:'注釈対象',tracks:f.gold.map(g=>({id:g.id,box:[g.bbox[0],g.bbox[1],g.bbox[2]+1,g.bbox[3]+1]})),gold:true}];for(const arm of Object.keys(names)){const r=f.states.find(r=>r.arm===arm),m=f.scores.find(r=>r.arm===arm);entries.push({...r,name:names[arm],metric:m})}for(const r of entries){const card=document.createElement('article'),h=document.createElement('h2'),p=document.createElement('pre'),c=document.createElement('canvas');h.textContent=r.name;c.width=c.height=384;if(r.metric)p.textContent=`回収 ${r.metric.main.tp}/${r.metric.main.gold} / 候補 ${r.tracks.length}\nSAM呼出し ${r.sam_called?'あり':'なし'} / ${(r.metric.cpu_seconds+r.sam_seconds).toFixed(3)}秒\n追跡消失 ${r.events.lost} / 対応保留 ${r.events.ambiguous}`;card.append(h,p,c);$('cards').append(card);const im=new Image;im.onload=()=>{if(ticket!==version)return;const ctx=c.getContext('2d');ctx.drawImage(im,0,0,384,384);if(!$('boxes').checked)return;ctx.lineWidth=1;ctx.font='10px monospace';for(const t of r.tracks){const [x,y,x2,y2]=t.box.map(v=>v*6);ctx.strokeStyle=ctx.fillStyle=r.gold?'#00ff55':`hsl(${(t.id*137)%360},100%,65%)`;ctx.strokeRect(x,y,x2-x,y2-y);ctx.fillText(String(t.id),x,Math.max(10,y))}};im.src=`images/${s.name}-${i}.png`}}
$('seq').onchange=()=>{$('frame').value=0;draw()};$('frame').oninput=draw;$('boxes').onchange=draw;$('prev').onclick=()=>{$('frame').value=Math.max(0,+$('frame').value-1);draw()};$('next').onclick=()=>{$('frame').value=Math.min(+$('frame').max,+$('frame').value+1);draw()};draw();</script>'''
    (out/'gallery.html').write_text(page.replace('__DATA__',json.dumps(data,ensure_ascii=False).replace('<','\\u003c')))


if __name__=='__main__':run(Path(sys.argv[1] if len(sys.argv)>1 else 'outputs/temporal-proposals-20260928'))
