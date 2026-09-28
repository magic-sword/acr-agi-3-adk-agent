"""Interactive review of saved candidates; no inference or scoring changes."""
import json
from pathlib import Path
import sys


def run(out):
    cases=json.loads((out/'cases.json').read_text())
    rows=json.loads((out/'selected.json').read_text())
    data=[]
    for c in cases:
        data.append(dict(name=c['name'],gold=[[v/64*1000 for v in [g['bbox'][0],g['bbox'][1],g['bbox'][2]+1,g['bbox'][3]+1]] for g in c['gold']],
            rows=[dict(arm=r['arm'],budget=r['budget'],boxes=r['boxes'],seconds=r['seconds']) for r in rows if r['case']==c['name']]))
    page='''<!doctype html><meta charset="utf-8"><title>EdgeBoxes / SAM 比較</title>
<style>body{font:16px system-ui;margin:24px;background:#f3f4f6;color:#18202b}header{position:sticky;top:0;background:#f3f4f6;padding:12px 0}label{display:inline-block;margin:8px}main{display:flex;flex-wrap:wrap;gap:16px}article{background:white;padding:12px;border-radius:8px}canvas{width:384px;height:384px;max-width:90vw}h2{font-size:18px}p{max-width:1000px}input[type=number]{width:75px}</style>
<h1>EdgeBoxes / SAM：候補領域の比較</h1>
<p>EdgeBoxesは反復0の保存出力、SAM・Qwenは前回の保存出力です。表示数を変えて候補の重なりを確認できます。正解は別枠に表示しています。実画面の注釈は部分的です。</p>
<header><label>画像 <select id="scene"></select></label><label>EdgeBoxes生成上限 <select id="budget"><option>128</option><option>1000</option></select></label><label>上位何候補を表示 <select id="cap"><option>16</option><option>32</option><option>64</option><option selected>128</option><option>1000</option></select></label><label>指定順位だけ表示（0で全選択） <input id="rank" type="number" min="0" max="1000" value="0"></label></header><main id="cards"></main>
<script>const data=__DATA__;
const $=id=>document.getElementById(id);data.forEach((d,i)=>{const o=document.createElement('option');o.value=i;o.textContent=d.name;$('scene').append(o)});
const names={default:'EdgeBoxes 既定',small:'EdgeBoxes 小物体',small_wide:'EdgeBoxes 小物体＋細長形状',sam_standard:'SAM 標準',sam_crops:'SAM 分割探索',qwen:'Qwen 一括'};
let serial=0;
function draw(){const ticket=++serial,d=data[+$('scene').value],budget=+$('budget').value,cap=+$('cap').value,rank=+$('rank').value;$('cards').replaceChildren();
const entries=[{arm:'原画像',boxes:[]},{arm:'注釈対象',boxes:d.gold}];
for(const arm of ['default','small','small_wide','sam_standard','sam_crops','qwen']){const r=d.rows.find(r=>r.arm===arm&&r.budget===(arm.startsWith('sam')||arm==='qwen'?128:budget));entries.push({...r,arm:names[arm],pred:true})}
entries.push({arm:'Structured Edges',boxes:[],edge:true});
for(const r of entries){const article=document.createElement('article'),title=document.createElement('h2'),info=document.createElement('p'),canvas=document.createElement('canvas');title.textContent=r.arm;canvas.width=384;canvas.height=384;const boxes=r.pred?r.boxes.slice(0,cap):r.boxes;info.textContent=r.pred?`${r.boxes.length}候補から${boxes.length}表示 / ${r.seconds.toFixed(4)}秒`:'';article.append(title,info,canvas);$('cards').append(article);const im=new Image;im.onload=()=>{if(ticket!==serial)return;const ctx=canvas.getContext('2d');ctx.drawImage(im,0,0,384,384);ctx.strokeStyle=r.pred?'#ff00e5':'#00ff66';ctx.fillStyle=ctx.strokeStyle;ctx.lineWidth=1;ctx.font='10px monospace';boxes.forEach((b,i)=>{if(r.pred&&rank>0&&i+1!==rank)return;const [x,y,x2,y2]=b.map(v=>v*.384);ctx.strokeRect(x,y,x2-x,y2-y);ctx.fillText(String(i+1),x,Math.max(10,y))})};im.src=r.edge?`edges/${d.name}.png`:`${d.name}.png`}
}
for(const id of ['scene','budget','cap','rank'])$(id).addEventListener('change',draw);draw();</script>'''
    (out/'interactive.html').write_text(page.replace('__DATA__',json.dumps(data,ensure_ascii=False).replace('<','\\u003c')))


if __name__=='__main__':run(Path(sys.argv[1] if len(sys.argv)>1 else 'outputs/edgeboxes-20260928'))
