"""Assemble reviewed outcomes, source observations and a self-contained gallery."""
import json,statistics,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.benchmark_parallel import read_lines
from scripts.benchmark_recognition import save_json

out=Path(sys.argv[1]);load=lambda n:json.loads((out/n).read_text());rows=[r for r in read_lines(out/'measurements.jsonl') if not r['warmup']]
assert len(rows)==60
cases=load('cases.json');states=load('states.json');program=load('program-results.json');reviews=load('semantic-review.json');bykey={(r['case'],r['arm']):r for r in reviews['rows']}
observations={(r['case'],r['frame']):r for r in rows if r['stage']=='observe'}
judges=[r for r in rows if r['stage']=='judge'];arms=[]
for arm in load('plan.json')['arms']:
    rs=[r for r in reviews['rows'] if r['arm']==arm];assert len(rs)==10
    valid=sum(r['pipeline_valid'] for r in rs);primary=sum(r['primary'] for r in rs if r['case']!='ambiguous_copies')
    times=[]
    if arm!='deterministic':
        for r in judges:
            if r['arm']!=arm:continue
            t=r['seconds']
            if arm=='independent_text':t+=sum(observations[(r['case'],fr)]['seconds'] for fr in ['before','after'])
            times.append(t)
    arms.append(dict(arm=arm,requests=10,pipeline_valid=valid,moving_targets_correct=sum(r['moving_targets_correct'] for r in rs),moving_targets=5,primary_correct=primary,primary_cases=9,
        strict_correct=sum(r['strict'] for r in rs if r['case']!='ambiguous_copies'),false_change_answers=sum(r['false_change'] for r in rs),panel_class_preserved=sum(r.get('panel_class_preserved',False) for r in rs if r['case'].startswith('four_panels')),panel_pattern_match=sum(r.get('panel_pattern_match',False) for r in rs if r['case']=='four_panels_change'),ambiguity_acknowledged=next(r.get('ambiguity_acknowledged',False) for r in rs if r['case']=='ambiguous_copies'),median_pipeline_seconds=statistics.median(times) if times else None))
summary=dict(arms=arms,requests=60,warmups=2,independent_observation_valid=sum(r['valid'] for r in observations.values()),independent_observations=20,observation_review=reviews.get('observations',[]),rows=reviews['rows'])
save_json(out/'summary.json',summary)
data=dict(summary=summary,cases=[dict(name=c['name'],title=c['title']) for c in cases],states=states,observations={c['name']:{fr:{k:observations[(c['name'],fr)].get(k) for k in ['parsed','answer','valid','seconds','error']} for fr in ['before','after']} for c in cases},judges=[{k:r.get(k) for k in ['case','arm','parsed','answer','valid','seconds']}|dict(review=bykey[(r['case'],r['arm'])]) for r in judges],program={r['case']:r['result'] for r in program},reviews=reviews)
save_json(out/'gallery-data.json',data)
html=r'''<!doctype html><html lang="ja"><meta charset="utf-8"><title>概念・個体・状態を分ける比較</title>
<style>body{font:16px/1.6 system-ui;background:#edf2f6;color:#18232f;margin:24px}h1{font-size:26px}table{border-collapse:collapse;width:100%;background:white;margin:12px 0}td,th{border:1px solid #c9d3df;padding:7px;text-align:left}th{background:#dce5ee}select{font:inherit;padding:7px;margin:6px}main,.grid{display:grid;grid-template-columns:repeat(2,minmax(340px,1fr));gap:18px}figure{margin:0}img{width:355px;height:355px;image-rendering:pixelated}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:13px/1.5 monospace}.card{background:white;padding:18px;border-radius:8px}details{margin:8px 0}summary{cursor:pointer}.ok{color:#116140}.bad{color:#a12525}.small{font-size:14px}.pattern{display:inline-grid;gap:2px;vertical-align:middle;margin:6px}.cell{width:16px;height:16px;border:1px solid #aaa}@media(max-width:800px){main,.grid{grid-template-columns:1fr}}</style>
<h1>概念・個体・各時点の状態を分ける比較</h1>
<p>通常量子化版Qwen。元の64×64画素から座標付き画像を作成。各時点は独立に観測し、プログラム測定にも前後対応や変化の正解を含めていません。実測側のまとまりは幾何学的な候補です。4パネルは添付ゲームの再現ではなく構造を似せた合成例です。</p>
<div id="metrics"></div><label>例<select id="case"></select></label><label>詳細の取得元<select id="source"><option value="measured">プログラムの実測</option><option value="vlm">Qwenの独立観測</option></select></label>
<main><figure><figcaption>BEFORE</figcaption><a id="bl"><img id="before"></a></figure><figure><figcaption>AFTER</figcaption><a id="al"><img id="after"></a></figure></main><h2>各時点のインスタンスと詳細</h2><div class="grid" id="observations"></div>
<h2>比較結果</h2><div class="grid" id="judgments"></div><h2>同じ実測値をプログラムだけで比較</h2><div class="card" id="program"></div>
<p class="small">「主対象」は各例の移動・色変化・出現等を認識したか。「整合」は誤った追加変化や矛盾がないか。無変化で得点を稼がないよう、移動5対象も独立に表示。曖昧対応例は9例の主得点に含めません。独立観測の時間は2回の観測＋比較の合計です。</p><p><a href="summary.json">集計</a> / <a href="semantic-review.json">意味評価</a> / <a href="plan.json">条件</a> / <a href="verification.json">検証</a></p>
<script>const data=DATA,el=x=>document.getElementById(x),esc=x=>String(x??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const names={direct_pair:'前後画像を直接比較',independent_text:'Qwen独立観測→文章のみ',measured_text:'実測値→文章のみ',measured_images:'実測値＋前後画像',deterministic:'実測値→プログラム判定'},palette=['#fff','#ccc','#999','#666','#333','#000','#e53aa3','#ff7bcc','#f93c31','#1e93ff','#88d8f1','#ffdc00','#ff851b','#921231','#4fcc30','#a356d6'];
for(const c of data.cases)el('case').add(new Option(c.title,c.name));
el('metrics').innerHTML='<table><tr><th>方式</th><th>移動対象</th><th>主対象の正答</th><th>整合する正答</th><th>有効な経路</th><th>時間中央値</th></tr>'+data.summary.arms.map(a=>'<tr><td>'+names[a.arm]+'</td><td>'+a.moving_targets_correct+'/5</td><td>'+a.primary_correct+'/9</td><td>'+a.strict_correct+'/9</td><td>'+a.pipeline_valid+'/10</td><td>'+(a.median_pipeline_seconds===null?'モデル呼出しなし':a.median_pipeline_seconds.toFixed(2)+'秒')+'</td></tr>').join('')+'</table>';
function pattern(p){if(!Array.isArray(p)||p.length!==3||!p.every(x=>Array.isArray(x)&&x.length===3&&x.every(v=>Number.isInteger(v)&&v>=0&&v<16)))return '';return '<span class="pattern" style="grid-template-columns:repeat(3,18px)">'+p.flat().map(v=>'<span class="cell" style="background:'+palette[v]+'" title="color '+v+'"></span>').join('')+'</span>'}
function draw(){const c=el('case').value,source=el('source').value;for(const fr of ['before','after']){el(fr).src=c+'-'+fr+'.png';el(fr==='before'?'bl':'al').href=el(fr).src}
el('observations').innerHTML=['before','after'].map(fr=>{const r=data.observations[c][fr],s=source==='measured'?data.states[c][fr]:r.parsed;let h='<section class="card"><h3>'+fr.toUpperCase()+'</h3>'+(source==='vlm'?'<p class="'+(r.valid?'ok':'bad')+'">'+(r.valid?'形式確認済み':'形式不備・打切り等')+' / '+r.seconds.toFixed(2)+'秒</p>':'');if(s?.instances)h+='<table><tr><th>ID／クラス</th><th>範囲・見た目</th><th>セル配列</th></tr>'+s.instances.map(o=>'<tr><td>'+esc(o.id)+'<br>'+esc(o.class_name)+'</td><td>'+esc(JSON.stringify(o.bbox))+'<br>'+esc(o.appearance||'')+'</td><td>'+pattern(o.pattern)+'</td></tr>').join('')+'</table>';h+='<details><summary>取得した詳細の全文</summary><pre>'+esc(source==='measured'?JSON.stringify(s,null,2):r.answer)+'</pre></details></section>';return h}).join('');
el('judgments').innerHTML=Object.keys(names).filter(a=>a!=='deterministic').map(arm=>{const r=data.judges.find(r=>r.case===c&&r.arm===arm),v=r.review;return '<section class="card"><h3>'+names[arm]+'</h3><p class="'+(v.strict?'ok':'bad')+'">主対象 '+(v.primary?'正答':'未達')+' / 比較回答 '+(r.valid?'形式有効':'形式不備')+' / 比較処理 '+r.seconds.toFixed(2)+'秒</p><p>'+esc(v.note)+'</p><pre>'+esc(JSON.stringify(r.parsed?.changes??[],null,2))+'</pre><details><summary>回答全文・共通クラス・同一パターン</summary><pre>'+esc(r.answer)+'</pre></details></section>'}).join('');
const review=data.reviews.rows.find(r=>r.case===c&&r.arm==='deterministic');el('program').innerHTML='<p>'+esc(review.note)+'</p><pre>'+esc(JSON.stringify(data.program[c],null,2))+'</pre>';document.body.dataset.ready=c+'|'+source;}
el('case').onchange=draw;el('source').onchange=draw;draw();</script></html>'''
(out/'gallery.html').write_text(html.replace('DATA',json.dumps(data,ensure_ascii=False).replace('</','<\/')))
print('Wrote summary and gallery')
