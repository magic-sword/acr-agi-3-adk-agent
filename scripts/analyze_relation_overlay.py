"""Structured scores and interactive 2x2 prompt/image comparison."""
import argparse,json,statistics,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.benchmark_parallel import read_lines
from scripts.benchmark_recognition import save_json
from scripts.summarize_temporal_overlay import review,summarize

def collect(out):
    rows=[]
    for backend in ['llama','official']:
        p=out/f'{backend}-measurements.jsonl'
        if p.exists():rows+=read_lines(p)
    (out/'measurements.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows));review(out)
    return rows

def finish(out):
    rows=collect(out);plan=json.loads((out/'plan.json').read_text())
    assert len(rows)==plan['requests']+plan['warmups'] and sum(not r['warmup'] for r in rows)==plan['requests']
    summarize(out);s=json.loads((out/'summary.json').read_text());s['warmups']=plan['warmups']
    cats={c['key']:c for c in json.loads((out/'catalogs.json').read_text())}
    detail={r['id']:r for r in s['rows']}
    for a in s['arms']:
        rs=[r for r in rows if not r['warmup'] and r['arm']==a['arm']];good=[r for r in rs if r['valid'] and r['within_budget']]
        a.update(requests=len(rs),median_output_tokens=statistics.median(r['output_tokens'] for r in rs),over_20_seconds=sum(not r['within_20_seconds'] for r in rs),
            valid_median_seconds=statistics.median(r['seconds'] for r in good) if good else None,
            valid_median_output_tokens=statistics.median(r['output_tokens'] for r in good) if good else None)
        a['by_case_primary']={r['case']:detail[r['id']]['primary'] for r in rs}
    save_json(out/'summary.json',s);gallery(out,rows,s)

def gallery(out,rows,s):
    reviews={r['id']:r['review'] for r in json.loads((out/'review-ledger.json').read_text())['rows']}
    cats=json.loads((out/'catalogs.json').read_text());cases=json.loads((out/'cases.json').read_text())
    data=dict(plan=json.loads((out/'plan.json').read_text()),summary=s,catalogs=cats,cases=[dict(name=c['name'],title=c['title']) for c in cases],rows=[{k:r[k] for k in ['id','case','backend','variant','valid','seconds','output_tokens','answer','catalog_key']}|dict(parsed=r.get('parsed'),review=reviews[r['id']]) for r in rows if not r['warmup']])
    save_json(out/'gallery-data.json',data)
    html=r'''<!doctype html><html lang="ja"><meta charset="utf-8"><title>相対関係の記述と差分境界画像の比較</title>
<style>body{font:16px/1.6 system-ui;margin:24px;background:#edf1f6;color:#18232e}h1{font-size:26px}select{font:inherit;padding:6px;margin:6px}table{border-collapse:collapse;background:white;width:100%}td,th{border:1px solid #ccd5df;padding:6px;text-align:left}th{background:#e1e8f0}.images{display:flex;gap:16px;flex-wrap:wrap}figure{margin:12px 0}img{width:256px;height:328px;image-rendering:pixelated}main{display:grid;grid-template-columns:repeat(2,minmax(340px,1fr));gap:18px;margin-top:20px}.card{background:white;padding:18px;border-radius:8px}.item{border-top:1px solid #d5dce5;margin-top:12px;padding-top:10px}.ok{color:#14603a}.bad{color:#aa2b22}.sub{font-size:14px;color:#516174}pre{white-space:pre-wrap;overflow-wrap:anywhere}details{margin-top:12px}summary{cursor:pointer}@media(max-width:800px){main{grid-template-columns:1fr}}</style>
<h1>相対関係の記述 × 差分境界画像</h1>
<p id="intro">7組の画像、固定した各1通りの物体一覧。両方式とも中間観測を先に記述します。全条件で元の前後画像と補助画像の計3枚、最大1024出力トークン。境界なしの条件はAFTERを再提示。座標・正確な方向・役割は採点しません。</p>
<div id="metrics"></div><label>例<select id="case"></select></label><label>環境<select id="backend"><option value="llama">通常量子化版</option><option value="official">公式BF16版</option></select></label><label><input id="all" type="checkbox">全対象を表示</label>
<div class="images"><figure><figcaption>元のBEFORE</figcaption><a id="lb"><img id="before"></a></figure><figure><figcaption>元のAFTER／境界なし対照</figcaption><a id="la"><img id="after"></a></figure><figure><figcaption>補助画像：AFTER＋異なる境界</figcaption><a id="le"><img id="edges"></a></figure></div>
<p class="sub">マゼンタ破線：前だけにある色境界。シアン実線：後だけにある色境界。線は注釈で、物体や移動の正解ではありません。色だけの変化と無変化では、補助画像は元のAFTERと同じです。画像をクリックすると原寸で開けます。</p>
<details><summary>モデルに与えたBEFORE物体一覧</summary><div id="inventory"></div></details><main id="panels"></main>
<p class="sub">表示行の色はIDの分類と正解表の一致。説明の誤認・矛盾は意味評価で別記します。対象の正解は表示・採点用で、モデルへの指示には追加していません。</p>
<p><a href="summary.json">集計</a> / <a href="semantic-review.json">意味評価</a> / <a href="plan.json">比較条件</a> / <a href="verification.json">検証</a></p>
<script>const data=DATA,el=id=>document.getElementById(id),esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const names={appearance_repeat:'見た目・境界なし',relation_repeat:'相対関係・境界なし',appearance_edges:'見た目・差分境界あり',relation_edges:'相対関係・差分境界あり',reference_repeat:'参照物の必須欄・境界なし',reference_edges:'参照物の必須欄・差分境界あり'};
if(data.plan.probe_script_sha256){document.title='参照物の必須欄：追加確認';document.querySelector('h1').textContent='参照物を独立した欄にする追加確認';el('intro').textContent='主測定の指示への従い方を受けて追加した2例の確認。参照物の必須欄を追加し、同じ参照物との関係を前後で記述。境界なし／ありを比較。主測定とは合算しません。'}
for(const c of data.cases)el('case').add(new Option(c.title,c.name));
function draw(){const name=el('case').value,b=el('backend').value,cat=data.catalogs.find(c=>c.case===name);for(const [id,fr,link] of [['before','before','lb'],['after','after','la'],['edges','after_edges','le']]){el(id).src=name+'-'+fr+'.png';el(link).href=el(id).src}
let table='<table><tr><th>方式</th><th>移動対象の認識</th><th>有効回答</th><th>誤った注釈の解釈</th><th>有効回答の時間中央値</th></tr>';
for(const a of data.summary.arms.filter(a=>a.arm.startsWith(b+'_'))){table+='<tr><td>'+names[a.arm.slice(b.length+1)]+'</td><td>'+a.counts.moving_targets_correct+'/'+a.counts.moving_targets+'</td><td>'+a.counts.valid+'/'+a.requests+'</td><td>'+a.counts.artifact_hallucination_answers+'/'+a.requests+'</td><td>'+(a.valid_median_seconds===null?'—':a.valid_median_seconds.toFixed(2)+'秒')+'</td></tr>'}el('metrics').innerHTML=table+'</table>';
el('inventory').innerHTML=cat.items.map(x=>'<p>'+esc(x.id)+': '+esc(x.description)+'</p>').join('');let panels='';
for(const [variant,label] of Object.entries(names).filter(([v])=>data.rows.some(r=>r.variant===v))){const r=data.rows.find(r=>r.case===name&&r.backend===b&&r.variant===variant);let h='<section class="card"><h2>'+label+'</h2><p class="'+(r.valid?'ok':'bad')+'">'+(r.valid?'形式確認済み':'形式不備あり')+' / '+r.seconds.toFixed(2)+'秒 / '+r.output_tokens+' tokens</p>';
if(data.plan.probe_script_sha256||r.review.contradiction||r.review.artifact_hallucination||r.review.rejected_ids.length)h+='<p class="bad">'+esc(r.review.note)+'</p>';
for(const o of r.parsed?.observations||[]){if(!el('all').checked&&name!=='synthetic_static'&&cat.truth[o.id]==='unchanged'&&o.kind==='unchanged')continue;h+='<div class="item"><strong class="'+(o.kind===cat.truth[o.id]?'ok':'bad')+'">'+esc(o.id)+': '+esc(o.kind)+'</strong> <span class="sub">正解：'+esc(cat.truth[o.id])+'</span>'+(o.reference?'<div>参照物：'+esc(o.reference)+'</div>':'')+'<div>前：'+esc(o.before)+'</div><div>後：'+esc(o.after)+'</div><div>比較：'+esc(o.description)+'</div></div>'}
h+='<details'+(r.valid?'':' open')+'><summary>生成した全文</summary><pre>'+esc(r.answer)+'</pre></details></section>';panels+=h}el('panels').innerHTML=panels}
for(const x of ['case','backend','all'])el(x).onchange=draw;draw();</script></html>'''
    (out/'gallery.html').write_text(html.replace('DATA',json.dumps(data,ensure_ascii=False).replace('</','<\/')))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['collect','finish']);p.add_argument('--output',required=True,type=Path);a=p.parse_args()
    if a.command=='collect':collect(a.output)
    else:finish(a.output)
