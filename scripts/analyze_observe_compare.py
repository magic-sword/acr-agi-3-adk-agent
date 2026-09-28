"""Create paired, auditable HTML and structured metrics from frozen runs."""
import argparse,json,shutil,statistics,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.benchmark_parallel import read_lines
from scripts.benchmark_recognition import save_json
from scripts.summarize_temporal_overlay import audit_key,review,summarize

def collect(out):
    rows=[]
    for backend in ['llama','official']:
        p=out/f'{backend}-measurements.jsonl'
        if p.exists():rows+=read_lines(p)
    (out/'measurements.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    review(out)
    return rows

def finish(out):
    rows=collect(out);n=json.loads((out/'plan.json').read_text())['requests']
    assert len(rows)==n+4 and sum(not r['warmup'] for r in rows)==n
    summarize(out)
    result=json.loads((out/'summary.json').read_text());cats={c['key']:c for c in json.loads((out/'catalogs.json').read_text())}
    old=read_lines(Path('outputs/official-relations-20260928/measurements.jsonl'))
    reproduction=[]
    for a in result['arms']:
        rs=[r for r in rows if not r['warmup'] and r['arm']==a['arm']]
        a['requests']=len(rs)
        a['mean_output_tokens']=statistics.mean(r['output_tokens'] for r in rs)
        a['median_output_tokens']=statistics.median(r['output_tokens'] for r in rs)
        good=[r for r in rs if r['valid'] and r['within_budget']]
        a['valid_median_seconds']=statistics.median(r['seconds'] for r in good) if good else None
        a['valid_median_output_tokens']=statistics.median(r['output_tokens'] for r in good) if good else None
        a['moving_targets_in_valid_answers']=sum(len(cats[r['catalog_key']]['mover_ids']) for r in good)
        a['micro_moving_targets_in_valid_answers']=sum(len(cats[r['catalog_key']]['mover_ids']) for r in good if r['case']!='ls20_up')
        a['over_20_seconds']=sum(not r['within_20_seconds'] for r in rs)
        a['truncated']=sum(not r.get('stopped',False) for r in rs)
        a['identical_before_after_on_movers']=[]
        for r in rs:
            if r['method']=='evidence':
                for k in cats[r['catalog_key']]['mover_ids']:
                    o=next((o for o in r.get('parsed',{}).get('observations',[]) if o['id']==k),None)
                    if o and isinstance(o.get('before'),str) and isinstance(o.get('after'),str):
                        a['identical_before_after_on_movers'].append(dict(id=r['id'],target=k,identical=o['before'].strip().casefold()==o['after'].strip().casefold(),valid=r['valid']))
            else:
                previous=next(o for o in old if not o['warmup'] and o['case']==r['case'] and o['repeat']==r['repeat'] and o['arm']==r['backend']+'_base')
                reproduction.append(dict(id=r['id'],case=r['case'],repeat=r['repeat'],backend=r['backend'],same_answer=r['answer']==previous['answer']))
    result['baseline_reproduction']=reproduction
    save_json(out/'summary.json',result)
    gallery(out,rows,result)

def gallery(out,rows,result):
    cases=json.loads((out/'cases.json').read_text());cats=json.loads((out/'catalogs.json').read_text())
    reviewed={r['id']:r['review'] for r in json.loads((out/'review-ledger.json').read_text())['rows']}
    for c in cases:
        if c['name']=='color_only':c['note']='評価用の正解：十字は同じ位置のまま白から橙に変化。移動していません。この正解文はモデルには与えていません。'
    (out/'images').mkdir(exist_ok=True)
    for c in cases:
        for fr in ['before','after']:
            name=c['name']+'-'+fr+'.png';shutil.copyfile(Path('outputs/video-comparison-20260928')/name,out/'images'/name)
    data=dict(plan=json.loads((out/'plan.json').read_text()),cases=[{k:c[k] for k in ['name','title','note']} for c in cases],catalogs=cats,rows=[{k:r[k] for k in ['id','case','repeat','backend','method','valid','seconds','output_tokens','answer','catalog_key']}|dict(parsed=r.get('parsed'),error=r.get('parse_error',r.get('error')),review=reviewed[r['id']]) for r in rows if not r['warmup']],summary=result)
    save_json(out/'gallery-data.json',data)
    html=r'''<!doctype html><html lang="ja"><meta charset="utf-8"><title>観測→比較→分類の検証</title>
<style>body{font:16px/1.6 system-ui;margin:24px;color:#172331;background:#eef2f6}h1{font-size:26px}select{font:inherit;padding:6px;margin:4px}table{border-collapse:collapse;width:100%;background:white;margin:12px 0}td,th{border:1px solid #cdd4df;padding:7px;text-align:left;vertical-align:top}th{background:#e4eaf1}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:13px}main,.images{display:flex;gap:20px;flex-wrap:wrap}.panel{flex:1;min-width:360px}img{width:256px;height:328px;object-fit:contain}figure{margin:0}.ok{color:#116338}.bad{color:#a12c24}.tag{font-weight:bold}.sub{font-size:14px;color:#475667}#note{max-width:1100px}details{margin:12px 0}summary{cursor:pointer}.evidence{border-left:4px solid #617da9;padding-left:10px;margin:8px 0}</style>
<h1>観測→比較→分類で移動認識は変わるか</h1>
<p id="note">同じ11組の画像・3通りの物体一覧で比較。上限は両方式1024出力トークン。座標・正確な方向・ゲームの役割は採点しません。特徴差マップはモデルに渡していません。3一覧は独立したゲームの反復ではありません。</p>
<div id="metrics"></div>
<label>画像<select id="case"></select></label><label>環境<select id="backend"><option value="llama">通常量子化版</option><option value="official">公式BF16版</option></select></label><label>事前一覧<select id="repeat"><option>0</option><option>1</option><option>2</option></select></label>
<p id="caseNote"></p><div class="images"><figure><figcaption>BEFORE</figcaption><img id="before"></figure><figure><figcaption>AFTER</figcaption><img id="after"></figure></div>
<details><summary>モデルに与えたBEFORE物体一覧</summary><div id="inventory"></div></details>
<main><section class="panel"><h2>直接判定</h2><div id="direct"></div></section><section class="panel"><h2>観測→比較→分類</h2><div id="evidence"></div></section></main>
<p class="sub">各行の色はIDの分類と正解表の一致です。回答全体の形式不備や対象の説明による認識回復は別扱いです。対象以外の行も表示し、誤検出を確認できます。</p>
<p><a href="summary.json">集計</a> / <a href="semantic-review.json">意味評価</a> / <a href="verification.json">検証結果</a> / <a href="plan.json">事前条件</a></p>
<script>const data=DATA;const el=id=>document.getElementById(id);const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
for(const c of data.cases)el('case').add(new Option(c.title,c.name));
let metric='<table><tr><th>環境・方式</th><th>移動対象の認識</th><th>無変化例の正答</th><th>色変化の正答</th><th>有効回答</th><th>有効回答の時間中央値</th><th>有効回答のトークン中央値</th></tr>';
for(const a of data.summary.arms){const c=a.counts;metric+='<tr><td>'+esc(a.arm)+'</td><td>'+c.moving_targets_correct+'/'+c.moving_targets+'</td><td>'+c.static_correct+'/'+c.static_n+'</td><td>'+c.color_only_correct+'/'+c.color_only_n+'</td><td>'+c.valid+'/'+a.requests+'</td><td>'+(a.valid_median_seconds===null?'—':a.valid_median_seconds.toFixed(2)+'秒')+'</td><td>'+(a.valid_median_output_tokens??'—')+'</td></tr>'}el('metrics').innerHTML=metric+'</table><p class="sub">認識得点は形式不備も失敗として含みます。特に有効回答0件の条件は、知覚能力の評価として解釈できません。時間とトークン数は有効回答だけの中央値です。</p>';
if(data.plan.post_hoc)el('note').textContent='補足：形式不備を受けて追加した6入力の比較。両方式でJSON配列の例を明示。主測定と合算しません。画像は同じ、上限は1024トークン。';
function draw(){const name=el('case').value,backend=el('backend').value,available=data.catalogs.filter(c=>c.case===name).map(c=>c.repeat);if(!available.includes(+el('repeat').value))el('repeat').value=available[0];for(const o of el('repeat').options)o.disabled=!available.includes(+o.value);const repeat=+el('repeat').value,c=data.cases.find(c=>c.name===name),cat=data.catalogs.find(c=>c.case===name&&c.repeat===repeat);el('caseNote').textContent=c.note;for(const f of ['before','after'])el(f).src='images/'+name+'-'+f+'.png';el('inventory').innerHTML=cat.items.map(x=>'<p>'+esc(x.id)+': '+esc(x.description)+'</p>').join('');
for(const method of ['direct','evidence']){const r=data.rows.find(r=>r.case===name&&r.repeat===repeat&&r.backend===backend&&r.method===method);const obs=r.parsed?.observations||[];let h='<p class="'+(r.valid?'ok':'bad')+'">'+(r.valid?'形式確認済み':'形式不備あり')+' / '+r.seconds.toFixed(2)+'秒 / '+r.output_tokens+' tokens</p>';
if(r.review.contradiction)h+='<p class="bad">記述内に矛盾あり：'+esc(r.review.note)+'</p>';if(r.review.recovered_ids.length)h+='<p class="sub">意味評価：'+esc(r.review.note)+'</p>';for(const o of obs){h+='<div class="evidence"><p class="tag '+(o.kind===cat.truth[o.id]?'ok':'bad')+'">'+esc(o.id)+'：'+esc(o.kind)+' <span class="sub">正解：'+esc(cat.truth[o.id])+'</span></p>';if(method==='evidence')h+='<div>前：'+esc(o.before)+'</div><div>後：'+esc(o.after)+'</div>';h+='<div>記述：'+esc(o.description)+'</div></div>'}h+='<details'+(r.valid?'':' open')+'><summary>生成した全文</summary><pre>'+esc(r.answer)+'</pre></details>';el(method).innerHTML=h}}
for(const id of ['case','backend','repeat'])el(id).onchange=draw;draw();</script></html>'''
    (out/'gallery.html').write_text(html.replace('DATA',json.dumps(data,ensure_ascii=False).replace('</','<\/')))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['collect','finish']);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.command=='collect':collect(a.output)
    else:finish(a.output)
