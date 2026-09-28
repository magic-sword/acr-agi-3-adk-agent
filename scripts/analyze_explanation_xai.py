"""Readable source attribution and intervention comparison, no causal attention claims."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.benchmark_parallel import read_lines
from scripts.benchmark_recognition import save_json
from scripts.benchmark_explanation_xai import TARGET

def area_weights(box):
    x0,y0,x1,y1=box
    return [max(0,min((x+1)*640/20,x1)-max(x*640/20,x0))/(640/20)*max(0,min((y+1)*820/26,y1)-max(y*820/26,y0))/(820/26) for y in range(26) for x in range(20)]

def analyze(out):
    rows=[r for r in read_lines(out/'measurements.jsonl') if not r['warmup']];assert len(rows)==10
    jobs=json.loads((out/'jobs.json').read_text());prefix=json.loads((out/'prefix-scores.json').read_text());data={};summary=[]
    for case,target in TARGET.items():
        rs=[r for r in rows if r['case']==case];base=next(r for r in rs if r['arm']=='base');j=next(j for j in jobs if j['case']==case and j['arm']=='base')
        trace=json.loads((out/f'{case}-trace-refined.json').read_text());meta=json.loads((out/f'{case}-attention-groups.json').read_text());idx={i:k for k,i in enumerate(meta['selected_answer_token_indices'])}
        target_weights=area_weights(j['roi']);control_weights=area_weights(j['control'])
        for r in trace['attention']:
            qs=[meta['query_indices'][idx[i]] for i in meta['groups'][r['field']]]
            r['mean_eligible_tokens']={label:sum(sum(x==label for x in meta['labels'][:q+1]) for q in qs)/len(qs) for label in r['mass']}
            r['mass_per_token']={label:r['mass'][label]/n if n else 0 for label,n in r['mean_eligible_tokens'].items()}
            maps=trace['image_maps'][r['field']][str(r['layer'])];flat=[v for frame in maps for line in frame for v in line]
            r['image_mass']=sum(flat);r['target_mass']=sum(v*target_weights[i%520] for i,v in enumerate(flat));r['control_mass']=sum(v*control_weights[i%520] for i,v in enumerate(flat))
            r['target_relative_density']=r['target_mass']/(r['image_mass']*sum(target_weights)/520+1e-30)
            r['control_relative_density']=r['control_mass']/(r['image_mass']*sum(control_weights)/520+1e-30)
        treatments=[]
        for r in rs:
            o=next((o for o in r.get('parsed',{}).get('observations',[]) if o['id']==target),None)
            scores=r['fixed_answer_scores'];deltas={k:v['mean_logp']-base['fixed_answer_scores'][k]['mean_logp'] for k,v in scores.items()}
            treatments.append(dict(arm=r['arm'],target=o,valid=r['valid'],answer=r['answer'],scores=scores,delta_mean_logp=deltas,generation_seconds=r['generation_seconds']))
        histories=[]
        for r in prefix:
            if r['case']!=case:continue
            comparable={k:v for k,v in r['scores'].items() if k in base['fixed_answer_scores'] and v['text']==base['fixed_answer_scores'][k]['text']}
            histories.append(dict(arm=r['arm'],scores=comparable,delta_mean_logp={k:v['mean_logp']-base['fixed_answer_scores'][k]['mean_logp'] for k,v in comparable.items()}))
        data[case]=dict(target=target,roi=j['roi'],trace=trace,treatments=treatments,histories=histories,baseline_reproduced=base['baseline_reproduced'])
        summary.append(dict(case=case,target=target,baseline_reproduced=base['baseline_reproduced'],attention=trace['attention'],treatments=treatments,histories=histories))
    save_json(out/'summary.json',dict(cases=summary,requests=10,warmups=1,valid=sum(r['valid'] for r in rows),peak_reserved_gib=max(r['peak_reserved_bytes'] for r in rows)/1024**3))
    save_json(out/'gallery-data.json',data)
    html=r'''<!doctype html><html lang="ja"><meta charset="utf-8"><title>粗い説明の根拠を調べるXAI</title>
<style>body{font:16px/1.6 system-ui;background:#eef2f6;color:#192633;margin:24px}h1{font-size:26px}select{font:inherit;padding:5px;margin:6px}table{border-collapse:collapse;width:100%;background:white;margin:16px 0}td,th{border:1px solid #c9d2dd;padding:7px;text-align:left}th{background:#dfe7ef}main{display:flex;flex-wrap:wrap;gap:18px}figure{margin:0}canvas{width:288px;height:369px}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:white;padding:12px}.note{max-width:1100px}.grid{display:grid;grid-template-columns:1fr 1fr;gap:18px}.box{background:white;padding:16px}.small{font-size:14px}.bad{color:#a32020}@media(max-width:850px){.grid{grid-template-columns:1fr}}</style>
<h1>粗い中間説明は何に依存しているか</h1>
<p class="note">公式BF16版・前回の失敗2例を再現。注意は参照の分布であり、因果的重要度ではありません。実際に生成した説明を固定して追跡し、入力・文章への介入による変化を別に示します。32ヘッドと選択語句の予測位置の平均です。</p>
<label>例<select id="case"></select></label><label>生成箇所<select id="field"></select></label><label>言語層<select id="layer"></select></label><label><input type="checkbox" id="overlay" checked>画像内の注意</label>
<p id="phrase"></p><main><figure><figcaption>BEFORE</figcaption><canvas id="before" width="640" height="820"></canvas></figure><figure><figcaption>AFTER</figcaption><canvas id="after" width="640" height="820"></canvas></figure><figure><figcaption>補助画像（AFTER再提示）</figcaption><canvas id="aux" width="640" height="820"></canvas></figure></main>
<p id="imageinfo" class="small"></p><div class="grid"><div id="mass"></div><div id="tokens"></div></div>
<h2>入力を変えて自由生成した説明</h2><p class="note">黒塗りは対象を消す操作です。同面積の無関係領域も黒塗りして比較。誤った位置説明は診断目的で事前一覧にのみ入れました。処置後の場面を元の移動課題と同じ正解率には集計しません。</p><div id="treatments"></div>
<h2>同じ語句の出やすさへの影響</h2><p class="note">元回答の接頭辞を固定した条件付き対数尤度の差（1トークン平均、自然対数）。負は元の説明が出にくくなったことを示します。原因の寄与率や、自由生成でその説明が出る確率ではありません。生成済み文章の介入はこの採点だけで、自由生成比較ではありません。</p><div id="scores"></div>
<p><a href="summary.json">集計・各層</a> / <a href="semantic-review.json">回答の確認</a> / <a href="plan.json">計測条件</a> / <a href="verification.json">検証</a></p>
<script>const data=DATA,el=x=>document.getElementById(x),esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const fieldNames={before_position_start:'BEFORE位置表現の最初の語',after_position_start:'AFTER位置表現の最初の語',before:'BEFOREの全文',after:'AFTERの全文',before_position:'BEFOREの位置表現',after_position:'AFTERの位置表現',description:'前後の比較文',kind:'最終分類'},armNames={base:'元の入力',target_black:'対象を黒塗り',control_black:'無関係領域を同面積で黒塗り',inventory_no_position:'対象の事前位置説明を削除',inventory_wrong_position:'対象の事前位置説明を誤った位置へ変更',omit_previous_objects:'先に生成した他物体の行を除去',replace_own_before:'自身のBEFORE文をnot recordedへ置換'},sourceNames={chat_template:'チャット役割・区切り',before_image:'BEFORE画像',after_image:'AFTER画像',aux_image:'補助画像',inventory:'事前物体一覧の節',instruction_and_structure:'指示本文・その他の入力形式',previous_objects:'先に生成した他物体の文章',target_structure:'対象行のキー・記号等',generated_before:'自身のBEFORE文',generated_after:'自身のAFTER文',generated_description:'自身の比較文',generated_kind:'自身の分類語'};
for(const c of Object.keys(data))el('case').add(new Option(c,c));for(const l of [1,12,24,36])el('layer').add(new Option(l,l));el('layer').value='24';let images={},drawId=0;
function setup(){el('field').innerHTML='';for(const k of Object.keys(data[el('case').value].trace.scores))el('field').add(new Option(fieldNames[k],k));el('field').value='before_position_start';draw()}
async function draw(){const id=++drawId,name=el('case').value,c=data[name],field=el('field').value,layer=+el('layer').value,r=c.trace.attention.find(r=>r.field===field&&r.layer===layer),maps=c.trace.image_maps[field][layer],max=Math.max(...maps.flat(2));
el('phrase').textContent='対象 '+c.target+' / 計測した語句：'+c.trace.scores[field].text;
for(let i=0;i<3;i++){const frame=['before','after','aux'][i],url=name+'-base-'+frame+'.png';if(!images[url]){const im=new Image();im.src=url;await im.decode();images[url]=im}if(id!==drawId)return;const ctx=el(frame).getContext('2d');ctx.drawImage(images[url],0,0);if(el('overlay').checked&&max>0){for(let y=0;y<26;y++)for(let x=0;x<20;x++){ctx.fillStyle='rgba(255,30,0,'+(maps[i][y][x]/max*.8)+')';ctx.fillRect(x*32,y*820/26,32,820/26)}}const b=c.roi;ctx.strokeStyle='#00ffff';ctx.lineWidth=2;ctx.strokeRect(b[0],b[1],b[2]-b[0],b[3]-b[1])}
el('imageinfo').textContent='3枚合計の注意質量：'+(r.image_mass*100).toFixed(2)+'%。画像内の対象密度／均等密度：'+r.target_relative_density.toFixed(2)+'倍（同面積対照 '+r.control_relative_density.toFixed(2)+'倍）。シアン枠は診断用でモデルには未提示。赤は3枚共通の相対スケール。計測有無のlogit差：'+c.trace.max_logit_difference;
let h='<table><tr><th>注意の参照先</th><th>質量</th><th>文脈内トークン数※</th><th>1トークンあたり</th></tr>';for(const [k,v] of Object.entries(r.mass)){if(!v&&!r.mean_eligible_tokens[k])continue;h+='<tr><td>'+sourceNames[k]+'</td><td>'+(v*100).toFixed(2)+'%</td><td>'+r.mean_eligible_tokens[k].toFixed(1)+'</td><td>'+(r.mass_per_token[k]*100).toFixed(4)+'%</td></tr>'}el('mass').innerHTML=h+'</table><p class="small">※選択語句の予測位置にわたる平均。長い文章群ほど質量が大きくなりやすいため、長さも表示します。</p>';
el('tokens').innerHTML='<table><tr><th>注意が高いテキストトークン</th><th>周辺の文脈</th><th>質量</th></tr>'+r.top_text.map(t=>'<tr><td>'+esc(t.token)+'<br><span class="small">'+sourceNames[t.group]+'</span></td><td>'+esc(t.context)+'</td><td>'+(100*t.weight).toFixed(2)+'%</td></tr>').join('')+'</table>';
el('treatments').innerHTML='<table><tr><th>条件</th><th>対象のBEFORE</th><th>対象のAFTER</th><th>比較文／分類</th></tr>'+c.treatments.map(t=>'<tr><td>'+armNames[t.arm]+(!t.valid?'<br><span class="bad">形式不備</span>':'')+'<details><summary>回答全文</summary><pre>'+esc(t.answer)+'</pre></details></td><td>'+esc(t.target?.before)+'</td><td>'+esc(t.target?.after)+'</td><td>'+esc(t.target?.description)+'<br>'+esc(t.target?.kind)+'</td></tr>').join('')+'</table>';
el('scores').innerHTML='<table><tr><th>介入</th><th>BEFORE位置表現</th><th>AFTER位置表現</th><th>比較文</th><th>分類</th></tr>'+[...c.treatments,...c.histories].map(t=>'<tr><td>'+armNames[t.arm]+'</td>'+['before_position','after_position','description','kind'].map(k=>'<td>'+(t.delta_mean_logp[k]===undefined?'—':t.delta_mean_logp[k].toFixed(4))+'</td>').join('')+'</tr>').join('')+'</table>';
document.body.dataset.ready=[name,field,String(layer)].join('|');
}
el('case').onchange=setup;for(const id of ['field','layer','overlay'])el(id).onchange=draw;setup();</script></html>'''
    (out/'gallery.html').write_text(html.replace('DATA',json.dumps(data,ensure_ascii=False).replace('</','<\/')))
    print('Analyzed',len(rows),'generations and',len(prefix),'prefix interventions')

if __name__=='__main__':analyze(Path(sys.argv[1]))
