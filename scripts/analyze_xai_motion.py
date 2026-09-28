"""Summarize intervention outputs, area-normalized attention and feature differences."""
import json,sys
from pathlib import Path
import numpy as np
from transformers import AutoTokenizer
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.benchmark_xai_motion import CASES,TARGET,ARMS

def area_mask(box,h,w):
    x0,y0,x1,y1=box;y0*=832/820;y1*=832/820
    xs=np.arange(w)*640/w;ys=np.arange(h)*832/h
    xx=np.maximum(0,np.minimum(xs+640/w,x1)-np.maximum(xs,x0))/(640/w)
    yy=np.maximum(0,np.minimum(ys+832/h,y1)-np.maximum(ys,y0))/(832/h)
    return yy[:,None]*xx[None,:]

def analyze(out):
    tokenizer=AutoTokenizer.from_pretrained('/tmp/model-cache/qwen3-vl-4b-official',local_files_only=True)
    records=[];attention=[];features=[];visual={}
    for backend in ['llama','official']:
        rows=[json.loads(s) for s in (out/f'{backend}.jsonl').read_text().splitlines()]
        assert len(rows)==56
        for r in rows:
            if r['warmup']:continue
            r['backend']=backend;records.append(r)
    for case in CASES:
        meta=json.loads((out/f'{case}-attention-meta.json').read_text())
        maps=np.load(out/f'{case}-feature-maps.npz');attn=np.load(out/f'{case}-attention.npz')
        visids=attn['visual_indices'];fulltext=meta['decoded_text']
        encoding=tokenizer(fulltext,add_special_tokens=False,return_offsets_mapping=True)
        # Decoding/re-encoding must preserve the attention token indices.
        assert encoding['input_ids']==tokenizer.convert_tokens_to_ids(meta['token_strings'])
        begin=fulltext.index('BEFORE inventory:');end=fulltext.index('<|im_end|>',begin)
        inventory=np.array([i for i,(a,b) in enumerate(encoding['offset_mapping']) if a<end and b>begin])
        visual[case]={'features':{},'attention':{},'meta':{k:meta[k] for k in ['kind','target','layers','max_logit_difference','roi']}}
        for label,m in maps.items():
            target=area_mask(meta['roi'],*m.shape);control=area_mask(meta['control'],*m.shape)
            features.append(dict(case=case,stage=label,mean=float(m.mean()),max=float(m.max()),
                target_mean=float((m*target).sum()/target.sum()),control_mean=float((m*control).sum()/control.sum())))
            visual[case]['features'][label]=m.tolist()
        for layer in meta['layers']:
            weights=attn[str(layer)];assert weights.shape[0]==32
            assert np.allclose(weights.sum(-1),1,atol=1e-6)
            assert np.all(weights[:,meta['query_index']+1:]==0)
            visualmaps=weights[:,visids].reshape(32,2,26,20)
            mean=visualmaps.mean(0)
            target=area_mask(meta['roi'],26,20);control=area_mask(meta['control'],26,20)
            board=area_mask([59,80,571,592],26,20)
            row=dict(case=case,layer=layer+1,kind=meta['kind'],
                before_mass=float(mean[0].sum()),after_mass=float(mean[1].sum()),
                board_mass=float((mean*board).sum()),ui_mass=float((mean*(1-board)).sum()),
                target_mass=float((mean*target).sum()),control_mass=float((mean*control).sum()),
                inventory_mass=float(weights[:,inventory].mean(0).sum()),
                generated_prefix_mass=float(weights[:,meta['prompt_tokens']:meta['query_index']+1].mean(0).sum()),
                target_area_fraction=float(target.mean()),control_area_fraction=float(control.mean()))
            row['target_vs_uniform_visual']=row['target_mass']/(float(mean.sum())*float(target.mean())+1e-20)
            row['control_vs_uniform_visual']=row['control_mass']/(float(mean.sum())*float(control.mean())+1e-20)
            attention.append(row);visual[case]['attention'][str(layer)]=visualmaps.tolist()
    # Actual answers remain available for human semantic review; no keyword-only success score.
    outcomes=[]
    for r in records:
        target=next((x for x in r.get('parsed',{}).get('observations',[]) if x.get('id')==TARGET[r['case']]),None)
        outcomes.append({k:r[k] for k in ['id','backend','case','arm','valid','seconds']}|dict(target=target,observations=r.get('parsed',{}).get('observations'),scores=r.get('candidate_scores')))
    baseline_comparison=[]
    previous=[json.loads(s) for s in Path('outputs/official-relations-20260928/measurements.jsonl').read_text().splitlines()]
    for r in records:
        if r['arm']!='base':continue
        old=next(x for x in previous if x['case']==r['case'] and x['arm']==r['backend']+'_base' and x['repeat']==0 and not x['warmup'])
        baseline_comparison.append(dict(backend=r['backend'],case=r['case'],same_answer=r['answer']==old['answer']))
    result=dict(outcomes=outcomes,features=features,attention=attention,baseline_reproduction=baseline_comparison,
        observed_attention_logit_max_difference=max(visual[c]['meta']['max_logit_difference'] for c in CASES))
    (out/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    (out/'visualization-data.json').write_text(json.dumps(visual,ensure_ascii=False)+'\n')
    make_html(out,visual)
    print(json.dumps({'baseline_reproduction':baseline_comparison,'attention':attention},indent=2))

def make_html(out,data):
    html='''<!doctype html><html lang="ja"><meta charset="utf-8"><title>Qwenの移動認識：内部特徴と注意</title>
<style>body{font:16px/1.6 system-ui;margin:24px;background:#eef1f5;color:#18222c}select,button{font:inherit;padding:5px;margin:4px}main{display:flex;gap:20px;flex-wrap:wrap}canvas{width:320px;height:410px;background:#fff}figure{margin:8px 0}#info{max-width:1050px}pre{white-space:pre-wrap;font:14px/1.5 monospace}.note{max-width:1000px}</style>
<h1>Qwenの移動認識：内部特徴と注意</h1>
<p class="note">注意は判断の原因や人間の視線そのものではありません。公式BF16・SDPAの出力を変えず、実際の分類語の最初のトークンを予測する位置の注意を別計算しました。両時点で同じ色スケールを使います。補間せず、取得した粗いグリッドを表示します。</p>
<label>例<select id="case"></select></label><label>表示<select id="mode"><option value="attention">判断時の注意</option><option value="features">前後の特徴差</option></select></label>
<label>層／段階<select id="layer"></select></label><label>ヘッド<select id="head"><option value="avg">32ヘッドの平均</option></select></label>
<label><input type="checkbox" id="overlay" checked>ヒートマップを重ねる</label>
<main><figure><figcaption>BEFORE</figcaption><canvas id="before" width="640" height="820"></canvas></figure><figure><figcaption>AFTER</figcaption><canvas id="after" width="640" height="820"></canvas></figure></main><pre id="info"></pre>
<p class="note">注意：赤いほど選択した判断トークンへの重みが高い領域です。特徴差：赤いほど、同じ空間位置の前後の特徴ベクトルの相対差が大きい領域です。特徴差は同じ差分マップを両画像に重ねています。例ごと・層ごとに色の最大値が変わるため、例間の色の濃さだけで大小を比較しないでください。</p>
<p><a href="summary.json">領域別・面積補正した集計</a> / <a href="semantic-review.json">回答の確認</a></p>
<script>const data=DATA;const el=id=>document.getElementById(id);for(const c of Object.keys(data)){const o=new Option(c,c);el('case').add(o)}for(let h=0;h<32;h++)el('head').add(new Option('ヘッド '+h,h));
let imgs={},drawId=0;
function layers(){el('layer').innerHTML='';const c=data[el('case').value],mode=el('mode').value;for(const k of Object.keys(c[mode]))el('layer').add(new Option(mode==='attention'?'言語層 '+(+k+1):k,k));el('head').disabled=mode!=='attention';draw()}
async function draw(){const n=++drawId,name=el('case').value,c=data[name],mode=el('mode').value,layer=el('layer').value;let maps;
if(mode==='attention'){const heads=c.attention[layer];if(el('head').value==='avg'){maps=[0,1].map(t=>heads[0][t].map((row,y)=>row.map((_,x)=>heads.reduce((a,h)=>a+h[t][y][x],0)/32)))}else maps=heads[+el('head').value]}else maps=[c.features[layer],c.features[layer]];
let max=Math.max(...maps.flat(2));for(let t=0;t<2;t++){const frame=t?'after':'before',url=name+'-base-'+frame+'.png';if(!imgs[url]){const im=new Image();im.src=url;await im.decode();imgs[url]=im}if(n!==drawId)return;const ctx=el(frame).getContext('2d');ctx.drawImage(imgs[url],0,0);if(el('overlay').checked&&max>0){const m=maps[t],h=m.length,w=m[0].length;for(let y=0;y<h;y++)for(let x=0;x<w;x++){const v=m[y][x]/max;ctx.fillStyle='rgba(255,30,0,'+(v*.8)+')';ctx.fillRect(x*640/w,y*820/h,640/w,820/h)}}const b=c.meta.roi;ctx.strokeStyle='#00ffff';ctx.lineWidth=2;ctx.strokeRect(b[0],b[1],b[2]-b[0],b[3]-b[1]);}
el('info').textContent='生成した分類: '+c.meta.kind+' / 対象ID: '+c.meta.target+'\\n色スケール: 0〜'+max.toPrecision(5)+' / シアンの枠は診断用の対象領域（モデルには未提示）\\n計測有無の次トークンlogit最大差: '+c.meta.max_logit_difference;
}
el('case').onchange=layers;el('mode').onchange=layers;for(const x of ['layer','head','overlay'])el(x).onchange=draw;layers();</script></html>'''
    (out/'gallery.html').write_text(html.replace('DATA',json.dumps(data,ensure_ascii=False)))

if __name__=='__main__':analyze(Path(sys.argv[1]))
