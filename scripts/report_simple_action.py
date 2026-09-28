"""Audit and report factual accuracy separately from one-letter compliance.

Exact option-label matching was added after the model echoed option descriptions.
Only a letter alone or that letter plus its exact option text is accepted. No
semantic guessing or resampling; preserve the pre-specified strict score too.
"""
from collections import Counter
import json
from pathlib import Path
import statistics
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.benchmark_simple_action import ARMS, measure, request_for, parse, render
from scripts.benchmark_parallel import digest, read_lines
from scripts.benchmark_recognition import save_json, decode
from scripts.benchmark_official_relations import sha


def reported_choice(answer,case):
    try:return parse(answer,case),'single_letter'
    except ValueError:pass
    choices=case['options']+[dict(letter='X',text='insufficient evidence')]
    for o in choices:
        if answer.strip() in [o['letter']+'. '+o['text'],o['letter']+') '+o['text']]:
            return o['letter'],'exact_option_label'
    return None,'unparseable'


def main(out):
    load=lambda n:json.loads((out/n).read_text())
    cases,measured,jobs,plan=[load(n) for n in ['cases.json','measured.json','jobs.json','plan.json']]
    assert digest(cases)==plan['cases_digest'] and digest(measured)==plan['measured_digest'] and digest(jobs)==plan['jobs_digest']
    for p,h in plan['sources'].items():assert sha(p)==h
    for p,h in plan['images'].items():assert sha(out/p)==h
    for c in cases:
        assert json.loads(json.dumps(measure(c)))==measured[c['name']]
        for j in [j for j in jobs if j['case']==c['name']]:
            expected=request_for(c,measured[c['name']],j['arm'],out)
            assert expected==j['payload'] and digest(expected)==j['payload_digest']
            imgs=[v for v in expected['messages'][1]['content'] if v['type']=='image_url']
            expected_frames=[t[fr] for t in c['trials'] for fr in (['after'] if c['after_only'] else ['before','after'])]
            assert len(imgs)==(len(expected_frames) if j['arm']=='images' else 0)
            if imgs:
                palette=json.loads(Path('outputs/instance-state-20260928-final/cases.json').read_text())[0]['palette']
                for part,frame in zip(imgs,expected_frames):assert decode(part).tobytes()==render(frame,palette).tobytes()
    allrows=read_lines(out/'responses.jsonl');rows=[r for r in allrows if not r['warmup']]
    assert sorted(r['id'] for r in rows)==list(range(len(jobs)))
    assert len(allrows)==len(jobs)+plan['warmups']
    assert load('server-before.json')==load('server-after.json') and load('health-after.json')['status']=='ok'
    bycase={c['name']:c for c in cases};scored=[]
    for r in rows:
        c=bycase[r['case']];assert r['payload_digest']==jobs[r['id']]['payload_digest']
        choice,form=reported_choice(r.get('answer',''),c)
        stopped=r.get('response',{}).get('choices',[{}])[0].get('finish_reason')=='stop'
        text=next((o['text'] for o in c['options'] if o['letter']==choice),'insufficient evidence' if choice=='X' else 'invalid')
        scored.append({k:r.get(k) for k in ['case','arm','answer','seconds']}|dict(group=c['group'],choice=choice,answer_text=text,
            format=form,valid=choice is not None and stopped,strict_valid=r['valid'],strict_correct=bool(r['valid'] and r.get('choice')==c['truth']),
            correct=bool(stopped and choice==c['truth']),truth=c['truth'],truth_text=c['truth_text'],magnitude=c.get('magnitude'),movement_class=c.get('movement_class'),
            action=c['trials'][0]['action'],prompt_tokens=r.get('response',{}).get('usage',{}).get('prompt_tokens'),truncated=not stopped))
    def score(rr):return dict(correct=sum(r['correct'] for r in rr),total=len(rr))
    arms=[]
    for arm in ARMS:
        rr=[r for r in scored if r['arm']==arm];movement=[r for r in rr if r['group']=='motion']
        arms.append(dict(arm=arm,**score(rr),valid=sum(r['valid'] for r in rr),strict_correct=sum(r['strict_correct'] for r in rr),strict_valid=sum(r['strict_valid'] for r in rr),
            groups={g:score([r for r in rr if r['group']==g]) for g in ['color','motion','panel','binding']},
            motion_by_magnitude={str(n):score([r for r in movement if r['magnitude']==n]) for n in [1,4]},
            actually_moved=score([r for r in movement if r['movement_class']!=3]),unchanged=score([r for r in movement if r['movement_class']==3]),
            wait=score([r for r in movement if r['action']=='WAIT']),
            motion_confusion={truth:dict(Counter(r['answer_text'] for r in movement if r['truth_text']==truth)) for truth in ['blue only','orange only','both blue and orange','neither']},
            median_seconds=statistics.median(r['seconds'] for r in rr),max_prompt_tokens=max(r['prompt_tokens'] or 0 for r in rr)))
    summary=dict(arms=arms,rows=scored,scoring='Factual choice accepts a single letter or a letter followed by exactly its own supplied option text. Exact-label parsing added after responses, with no changes to inputs or reruns. Original strict single-letter score is retained separately.')
    save_json(out/'summary-reviewed.json',summary)
    data=dict(cases=cases,measured=measured,summary=summary)
    template=r'''<!doctype html><html lang="ja"><meta charset="utf-8"><title>操作に紐づく単純な観測質問</title>
<style>body{font:16px/1.6 system-ui;background:#eef3f7;color:#192635;margin:24px}h1{font-size:26px}table{border-collapse:collapse;width:100%;background:white}td,th{border:1px solid #c8d2df;padding:8px;text-align:left}th{background:#dde6ef}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(330px,1fr));gap:16px}.card{background:white;border-radius:8px;padding:16px;margin:10px 0}img{width:256px;max-width:100%;image-rendering:pixelated}figure{display:inline-block;margin:5px}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:13px/1.5 monospace}select{font:inherit;padding:8px;margin:12px}.ok{color:#176343}.bad{color:#a12820}summary{cursor:pointer}</style>
<h1>操作に紐づく単純な質問：どの段階まで認識できるか</h1>
<p>合成29問×3条件。未来予測やルール推論を求めず、観測した事実を1問ずつ質問。画像条件には物体一覧・座標・差分を与えません。測定済み変更の条件は文章の読取り・選択を測る対照です。</p>
<p>回答の「B. orange」のような選択肢本文の付記は内容正答と形式順守を分けて集計しています。本文まで選択肢に完全一致する場合だけ採用し、元の厳密採点も保存しています。<a href="../../docs/simple-action-comparison-20260928-ja.md">詳細報告</a></p>
<div id="metrics"></div><label>問題<select id="case"></select></label><div class="card" id="question"></div><div class="grid" id="frames"></div><h2>回答</h2><div class="grid" id="answers"></div>
<details><summary>プログラムが測定した事実・状態</summary><pre id="records"></pre></details>
<p><a href="summary-reviewed.json">内容正答と形式順守</a> / <a href="summary.json">当初の厳密採点</a> / <a href="plan.json">実験条件</a> / <a href="verification.json">検査結果</a></p>
<script>const data=DATA,el=id=>document.getElementById(id),esc=x=>String(x??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const names={images:'画像のみ',states:'各時点の測定値',facts:'測定済みの変更'},groups={color:'静止画の色',motion:'操作前後の移動',panel:'パネル内部の変化',binding:'操作と変化の対応'},frac=x=>x.correct+'/'+x.total;
el('metrics').innerHTML='<table><tr><th>入力</th><th>色</th><th>移動</th><th>内部変化</th><th>操作との対応</th><th>内容正答</th><th>1文字形式順守</th></tr>'+data.summary.arms.map(a=>'<tr><td>'+names[a.arm]+'</td>'+['color','motion','panel','binding'].map(g=>'<td>'+frac(a.groups[g])+'</td>').join('')+'<td>'+a.correct+'/29</td><td>'+a.strict_valid+'/29</td></tr>').join('')+'</table>';
for(const c of data.cases)el('case').add(new Option(groups[c.group]+' / '+c.name,c.name));
function draw(){const c=data.cases.find(c=>c.name===el('case').value);el('question').innerHTML='<b>'+esc(c.question)+'</b><p>'+c.options.map(o=>esc(o.letter+'. '+o.text)).join('<br>')+'<br>X. insufficient evidence</p><p>評価用正解：'+esc(c.truth+'. '+c.truth_text)+'</p>';
el('frames').innerHTML=c.trials.map(t=>'<section class="card"><b>'+esc(t.id+' / 操作 '+t.action)+'</b><br>'+['before','after'].map(fr=>'<figure><figcaption>'+fr.toUpperCase()+(c.after_only&&fr==='before'?'（参考・モデル非提示）':'')+'</figcaption><a href="'+c.name+'-'+t.id+'-'+fr+'.png"><img src="'+c.name+'-'+t.id+'-'+fr+'.png"></a></figure>').join('')+'</section>').join('');
el('answers').innerHTML=Object.keys(names).map(arm=>{const r=data.summary.rows.find(r=>r.case===c.name&&r.arm===arm);return '<section class="card"><h3>'+names[arm]+'</h3><p class="'+(r.correct?'ok':'bad')+'">内容 '+(r.correct?'正答':'未達')+' / '+r.seconds.toFixed(2)+'秒</p><pre>'+esc(r.answer)+'</pre><p>形式：'+esc(r.format)+'</p></section>'}).join('');
el('records').textContent=JSON.stringify(data.measured[c.name],null,2);document.body.dataset.ready=c.name;}
el('case').onchange=draw;draw();</script></html>'''
    (out/'gallery.html').write_text(template.replace('DATA',json.dumps(data,ensure_ascii=False).replace('</','<\\/')))
    save_json(out/'verification.json',dict(passed=True,requests=len(rows),warmups=len(allrows)-len(rows),
        checks=['Frozen cases/source/input image hashes','Recomputed independent pixel measurements','Reconstructed all exact requests','All image pixels identical to expected nearest-neighbor render','No image or hidden truth in text conditions; no measured facts in image condition','No missing/duplicate jobs','Original strict parser plus exact-label factual score','Server unchanged and health OK'],
        artifacts_sha256={n:sha(out/n) for n in ['cases.json','measured.json','jobs.json','responses.jsonl','summary.json','summary-reviewed.json']}))
    print(json.dumps(arms,ensure_ascii=False,indent=2))


if __name__=='__main__':main(Path(sys.argv[1]))
