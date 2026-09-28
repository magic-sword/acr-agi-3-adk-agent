"""Verify frozen single-token experiment and publish a compact local report."""
import json
from pathlib import Path
import statistics
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.benchmark_one_token import SOURCE,MODES,variant
from scripts.observation_questions import build_questions
from scripts.benchmark_parallel import digest,read_lines
from scripts.benchmark_recognition import save_json
from scripts.benchmark_official_relations import sha
from scripts.report_simple_action import reported_choice

out=Path(sys.argv[1]);load=lambda n:json.loads((out/n).read_text())
plan,cases,jobs,generated,summary=[load(n) for n in ['plan.json','cases.json','jobs.json','generated-questions.json','summary.json']]
assert digest(cases)==plan['cases_digest'] and digest(jobs)==plan['jobs_digest']
for p,h in plan['sources'].items():assert sha(p)==h
assert sha(SOURCE/'jobs.json')==plan['source_jobs_hash']
base={(j['case'],j['arm']):j for j in json.loads((SOURCE/'jobs.json').read_text())}
bycase={c['name']:c for c in cases}
for j in jobs:
    assert j['payload']==variant(base[(j['case'],j['source'])]['payload'],bycase[j['case']],j['mode'])
    assert digest(j['payload'])==j['payload_digest']
    assert not any(c['type']=='image_url' for c in j['payload']['messages'][1]['content'])
rows=read_lines(out/'responses.jsonl');main=[r for r in rows if not r['warmup']]
assert len(rows)==plan['requests']+plan['warmups'] and sorted(r['id'] for r in main)==list(range(len(jobs)))
for r in rows:
    j=jobs[r['id']];assert r['payload_digest']==j['payload_digest']
    assert 'response' in r
    response=r['response'];assert r['completion_tokens']==response['usage']['completion_tokens']<=j['payload']['max_tokens']
    selected,form=reported_choice(r['answer'],bycase[r['case']]);assert selected==r['choice'] and form==r['format']
    assert r['correct']==(selected==bycase[r['case']]['truth'])
    if j['mode']=='grammar1':assert r['strict_valid'] and r['completion_tokens']==1
tokens=load('token-audit.json');assert all(len(v['tokens'])==1 for v in tokens.values())
for g in generated:
    t=next(t for t in bycase[g['case']]['trials'] if t['id']==g['trial'])
    expected=json.loads(json.dumps(build_questions(t['before'],t['after'],t['action'])))
    assert expected=={k:v for k,v in g.items() if k not in ['case','trial','seconds']}
    assert all('truth' not in q and 'answer' not in q for q in g['questions'])
assert load('server-before.json')==load('server-after.json') and load('health-after.json')['status']=='ok'
generation=dict(trials=len(generated),questions=sum(len(g['questions']) for g in generated),median_seconds=statistics.median(g['seconds'] for g in generated),max_seconds=max(g['seconds'] for g in generated),uncovered_changed_pixels=sum(g['coverage']['uncovered_changed_pixels'] for g in generated),requires_review=sum(g['coverage']['requires_review'] for g in generated))
differences=[dict(case=p['case'],source=p['source']) for p in summary['paired'] if p['correct']['baseline16']!=p['correct']['grammar1']]
verification=dict(passed=True,requests=len(main),warmups=len(rows)-len(main),generation=generation,changed_correctness= differences,
    checks=['Frozen code, cases and source jobs','All messages identical across output variants','Tokenizer verified all letters as one token','All raw usage counts within limits','Grammar outputs exactly one allowed letter','Expected length finish treated as a complete enum answer, not a partial JSON','All 522 jobs present','Question generation reproduced without expected answers','Server properties unchanged and health OK'],
    artifacts_sha256={n:sha(out/n) for n in ['cases.json','jobs.json','responses.jsonl','summary.json','generated-questions.json','token-audit.json']})
save_json(out/'verification.json',verification)
data=dict(summary=summary,cases=[{k:c[k] for k in ['name','question','options','truth','truth_text']} for c in cases],generated=generated,generation=generation)
template=r'''<!doctype html><html lang="ja"><meta charset="utf-8"><title>自動出題と1トークン出力</title>
<style>body{font:16px/1.65 system-ui;background:#edf3f7;color:#192637;margin:24px}h1{font-size:26px}table{border-collapse:collapse;background:white;width:100%;margin:12px 0}td,th{border:1px solid #c9d3df;padding:8px;text-align:left}th{background:#dce6ef}.card{background:white;padding:16px;border-radius:8px;margin:12px 0}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:16px}select{font:inherit;padding:8px;margin:10px}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:13px/1.5 monospace}.ok{color:#176340}.bad{color:#a02820}</style>
<h1>自動出題と1トークン出力の比較</h1>
<p>前回と同じ29問×2種類の測定記録。入力を固定し、出力上限16、上限1、上限1＋許可文字制約を各3回比較。物体ごとの自動出題は別の試作であり、その228問の認識精度をこの表が示すわけではありません。</p>
<p>1トークンは回答の出力長です。入力読取り・前処理・通信は残り、認識の完全性は保証しません。<a href="../../docs/one-token-comparison-20260928-ja.md">詳細報告</a></p>
<div id="metrics"></div><label>固定問題<select id="case"></select></label><div class="card" id="question"></div><div class="grid" id="answers"></div>
<h2>正解を使わない自動出題の試作</h2><p id="generation"></p><label>観測<select id="generated"></select></label><label>個別問題<select id="item"></select></label><div class="card"><pre id="generated-detail"></pre></div>
<p><a href="summary.json">集計</a> / <a href="token-audit.json">トークナイザー確認</a> / <a href="plan.json">試験条件</a> / <a href="verification.json">再現性検査</a></p>
<script>const data=DATA,el=id=>document.getElementById(id),esc=x=>String(x??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const mode={baseline16:'上限16',limit1:'上限1',grammar1:'上限1＋文字制約'},src={states:'前後の測定値',facts:'測定済み変更'};
el('metrics').innerHTML='<table><tr><th>入力</th><th>出力設定</th><th>3回とも正答</th><th>許可文字だけ</th><th>生成1トークン</th><th>時間中央値</th></tr>'+data.summary.arms.map(a=>'<tr><td>'+src[a.source]+'</td><td>'+mode[a.mode]+'</td><td>'+a.correct_all_repeats+'/29</td><td>'+a.exact_letter+'/'+a.requests+'</td><td>'+a.one_token+'/'+a.requests+'</td><td>'+(1000*a.median_seconds).toFixed(1)+' ms</td></tr>').join('')+'</table>';
for(const c of data.cases)el('case').add(new Option(c.name,c.name));
function draw(){const c=data.cases.find(c=>c.name===el('case').value);el('question').innerHTML='<b>'+esc(c.question)+'</b><p>'+c.options.map(o=>esc(o.letter+'. '+o.text)).join('<br>')+'</p><p>評価用正解：'+esc(c.truth+'. '+c.truth_text)+'</p>';el('answers').innerHTML=['states','facts'].map(s=>{const p=data.summary.paired.find(p=>p.case===c.name&&p.source===s);return '<section class="card"><h3>'+src[s]+'</h3>'+Object.keys(mode).map(m=>'<p class="'+(p.correct[m]?'ok':'bad')+'">'+mode[m]+'：'+esc(JSON.stringify(p.choices[m]))+' / '+(p.timings[m]*1000).toFixed(1)+' ms</p>').join('')+'</section>'}).join('');document.body.dataset.caseReady=c.name}
for(let i=0;i<data.generated.length;i++){const g=data.generated[i];el('generated').add(new Option(g.case+' / '+g.trial,i))}
function drawItem(){const g=data.generated[Number(el('generated').value)],q=g.questions[Number(el('item').value)];el('generated-detail').textContent=JSON.stringify({coverage:g.coverage,question:q},null,2);document.body.dataset.generatedReady=el('generated').value+'|'+el('item').value}
function drawGenerated(){const g=data.generated[Number(el('generated').value)];el('item').replaceChildren();for(let i=0;i<g.questions.length;i++)el('item').add(new Option(g.questions[i].id,i));drawItem()}
el('generation').textContent=data.generation.trials+'組の前後画素から'+data.generation.questions+'問を生成。抽出・対応づけ・出題を合わせた中央値 '+(data.generation.median_seconds*1000).toFixed(2)+' ms。候補外の変更画素と対応不明は再確認対象として保持。';
el('case').onchange=draw;el('generated').onchange=drawGenerated;el('item').onchange=drawItem;draw();drawGenerated();</script></html>'''
(out/'gallery.html').write_text(template.replace('DATA',json.dumps(data,ensure_ascii=False).replace('</','<\\/')))
print(json.dumps(verification,ensure_ascii=False,indent=2))
