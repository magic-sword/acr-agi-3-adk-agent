"""Human-readable gallery and reproducibility checks for rule inference."""
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.benchmark_rule_inference import measure_case, request_for
from scripts.benchmark_parallel import digest, read_lines
from scripts.benchmark_official_relations import sha
from scripts.benchmark_recognition import save_json

out = Path(sys.argv[1])
load = lambda n: json.loads((out/n).read_text())
cases, measured, jobs, plan, summary = [load(n) for n in ['cases.json', 'measured.json', 'jobs.json', 'plan.json', 'summary.json']]
assert digest(cases) == plan['cases_digest'] and digest(jobs) == plan['jobs_digest']
rebuilt = {c['name']: measure_case(c, 5000+i*1000) for i, c in enumerate(cases)}
# The frozen prepare hash used integer color-count keys before JSON serialization.
assert digest(rebuilt) == plan['measured_digest']
assert json.loads(json.dumps(rebuilt)) == measured
for p, h in plan['sources'].items(): assert sha(p) == h
for i, c in enumerate(cases):
    assert json.loads(json.dumps(rebuilt[c['name']])) == measured[c['name']]
    for j in [j for j in jobs if j['case'] == c['name']]:
        p = request_for(c, measured[c['name']], j['arm'])
        assert p == j['payload'] and digest(p) == j['payload_digest']
        assert all(isinstance(m['content'], str) for m in p['messages'])
responses = read_lines(out/'responses.jsonl')
main = [r for r in responses if not r['warmup']]
assert sorted(r['id'] for r in main) == list(range(len(jobs)))
for r in responses:
    j = jobs[r['id']]
    assert r['payload_digest'] == j['payload_digest']
    if 'response' in r:
        assert r['response']['usage']['prompt_tokens'] + j['payload']['max_tokens'] < 16384
assert load('server-before.json') == load('server-after.json') and load('health-after.json')['status'] == 'ok'
extra_runs = []
for arg in sys.argv[2:]:
    directory = Path(arg)
    extra_plan = json.loads((directory/'plan.json').read_text())
    extra_cases = json.loads((directory/'cases.json').read_text())
    extra_jobs = json.loads((directory/'jobs.json').read_text())
    extra_rows = read_lines(directory/'responses.jsonl')
    assert extra_cases == cases
    assert digest(extra_jobs) == extra_plan['jobs_digest'] and digest(cases) == extra_plan['cases_digest']
    assert digest(measured) == extra_plan['source_measurements_digest']
    for path, h in extra_plan['sources'].items(): assert sha(path) == h
    if extra_plan['arms'] == ['brief']:
        from scripts.benchmark_rule_brief import request_for as extra_request
    else:
        from scripts.benchmark_rule_split import request_for as extra_request
    for j in extra_jobs:
        c = next(c for c in cases if c['name'] == j['case'])
        p = extra_request(c, measured[c['name']], j['task']) if 'task' in j else extra_request(c, measured[c['name']])
        assert p == j['payload'] and digest(p) == j['payload_digest']
        if j.get('task') == 'action':
            ctx = json.loads(p['messages'][1]['content'])
            assert 'query_action' not in ctx and 'prediction_options' not in ctx
        if j.get('task') == 'prediction': assert 'target' not in json.loads(p['messages'][1]['content'])
    extra_main = [r for r in extra_rows if not r['warmup']]
    assert sorted(r['id'] for r in extra_main) == list(range(len(extra_jobs)))
    for r in extra_rows:
        assert r['payload_digest'] == extra_jobs[r['id']]['payload_digest']
        assert r['response']['usage']['prompt_tokens'] + extra_jobs[r['id']]['payload']['max_tokens'] < 16384
    assert json.loads((directory/'server-before.json').read_text()) == json.loads((directory/'server-after.json').read_text()) == load('server-before.json')
    assert json.loads((directory/'health-after.json').read_text())['status'] == 'ok'
    extra_summary = json.loads((directory/'summary.json').read_text())
    summary['arms'].extend(extra_summary['arms']); summary['rows'].extend(extra_summary['rows'])
    extra_runs.append(dict(directory=str(directory), requests=len(extra_main), warmups=len(extra_rows)-len(extra_main), exploratory=True))
summary['query_echo'] = {a: dict(count=sum(r['parsed'].get('action_choice') == next(c for c in cases if c['name']==r['case'])['query_action'] for r in summary['rows'] if r['arm']==a), total=14) for a in ['states','changes','relations','brief'] if any(r['arm']==a for r in summary['rows'])}
save_json(out/'summary-combined.json', summary)
review = load('semantic-review.json')
assert {(r['case'],r['arm']) for r in review['rows']} == {(r['case'],r['arm']) for r in summary['rows'] if r['joint'] and r['family'] != 'uncertain'}
data = dict(cases=cases, measured=measured, summary=summary, review=review)
html = r'''<!doctype html><html lang="ja"><meta charset="utf-8"><title>測定済みの変化からルールを推論</title>
<style>body{font:16px/1.65 system-ui;background:#edf2f6;color:#172332;margin:24px}h1{font-size:26px}table{border-collapse:collapse;width:100%;background:white}td,th{border:1px solid #c9d3df;padding:8px;text-align:left}th{background:#dce5ee}select{font:inherit;padding:8px;margin:12px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:16px}.card{background:white;padding:16px;border-radius:8px;margin:10px 0}img{width:250px;max-width:100%;image-rendering:pixelated}figure{margin:0;display:inline-block}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:13px/1.5 monospace}.ok{color:#176440}.bad{color:#a12b20}summary{cursor:pointer}small{color:#46576a}.badge{background:#e1e8f0;padding:3px 9px;border-radius:6px}</style>
<h1>測定済みの変化・個体の関係 → Qwenの局所ルール推論</h1>
<p>合成ゲーム4種類×3条件、情報不足の対照2例。Qwenには画像を送信せず、元画素から独立に測定した情報を与えています。画像は人間の確認用です。ゲームの規則・役割・正解はモデル入力に含めません。</p>
<p>「予測」は未観測の開始状態でボタンを押した後の状態を選択。「操作」は目標状態を実現するボタンを選択。12例は単純な局所規則の外挿であり、普遍的なルールの証明や実ゲーム攻略成功ではありません。</p>
<p>「短縮版」と「1問ずつ」は初回の回答を確認して追加した探索条件です。「1問ずつ」は2要求の所要時間を合算しています。選択肢の正答だけでは規則説明の正しさを保証しません。一般的な勝利条件・役割名の獲得は今回の評価対象外です。</p>
<p><strong>両方の選択肢に正答した6件にも、規則説明の矛盾がありました。</strong>各回答の注記と<a href="semantic-review.json">説明の整合性レビュー</a>を併せて確認してください。<a href="../../docs/rule-inference-comparison-20260928-ja.md">日本語の詳細報告</a></p>
<div id="metrics"></div><label>例<select id="case"></select></label><p id="note"></p>
<div class="grid" id="query"></div><h2>予測候補（順序は固定乱数で入れ替え）</h2><div class="grid" id="options"></div>
<h2>事前に観測した操作と変化</h2><p>各試行は独立した開始状態から実行。WAITでも同じ時間だけ経過します。</p><div class="grid" id="trials"></div>
<h2>Qwenの推論と採点</h2><div class="grid" id="answers"></div><details><summary>画素から測定した記録</summary><pre id="records"></pre></details>
<p><a href="summary.json">集計・全回答</a> / <a href="plan.json">固定した試験条件</a> / <a href="verification.json">再現性検査</a></p>
<script>const data=DATA,el=id=>document.getElementById(id),esc=x=>String(x??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const names={states:'各時点の状態のみ',changes:'状態＋測定済み変化',relations:'状態＋変化＋関係',brief:'短縮した変更・関係（探索）',split:'短縮版を1問ずつ提示（探索・2要求）'},families={counter:'1画素・逆方向の連動',orbit:'位置に応じて方向が変わる連動',clock:'自律変化と操作への応答',panels:'3×3パネル間の関係',uncertain:'情報不足・対応の曖昧さ'};
el('metrics').innerHTML='<table><tr><th>入力</th><th>結果予測</th><th>操作選択</th><th>両方正答</th><th>情報不足で両方保留</th><th>時間中央値</th></tr>'+data.summary.arms.map(a=>'<tr><td>'+names[a.arm]+'</td><td>'+a.prediction+'/12</td><td>'+a.action+'/12</td><td>'+a.joint+'/12</td><td>'+a.uncertainty_joint+'/2</td><td>'+a.median_seconds.toFixed(2)+'秒</td></tr>').join('')+'</table>';
for(const c of data.cases)el('case').add(new Option(families[c.family]+' / '+c.name,c.name));
function fig(c,label,title){return '<figure><figcaption>'+esc(title)+'</figcaption><a href="'+c.name+'-'+label+'.png"><img src="'+c.name+'-'+label+'.png"></a></figure>'}
function draw(){const c=data.cases.find(x=>x.name===el('case').value);el('note').textContent=c.note;
el('query').innerHTML='<section class="card">'+fig(c,'current','現在の状態：予測する操作 '+c.query_action)+'</section><section class="card">'+fig(c,'target','目標状態：A・B・WAITのどれで実現するか')+'<p>評価用正解（モデルには非提示）：予測 '+esc(c.truth.prediction_choice)+' / 操作 '+esc(c.truth.action_choice)+'</p></section>';
el('options').innerHTML=c.options.map(o=>'<section class="card">'+fig(c,o.id,o.id)+'</section>').join('');
el('trials').innerHTML=c.trials.map(t=>'<section class="card"><b>'+t.id+' / '+t.action+'</b><br>'+fig(c,t.id+'-before','BEFORE')+fig(c,t.id+'-after','AFTER')+'</section>').join('');
el('answers').innerHTML=data.summary.rows.filter(r=>r.case===c.name).sort((a,b)=>Object.keys(names).indexOf(a.arm)-Object.keys(names).indexOf(b.arm)).map(r=>'<section class="card"><h3>'+names[r.arm]+'</h3><p class="'+(r.joint?'ok':'bad')+'">予測 '+(r.prediction?'正答':'未達')+' / 操作 '+(r.action?'正答':'未達')+' / '+r.seconds.toFixed(2)+'秒</p><p class="bad">'+esc(data.review.rows.find(v=>v.case===r.case&&v.arm===r.arm)?.note||'')+'</p><pre>'+esc(JSON.stringify(r.parsed??r.answer,null,2))+'</pre></section>').join('');
el('records').textContent=JSON.stringify(data.measured[c.name],null,2);document.body.dataset.ready=c.name;}
el('case').onchange=draw;draw();</script></html>'''
(out/'gallery.html').write_text(html.replace('DATA', json.dumps(data, ensure_ascii=False).replace('</', '<\\/')))
save_json(out/'verification.json', dict(passed=True, model_requests=len(main), warmups=len(responses)-len(main),
    exploratory_runs=extra_runs, total_model_requests=len(main)+sum(r['requests'] for r in extra_runs),
    checks=['Frozen cases, measurements, jobs and source hashes', 'Recomputed independent single-frame extraction and temporal differences',
            'Reconstructed every exact request; text-only input in all arms', 'Complete job coverage and saved raw responses',
            'Prompt/output budget below context limit', 'Server properties unchanged and health OK'],
    artifacts_sha256={n: sha(out/n) for n in ['cases.json','measured.json','jobs.json','responses.jsonl','summary.json']}))
print('Verified requests and wrote gallery')
