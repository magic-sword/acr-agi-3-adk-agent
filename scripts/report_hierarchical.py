"""Audit actual adaptive routing and render a self-contained result browser."""
import argparse
import json
from collections import Counter,defaultdict
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.benchmark_parallel import digest,read_lines
from scripts.benchmark_official_relations import sha
from scripts.benchmark_recognition import save_json
from scripts.observation_questions import build_questions
from scripts.hierarchical_questions import make_request,measured_flags,encode_flags,flat_result,KINDS,DETAIL_OPTIONS
from scripts.benchmark_hierarchical import SOURCE


def report(out):
    load=lambda name:json.loads((out/name).read_text())
    plan,subjects,jobs,scenes,summary=[load(n+'.json') for n in ['plan','subjects','jobs','scenes','summary']]
    for name,value in [('subjects',subjects),('jobs',jobs),('scenes',scenes)]:
        assert digest(value)==plan[name+'_digest'],name
    assert sha(SOURCE)==plan['source_data_hash']
    for p,h in plan['sources'].items():
        assert sha(p)==h and sha(out/'sources'/Path(p).name)==h
    for scene in scenes:
        regenerated=build_questions(scene['before'],scene['after'],scene['action'])
        assert digest(json.loads(json.dumps(regenerated)))==digest(scene['generated']),scene['name']
    for s in subjects:assert s['truth']==encode_flags(measured_flags(s['context']))
    calls=read_lines(out/'calls.jsonl');pipelines=read_lines(out/'pipelines.jsonl')
    assert Counter(p['id'] for p in pipelines)==Counter(j['id'] for j in jobs)
    grouped=defaultdict(list)
    for c in calls:
        assert c['valid'] and c['response']['usage']['completion_tokens']==1
        assert len(c['answer'])==1
        payload=make_request(subjects[c['subject_id']]['context'],c['stage'])
        assert digest(payload)==c['payload_digest']
        assert all(isinstance(m['content'],str) for m in payload['messages'])
        if not c['warmup']:grouped[c['job_id']].append(c)
    assert sum(c['warmup'] for c in calls)==plan['warmup_calls']
    for p in pipelines:
        j=jobs[p['id']]
        assert all(p[k]==v for k,v in j.items())
        cc=grouped[p['id']];mode=p['mode'];ctx=subjects[p['subject_id']]['context']
        assert len(cc)==p['calls'] and [c['stage'] for c in cc]==p['stages']
        assert all(c['subject_id']==p['subject_id'] and c['mode']==mode and c['repeat']==p['repeat'] for c in cc)
        if mode=='flat':
            assert p['stages']==KINDS
            pred=flat_result({c['stage']:c['answer'] for c in cc})
        elif mode=='direct':
            assert p['stages']==['detail'];pred=cc[0]['answer']
        else:
            flags=measured_flags(ctx)
            gate=cc[0]['answer'] if mode=='model_gate' else 'X' if flags is None else 'B' if any(flags) else 'A'
            assert gate==p['gate']
            expected=(['gate'] if mode=='model_gate' else [])+(['detail'] if gate!='A' else [])
            assert p['stages']==expected
            pred=cc[-1]['answer'] if gate!='A' else 'A'
        assert pred==p['prediction']
        assert abs(p['seconds']-sum(c['seconds'] for c in cc)-p['host_gate_seconds'])<1e-8
    assert len(calls)==sum(p['calls'] for p in pipelines)+2
    assert load('health-after.json')['status']=='ok'
    assert load('server-before.json')==load('server-after.json')
    coverage=[dict(scene=s['name'],**s['generated']['coverage']) for s in scenes]
    audit=dict(passed=True,subjects=len(subjects),scenes=len(scenes),pipelines=len(pipelines),calls=len(calls),
               all_outputs_one_token=True,hashes_and_regeneration=True,adaptive_routes_verified=True,
               coverage=coverage,uncovered_changed_pixels=sum(s['uncovered_changed_pixels'] for s in coverage))
    save_json(out/'verification.json',audit)
    data=dict(subjects=subjects,scenes=[dict(name=s['name'],coverage=s['generated']['coverage']) for s in scenes],
              arms=summary['arms'],rows=summary['rows'],labels=DETAIL_OPTIONS)
    html='''<!doctype html><html lang="ja"><meta charset="utf-8"><title>階層クイズの比較</title>
<style>body{font:16px system-ui;background:#eef2f7;color:#172438;max-width:1280px;margin:28px auto;padding:0 20px}h1{font-size:27px}table{border-collapse:collapse;background:white;width:100%}td,th{padding:10px;border:1px solid #ccd4df;text-align:left}select{font:inherit;padding:7px;max-width:100%}.frames,.cards{display:flex;gap:16px;flex-wrap:wrap}.frames img{width:360px;max-width:100%;image-rendering:pixelated}.card{background:white;border:1px solid #ccd4df;border-radius:8px;padding:14px;flex:1;min-width:230px}.bad{color:#af1428}.ok{color:#166448}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:white;padding:15px}small{color:#536174}</style>
<h1>物体ごとの階層クイズ：質問数と見落とし</h1>
<p>測定済み状態をテキストで提示。Qwen3-VL-4B、1トークン制約、同じ87候補を各方式2回。画像は閲覧用で、モデルには渡していません。</p>
<div id="summary"></div><p>正解は位置・見た目・大きさの変化の組み合わせです。候補の物理的同一性や、実ゲームの攻略能力を保証する評価ではありません。所要時間は各巡回のリクエスト時間合計の中央値です。</p>
<label>場面 <select id="scene"></select></label> <label>物体候補 <select id="subject"></select></label>
<div class="frames"><div><p>BEFORE</p><img id="before"></div><div><p>AFTER</p><img id="after"></div></div>
<p id="coverage"></p><p id="truth"></p><div id="answers" class="cards"></div>
<details><summary>モデルへ提示した測定状態</summary><pre id="context"></pre></details>
<p><a href="summary.json">集計・全結果</a> · <a href="verification.json">検証記録</a> · <a href="calls.jsonl">応答ログ</a> · <a href="plan.json">事前計画</a></p>
<script>const D=DATA;const names={flat:'全3項目を巡回',model_gate:'Qwen判定 → 詳細',direct:'1問で組み合わせ分類',program_gate:'プログラム判定 → 詳細'};
const el=id=>document.getElementById(id),txt=(id,v)=>el(id).textContent=v;
el('summary').innerHTML='<table><tr><th>方式</th><th>質問数／巡回</th><th>時間／巡回</th><th>全反復で正解</th><th>変化の検知</th><th>不明を維持</th></tr>'+D.arms.map(a=>`<tr><td>${names[a.mode]}</td><td>${a.median_calls_per_pass}</td><td>${a.median_seconds_per_pass.toFixed(2)}秒</td><td>${a.correct_all_repeats}/${a.subjects}</td><td>${a.detected_changes}/${a.known_changed}</td><td>${a.unknown_retained}/${a.unknown}</td></tr>`).join('')+'</table>';
for(const s of D.scenes)el('scene').add(new Option(s.name,s.name));
function drawSubject(){const s=D.subjects.find(x=>String(x.id)===el('subject').value);el('answers').replaceChildren();if(!s){txt('truth','物体候補なし。未説明の差分があれば要確認です。');txt('context','');document.body.dataset.ready=el('scene').value+'|none';return}
txt('truth','測定状態からの正解：'+s.truth+' — '+D.labels[s.truth]);txt('context',JSON.stringify(s.context,null,2));
for(const mode of Object.keys(names)){const card=document.createElement('div');card.className='card';const h=document.createElement('b');h.textContent=names[mode];card.append(h);for(const r of D.rows.filter(r=>r.subject_id===s.id&&r.mode===mode).sort((a,b)=>a.repeat-b.repeat)){const p=document.createElement('p');p.className=r.correct?'ok':'bad';p.textContent=`反復${r.repeat+1}: ${r.prediction} (${D.labels[r.prediction]}) / ${r.calls}問 / ${(r.seconds*1000).toFixed(0)}ms / gate=${r.gate??'なし'}`;card.append(p)}el('answers').append(card)}document.body.dataset.ready=el('scene').value+'|'+s.id}
function drawScene(){const name=el('scene').value,s=D.scenes.find(s=>s.name===name);el('before').src=name+'-before.png';el('after').src=name+'-after.png';el('subject').replaceChildren();for(const o of D.subjects.filter(s=>s.scene===name))el('subject').add(new Option(o.subject+' / '+o.id,String(o.id)));txt('coverage',`変更${s.coverage.changed_pixels}画素／候補外${s.coverage.uncovered_changed_pixels}画素／対応未確定${s.coverage.unresolved_subjects}候補／${s.coverage.requires_review?'要確認':'差分の未カバーなし（意味の網羅性は未保証）'}`);drawSubject()}
el('scene').onchange=drawScene;el('subject').onchange=drawSubject;drawScene();</script></html>'''
    (out/'gallery.html').write_text(html.replace('const D=DATA;', 'const D='+json.dumps(data,ensure_ascii=False).replace('<','\\u003c')+';'))
    print(json.dumps({k:v for k,v in audit.items() if k!='coverage'},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output',type=Path);report(p.parse_args().output)
