"""Render the historical 2026-09-28 comparison from its frozen logs.

The recorded mode labels belong to that experiment, not the current runtime.
This does not execute an agent or provide a legacy compatibility path.
"""
import json
from collections import Counter
from pathlib import Path
import statistics
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from agent.cognition.perception import measure
from scripts.benchmark_parallel import read_lines,digest
from scripts.benchmark_official_relations import sha


def report(out):
    load=lambda p:json.loads(p.read_text())
    summary=load(out/'summary.json');cases=load(out/'cases.json');measured=load(out/'measurements.json')
    for c in cases:
        for t,r in zip(c['trials'],measured[c['name']]):
            again=measure(None if c['after_only'] else t['before'],t['after'],after_id=t['id']+'after',
                          before_id=t['id']+'before',action=t['action'])
            assert {k:v for k,v in again.items() if k!='seconds'}=={k:v for k,v in r.items() if k!='seconds'}
    live=[];frame_records=[];source_sets=[]
    for folder in sorted(out.glob('live-*-*')):
        if not folder.is_dir():continue
        manifest=load(folder/'manifest.json');mode=manifest['cognition_settings']['COGNITION_PERCEPTION_MODE']
        for rel,h in manifest['source_sha256'].items():assert sha(folder/'package'/rel)==h
        source_sets.append(manifest['source_sha256'])
        for game in manifest['games']:
            directory=folder/game;result=load(directory/'result.json');logs=directory/'cognition'
            artifacts=[r for p in logs.glob('*.artifacts.jsonl') for r in read_lines(p)]
            observations=[r for p in logs.glob('*.observations.jsonl') for r in read_lines(p)]
            byid={r['observation_id']:r for r in observations}
            answers=[r for r in artifacts if r['event']=='semantic_question_answered']
            measurements=[r for r in artifacts if r['event']=='objects_measured']
            calls=[r for p in logs.glob('*.model.jsonl') for r in read_lines(p)]
            for r in measurements:
                value=r['measurement'];b=byid.get(value['before_observation_id'],{});a=byid[value['observation_id']]
                regenerated=measure(b.get('grid'),a.get('grid'),before_id=value['before_observation_id'],
                    after_id=value['observation_id'],action=value['action'],boundary=value['status']=='boundary')
                assert {k:v for k,v in value.items() if k!='seconds'}=={k:v for k,v in regenerated.items() if k!='seconds'}
            if mode=='measured':assert len(measurements)==len(observations)
            semantic_calls=[c for c in calls if c['work']=='answer_question']
            for c in semantic_calls:
                if c['schema_valid']:assert c['usage']['completion_tokens']==1
            row=dict(game=game,mode=mode,actions=result['actions'],levels=result['levels_completed'],
                score=result['sdk_score_full_game'],seconds=result['wall_seconds'],stop=result['stop_reason'],
                model_calls=result['model_calls'],calls_by_work=result['reasoning_calls_by_work'],
                measurements=len(measurements),changed_candidates=sum(len(r['measurement']['changes']) for r in measurements),
                review_observations=sum(r['measurement']['status']=='measured' and r['measurement']['requires_review'] for r in measurements),
                semantic_calls=len(semantic_calls),semantic_valid=sum(c['schema_valid'] for c in semantic_calls),
                semantic_interpretations=dict(Counter(r['answer']['interpretation'] for r in answers)),
                measurement_seconds=sum(r['measurement']['seconds'] for r in measurements),
                report=str((folder/'report.md').relative_to(out)),
                errors=result.get('validation_errors',[]))
            live.append(row)
            for o in observations:
                m=next((r['measurement'] for r in measurements if r['observation_id']==o['observation_id']),None)
                stages=[dict(work=r['work'],result=r['result']) for r in artifacts
                        if r['event']=='stage_accepted' and r['observation_id']==o['observation_id']]
                aa=[r['answer'] for r in answers if r['observation_id']==o['observation_id']]
                frame_records.append(dict(game=game,mode=mode,step=o['step'],
                    image=str((logs/o['image_path']).relative_to(out)) if o.get('image_path') else None,
                    measurement=m,stages=stages,answers=aa))
    assert len(live)==6,'All six bounded live runs must finish before publishing report'
    assert all(x==source_sets[0] for x in source_sets),'Live source snapshots differ'
    data=dict(cases=cases,summary=summary,measurements=measured,live=live,frames=frame_records)
    (out/'combined-summary.json').write_text(json.dumps(dict(recognition=summary['arms'],live=live),ensure_ascii=False,indent=2)+'\n')
    timings=[r['seconds'] for rows in measured.values() for r in rows]
    audit=dict(passed=True,fixed_frame_requests=len(summary['rows']),live_runs=len(live),
        live_observations=len(frame_records),identical_live_source_snapshots=True,
        measured_records_recomputed=True,measurement_median_ms=statistics.median(timings)*1000,
        all_measurements_total_ms=sum(timings)*1000,
        limits='Same-frame factual recognition has labels. Live policy paths diverge and semantic answers are hypotheses; no pooled recognition accuracy or causal score is assigned.')
    (out/'combined-verification.json').write_text(json.dumps(audit,indent=2)+'\n')
    template='''<!doctype html><html lang="ja"><meta charset="utf-8"><title>測定を使う認識：実装と比較</title>
<style>body{font:16px/1.6 system-ui;background:#eff3f8;color:#18283d;margin:24px auto;padding:0 20px;max-width:1250px}table{border-collapse:collapse;width:100%;background:white}td,th{border:1px solid #c7d3df;padding:8px;text-align:left}.cards{display:flex;gap:15px;flex-wrap:wrap}.card{background:white;padding:16px;border-radius:10px;flex:1;min-width:240px}img{width:260px;max-width:100%;image-rendering:pixelated}figure{display:inline-block;margin:8px}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:13px monospace}#stages{max-height:520px;overflow:auto}select{padding:8px;font:inherit}.ok{color:#167044}.bad{color:#b22131}h1{font-size:28px}</style>
<h1>プログラム測定＋Qwen：認識と実ゲームの比較</h1>
<p>固定29問の認識比較と、公開3ゲーム×2方式の実行を分けて評価しました。測定入力は実装した認識モジュールが作成しています。意味・因果推論の正確さを、事実の読取り成績と同一視しません。</p>
<h2>同じ画面・同じ質問での認識</h2><div id="metrics"></div>
<p>各問2反復、全応答1トークン。移動16問には無変化4問を含みます。画像のみは実際に動いた12問すべてを見落とし、測定入力は12問すべて正答しました。測定入力への変更は、Qwen単独の視覚能力の改善ではありません。過去に使用した小規模な合成問題です。</p>
<label>問題 <select id="case"></select></label><div id="question" class="card"></div><div id="frames" class="cards"></div><div id="answers" class="cards"></div>
<details><summary>実装が生成した測定記録</summary><pre id="measurements"></pre></details>
<h2>公開ゲームの短時間実行</h2><p>各方式・各ゲーム1回、最大4操作・120秒、1判断60秒。実行により画面が分岐するため、固定問題のような対応のある正答率比較ではありません。レベル達成・時間と、状態遷移が実際に使われたかを確認します。</p>
<div id="live"></div><label>観測 <select id="observation"></select></label><div class="cards"><div class="card"><img id="live-image"><pre id="live-summary"></pre></div><div class="card"><b>モデルの理解・照合（仮説）</b><pre id="stages"></pre></div></div>
<details><summary>この観測のプログラム測定</summary><pre id="live-measurement"></pre></details>
<p><a href="workflow/index.html">実装のステートマシン</a> · <a href="combined-summary.json">集計</a> · <a href="combined-verification.json">検証</a> · <a href="../../docs/visual-recognition-adopted-ja.md">統合資料</a></p>
<script>const D=DATA;const el=id=>document.getElementById(id),names={images:'画像のみ',measured:'プログラム測定入力',legacy:'従来経路'};
const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
el('metrics').innerHTML='<table><tr><th>入力</th><th>色</th><th>移動対象</th><th>内部変化</th><th>操作との対応</th><th>合計</th><th>応答中央値</th></tr>'+D.summary.arms.map(a=>'<tr><td>'+names[a.arm]+'</td>'+['color','motion','panel','binding'].map(g=>'<td>'+a.groups[g].correct+'/'+a.groups[g].total+'</td>').join('')+'<td>'+a.correct+'/29</td><td>'+(a.median_seconds*1000).toFixed(1)+'ms</td></tr>').join('')+'</table>';
for(const c of D.cases)el('case').add(new Option(c.name,c.name));
function draw(){const c=D.cases.find(c=>c.name===el('case').value);el('question').innerHTML='<b>'+esc(c.question)+'</b><p>'+c.options.map(o=>esc(o.letter+'. '+o.text)).join('<br>')+'</p><p>評価用正解：'+esc(c.truth+' '+c.truth_text)+'</p>';
el('frames').innerHTML=c.trials.map(t=>'<div class="card">'+esc(t.id+' / '+t.action)+'<br>'+['before','after'].map(f=>'<figure><figcaption>'+f.toUpperCase()+'</figcaption><img alt="'+f+'" src="'+c.name+'-'+t.id+'-'+f+'.png"></figure>').join('')+'</div>').join('');
el('answers').innerHTML=['images','measured'].map(arm=>'<div class="card"><b>'+names[arm]+'</b>'+D.summary.rows.filter(r=>r.case===c.name&&r.arm===arm).sort((a,b)=>a.repeat-b.repeat).map(r=>'<p class="'+(r.correct?'ok':'bad')+'">反復'+(r.repeat+1)+': '+esc(r.answer)+' / '+(r.correct?'正答':'誤答')+' / '+(r.seconds*1000).toFixed(0)+'ms</p>').join('')+'</div>').join('');
el('measurements').textContent=JSON.stringify(D.measurements[c.name],null,2);document.body.dataset.caseReady=c.name;}
el('case').onchange=draw;draw();
el('live').innerHTML='<table><tr><th>ゲーム</th><th>方式</th><th>操作</th><th>到達レベル</th><th>全体時間</th><th>意味質問</th><th>停止理由</th></tr>'+D.live.map(r=>`<tr><td><a href="${r.report}">${r.game}</a></td><td>${names[r.mode]}</td><td>${r.actions}</td><td>${r.levels}</td><td>${r.seconds.toFixed(1)}秒</td><td>${r.semantic_calls}</td><td>${r.stop}</td></tr>`).join('')+'</table>';
D.frames.forEach((r,i)=>el('observation').add(new Option(r.game+' / '+names[r.mode]+' / step '+r.step,String(i))));
function drawObservation(){const i=Number(el('observation').value),r=D.frames[i];if(r.image){el('live-image').src=r.image;el('live-image').hidden=false}else{el('live-image').hidden=true}el('live-summary').textContent=JSON.stringify({step:r.step,answers:r.answers},null,2);el('stages').textContent=JSON.stringify(r.stages,null,2);el('live-measurement').textContent=JSON.stringify(r.measurement,null,2);document.body.dataset.observationReady=String(i)}
el('observation').onchange=drawObservation;drawObservation();</script></html>'''
    (out/'gallery.html').write_text(template.replace('const D=DATA;', 'const D='+json.dumps(data,ensure_ascii=False).replace('<','\\u003c')+';'))
    print(json.dumps(dict(live=live,verification=audit),ensure_ascii=False,indent=2))


if __name__=='__main__':report(Path(sys.argv[1]))
