"""Compare frozen candidate-first measurements; retain per-stage failures."""
import argparse
from collections import Counter
import html
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.benchmark_region_descriptions import descriptions,verify,eligible,save


def result(out):
    plan,cases,_=verify(out);_,baseline,desc=descriptions(out)
    scores=json.loads((out/'binding-scores.json').read_text());summary=json.loads((out/'summary.json').read_text())
    match=list(map(json.loads,(out/'match-responses.jsonl').read_text().splitlines()))
    source={r['id']:r for r in baseline+match};dby={(d['case'],d['candidate_id']):d for d in desc}
    summary['per_scene']=[];summary['stage_recall']={};summary['stage_recall_primary']={}
    for group in ['synthetic','real']:
        cs=[c for c in cases if (c['origin']=='synthetic')==(group=='synthetic')];gate=matched=total=0
        for c in cs:
            for g in c['gold']:
                total+=1;good=[o['id'] for o in c['candidates'] if eligible(o,g)]
                gate+=any(dby[c['name'],i]['valid'] and dby[c['name'],i]['extent']=='whole' for i in good)
                cid=next(k['id'] for k in c['classes'] if g['id'] in k['members'])
                matched+=any(r['case']==c['name'] and r['class_id']==cid and r['candidate_id'] in good and r['answer'].strip()=='1' for r in match)
        summary['stage_recall'][group]=dict(targets=total,eligible_candidate_passes_whole_gate=gate,eligible_candidate_passes_text_match=matched)
        primary=[s for s in scores if s['comparable'] and (s['origin']=='synthetic')==(group=='synthetic')]
        pg=pm=0
        for s in primary:
            c=next(c for c in cases if c['name']==s['case']);g=next(g for g in c['gold'] if g['id']==s['gold_id'])
            good=[o['id'] for o in c['candidates'] if eligible(o,g)]
            pg+=any(dby[c['name'],i]['valid'] and dby[c['name'],i]['extent']=='whole' for i in good)
            pm+=bool(set(good)&set(s['class_matches']))
        summary['stage_recall_primary'][group]=dict(targets=len(primary),eligible_candidate_passes_whole_gate=pg,eligible_candidate_passes_text_match=pm)
    summary['baseline_cost']={}
    for stage in ['baseline_binding','baseline_discovery']:
        rows=[r for r in baseline if r['stage']==stage]
        summary['baseline_cost'][stage]=dict(calls=len(rows),seconds=sum(r['seconds'] for r in rows))
    for c in cases:
        ds=[d for d in desc if d['case']==c['name']];ms=[r for r in match if r['case']==c['name']]
        bs=[r for r in baseline if r['case']==c['name'] and r['stage'].startswith('baseline')]
        ss=[r for r in scores if r['case']==c['name'] and r['comparable']]
        summary['per_scene'].append(dict(case=c['name'],candidates=len(ds),description_calls=len(ds),match_calls=len(ms),
            total_new_seconds=sum(d['seconds'] for d in ds)+sum(r['seconds'] for r in ms),
            baseline_requests=len(bs),baseline_total_seconds=sum(r['seconds'] for r in bs),
            description_tokens=sum(source[d['id']]['response'].get('usage',{}).get('completion_tokens',0) for d in ds if not source[d['id']].get('error')),
            correct=sum(r['correct'] for r in ss),baseline_correct=sum(r['baseline_correct'] for r in ss),targets=len(ss)))
    review_path=out/'semantic-review.json'
    if review_path.exists():
        review=json.loads(review_path.read_text());inputs=json.loads((out/'semantic-review-input.json').read_text())
        assert {(r['case'],r['candidate_id']) for r in review}=={(r['case'],r['candidate_id']) for r in inputs}
        accurate={(r['case'],r['candidate_id']) for r in review if r['accurate_appearance']}
        summary['semantic_recall']={}
        for group in ['synthetic','real']:
            cs=[c for c in cases if (c['origin']=='synthetic')==(group=='synthetic')];classes=recognized=bound=0
            for c in cs:
                for k in c['classes']:
                    classes+=1
                    candidates=[o for o in c['candidates'] if any(eligible(o,g) for g in c['gold'] if g['id'] in k['members'])]
                    good=[o for o in candidates if (c['name'],o['id']) in accurate]
                    recognized+=bool(good)
                    bound+=any(dby[c['name'],o['id']]['valid'] and dby[c['name'],o['id']]['extent']=='whole' for o in good)
            summary['semantic_recall'][group]=dict(classes=classes,accurate_appearance=recognized,accurate_and_whole=bound)
    # Record classifier validity and server consistency independently of task accuracy.
    summary['verification']=dict(description_requests=len(baseline),match_requests=len(match),
        match_valid=sum(r['answer'].strip() in ['1','2','8'] and not r.get('error') for r in match),
        description_settings_unchanged=json.loads((out/'description-server-before.json').read_text())==json.loads((out/'description-server-after.json').read_text()),
        match_settings_unchanged=json.loads((out/'match-server-before.json').read_text())==json.loads((out/'match-server-after.json').read_text()),
        phases_same_settings=json.loads((out/'description-server-before.json').read_text())==json.loads((out/'match-server-before.json').read_text()),
        source_hashes=plan['sources'])
    save(out/'summary-reviewed.json',summary)
    return cases,desc,scores,summary


def main(overview,local,out):
    out.mkdir(parents=True,exist_ok=True)
    packs=[result(p) for p in (overview,local)]
    summary={name:pack[-1] for name,pack in zip(['overview','local_only'],packs)}
    save(out/'summary.json',summary)
    page=['<!doctype html><meta charset="utf-8"><title>Region descriptions comparison</title>',
        '<style>body{font:16px sans-serif;margin:24px;max-width:1500px}table{border-collapse:collapse}td,th{border:1px solid #aaa;padding:8px;vertical-align:top}img{max-width:460px}pre{white-space:pre-wrap}.bad{background:#fee}.ok{background:#dfd}</style>',
        '<h1>候補→外観記述→位置による選択</h1><p>左：全体図＋候補＋近傍。右：候補＋近傍のみ（探索的な追加比較）。候補IDはホストで保持。位置条件は手作業で分解したもの。</p>',
        '<table><tr><th>条件</th><th>合成</th><th>実画面</th><th>新方式の呼出し数</th><th>新方式合計秒</th></tr>']
    for name,pack in zip(['overview','local_only'],packs):
        s=pack[-1];a=s['primary']['synthetic'];b=s['primary']['real']
        page.append(f'<tr><td>{name}</td><td>{a["correct"]}/{a["total"]}</td><td>{b["correct"]}/{b["total"]}</td><td>{s["calls"]["description"]+s["calls"]["matching"]}</td><td>{s["timing"]["description_total_seconds"]+s["timing"]["matching_total_seconds"]:.2f}</td></tr>')
    page+=['</table><h2>正解対象を含む候補の記述</h2><p>意味の採点はアシスタントによるレビュー。whole以外の判定は後段で保留。以下は診断用の注釈対象に対応する候補で、推論自体は全192候補へ実行した。</p>']
    inputs=json.loads((overview/'semantic-review-input.json').read_text())
    bydesc=[{(d['case'],d['candidate_id']):d for d in pack[1]} for pack in packs]
    for item in sorted(inputs,key=lambda r:(r['case'],r['candidate_id'])):
        key=item['case'],item['candidate_id'];page.append('<h3>'+html.escape(' / '.join(key))+'</h3><table><tr>')
        for root,records in zip([overview,local],bydesc):
            d=records[key];rel=Path('..')/root.name/'cards'/f'{key[0]}-{key[1]}.png'
            page.append('<td><img src="'+str(rel)+'"><p><b>'+html.escape(d['extent'])+'</b></p><p>'+html.escape(d['appearance'])+'</p></td>')
        page.append('</tr></table>')
    page+=['<h2>個体選択：同じ36対象</h2><table><tr><th>対象</th><th>再測定した従来方式</th><th>全体図あり</th><th>全体図なし</th></tr>']
    b2={(s['case'],s['gold_id']):s for s in packs[1][2]}
    for row in packs[0][2]:
        if not row['comparable']:continue
        other=b2[row['case'],row['gold_id']]
        page.append('<tr><td>'+row['case']+' / '+row['gold_id']+'</td>')
        for ok,text in [(row['baseline_correct'],row['baseline_selected']),
            (row['correct'],str(row['selected'])+' '+row['reason']),(other['correct'],str(other['selected'])+' '+other['reason'])]:
            page.append('<td class="'+('ok' if ok else 'bad')+'">'+html.escape(text)+'</td>')
        page.append('</tr>')
    page+=['</table><h2>詳細集計</h2><pre>'+html.escape(json.dumps(summary,ensure_ascii=False,indent=2))+'</pre>']
    (out/'gallery.html').write_text('\n'.join(page))
    print(json.dumps({k:dict(primary=v['primary'],stage_recall=v['stage_recall'],semantic_recall=v.get('semantic_recall'),calls=v['calls']) for k,v in summary.items()},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--overview',type=Path,required=True);p.add_argument('--local',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();main(a.overview,a.local,a.out)
