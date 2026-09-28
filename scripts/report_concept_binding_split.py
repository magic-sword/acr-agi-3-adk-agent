"""Reviewed summary, respecting the documented pre-scoring input correction."""
import argparse
from collections import Counter
import html
import json
from pathlib import Path
import statistics
import sys

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.benchmark_concept_binding_split import save, sha, verify


def summarize(rows):
    positives=[r for r in rows if not r['control']]
    controls=[r for r in rows if r['control']]
    return dict(correct=sum(r['correct'] for r in positives),total=len(positives),
        candidates_available=sum(r['available'] for r in positives),
        false_selection=sum(not r['correct'] and r['answer']!='X' for r in positives),
        false_abstention=sum(not r['correct'] and r['answer']=='X' for r in positives),
        control_correct=sum(r['correct'] for r in controls),controls=len(controls),
        valid=sum(r['valid'] for r in rows),requests=len(rows),
        median_seconds=statistics.median(r['seconds'] for r in rows))


def report(out, supplement):
    plan,cases,jobs=verify(out); _,_,extra_jobs=verify(supplement)
    rows=json.loads((out/'binding-scores.json').read_text())
    amendment=json.loads((out/'protocol-amendment.json').read_text())
    excluded=set(amendment['excluded_original_job_ids'])
    scores=[r for r in rows if r['id'] not in excluded]
    original=json.loads((out/'summary.json').read_text()); bycase={c['name']:c for c in cases}
    reviews=json.loads((out/'concept-review.json').read_text())
    responses=list(map(json.loads,(out/'responses.jsonl').read_text().splitlines()))
    byresponse={r['id']:r for r in responses}; byjob={j['id']:j for j in jobs}
    # Independently recompute region eligibility from raw annotations and answers.
    checked=0
    for r in scores:
        j=byjob[r['id']];c=bycase[r['case']]
        assert r['answer']==byresponse[r['id']]['answer'].strip()
        if not r['control']:
            g=next(g for g in c['gold'] if g['id']==r['gold_id']);gx,gy,gr,gb=g['bbox']
            gold_area=(gr-gx+1)*(gb-gy+1); accepted=[]
            for o in c['candidates']:
                x,y,rr,b=o['bbox']; area=(rr-x+1)*(b-y+1)
                overlap=max(0,min(gr,rr)-max(gx,x)+1)*max(0,min(gb,b)-max(gy,y)+1)
                support=sum(x<=px<=rr and y<=py<=b for px,py in g['support'])
                if support*10>=len(g['support'])*9 and area<=gold_area*2 and overlap*2>=area+gold_area-overlap:
                    accepted.append(o['id'])
            assert accepted==j['acceptable']==r['acceptable']
            assert r['correct']==(r['answer'] in accepted if accepted else r['answer']=='X')
        else: assert r['correct']==(r['answer']=='X')
        checked+=1
    summary=dict(discovery=original['discovery'],discovery_format=original['discovery_format'],
        primary={},second_permutation={},excluded_original_jobs=sorted(excluded),correction=[],
        interpretation='Class discovery and oracle-query binding have different denominators. Binding does not use generated concepts. Exact current-runtime accuracy and end-to-end improvement were not measured.')
    for index,key in [(0,'primary'),(1,'second_permutation')]:
        for group in ['all','synthetic','real','ls20','vc33','ft09']:
            rr=[r for r in scores if r['permutation']==index and
                (group=='all' or r['origin']==group or group=='real' and r['origin']!='synthetic')]
            summary[key][group]=summarize(rr)
    pairs={}
    for r in scores:pairs.setdefault((r['case'],r['gold_id']),[]).append(r)
    summary['stability']=dict(pairs=len(pairs),both_correct=sum(all(r['correct'] for r in rr) for rr in pairs.values()),
        correctness_disagreements=sum(len({r['correct'] for r in rr})>1 for rr in pairs.values()),
        answer_disagreements=sum(len({r['answer'] for r in rr})>1 for rr in pairs.values()))
    extra=list(map(json.loads,(supplement/'responses.jsonl').read_text().splitlines()))
    assert Counter(r['id'] for r in extra)==Counter(j['id'] for j in extra_jobs)
    for r in extra:
        j=next(j for j in extra_jobs if j['id']==r['id'])
        summary['correction'].append(dict(id=r['id'],permutation=j['permutation'],answer=r['answer'],
            acceptable=j['acceptable'],correct=r['answer'].strip() in j['acceptable']))
    summary['by_scene']=[]
    for c in cases:
        rr=[r for r in reviews if r['case']==c['name']]
        summary['by_scene'].append(dict(case=c['name'],candidate_count=len(c['candidates']),
            classes=len(c['classes']),discovery={r['arm']:len(r['matches']) for r in rr},
            binding=summarize([r for r in scores if r['case']==c['name'] and r['permutation']==0])))
    save(out/'summary-reviewed.json',summary)
    save(out/'verification.json',dict(frozen_inputs_verified=True,independent_binding_checks=checked,
        all_original_responses=len(responses),supplement_responses=len(extra),
        server_props_unchanged=json.loads((out/'server-before.json').read_text())==json.loads((out/'server-after.json').read_text()),
        server_props_same_in_supplement=json.loads((out/'server-before.json').read_text())==json.loads((supplement/'server-before.json').read_text()),
        concept_review_sha256=sha(out/'concept-review.json'),amendment_sha256=sha(out/'protocol-amendment.json'),
        report_source_sha256=sha(Path(__file__)),review_note='Semantic review by implementing assistant, not independent human.'))
    page=['<!doctype html><meta charset="utf-8"><title>Reviewed split evaluation</title>',
        '<style>body{font:16px sans-serif;max-width:1250px;margin:auto}pre{white-space:pre-wrap}img{max-width:380px}section{display:inline-block;vertical-align:top;width:400px;margin:7px}td,th{border:1px solid #aaa;padding:8px}table{border-collapse:collapse}</style>',
        '<h1>概念生成と候補紐づけの分離評価</h1><p>候補不足と紐づけ誤りを分離。概念は種類、紐づけは個体を採点。実画面は部分注釈。</p>',
        '<p>ft09の赤い中心を赤い枠と記した1問は主集計から除外。修正版2要求は追加測定。<a href="protocol-amendment.json">修正記録</a></p>',
        '<p><a href="gallery.html">全概念出力・元の紐づけ採点（修正前を含む）</a> / <a href="concept-review.json">概念の採点理由</a> / <a href="summary-reviewed.json">修正反映済み集計</a></p>',
        '<table><tr><th>画面</th><th>概念のみ4件</th><th>概念のみ12件</th><th>正解記述からの紐づけ</th></tr>']
    for row in summary['by_scene']:
        d=row['discovery']; b=row['binding']
        page.append(f'<tr><td>{row["case"]}</td><td>{d["concept_only_4"]}/{row["classes"]}</td><td>{d["concept_only_12"]}/{row["classes"]}</td><td>{b["correct"]}/{b["total"]}</td></tr>')
    page+=['</table><h2>紐づけの誤答例（主集計）</h2><p>水色の枠：注釈対象。橙の枠：モデルが選んだ候補。Xは保留。</p>']
    for r in scores:
        if r['permutation'] or r['control'] or r['correct']:continue
        c=bycase[r['case']];g=next(g for g in c['gold'] if g['id']==r['gold_id'])
        candidate=next((o for o in c['candidates'] if o['id']==r['answer']),None)
        im=Image.open(out/(c['name']+'.png')).convert('RGB');draw=ImageDraw.Draw(im)
        for box,color in [(g['bbox'],'cyan')]+([(candidate['bbox'],'orange')] if candidate else []):
            x,y,rr,b=box;draw.rectangle((24+x*8-2,24+y*8-2,24+(rr+1)*8+1,24+(b+1)*8+1),outline=color,width=2)
        name=f'error-{r["id"]}.png';im.save(out/name)
        query=byjob[r['id']]['payload']['messages'][1]['content'][1]['text'].split('\n')[0]
        page.append('<section><b>'+c['name']+' / '+r['gold_id']+'</b><br><img src="'+name+'"><p>'+html.escape(query)+'</p><p>選択：'+r['answer']+' / 許容候補：'+', '.join(r['acceptable'])+'</p></section>')
    page+=['<h2>集計詳細</h2><pre>'+html.escape(json.dumps(summary,ensure_ascii=False,indent=2))+'</pre>']
    (out/'reviewed-gallery.html').write_text('\n'.join(page))
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--supplement',type=Path,required=True)
    a=p.parse_args();report(a.out,a.supplement)
