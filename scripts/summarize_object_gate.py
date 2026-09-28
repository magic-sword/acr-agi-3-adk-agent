"""Object-level detection confusion counts plus semantically reviewed gated outcomes."""
import argparse
from collections import Counter
import json
from pathlib import Path
import statistics
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.benchmark_parallel import read_lines
from scripts.benchmark_recognition import save_json


def counts(pairs):
    c=Counter()
    for truth,pred in pairs:
        if pred is None:
            c['invalid_positive' if truth else 'invalid_negative']+=1
            if truth:c['fn']+=1  # Failure to deliver a usable positive is a miss.
            continue
        c['tp' if truth and pred else 'fn' if truth else 'fp' if pred else 'tn']+=1
    return {k:c[k] for k in ['tp','fn','fp','tn','invalid_positive','invalid_negative']}


def summarize(out):
    cats=json.loads((out/'catalogs.json').read_text());cases={c['name']:c for c in json.loads((out/'cases.json').read_text())}
    rows=[r for r in read_lines(out/'measurements.jsonl') if not r['warmup']]
    gates=json.loads((out/'gates.json').read_text());explain={r['id']:r for r in read_lines(out/'explanations.jsonl')}
    reviews={r['key']:r for r in json.loads((out/'assistant-review.json').read_text())['rows']}
    expected={'free:'+str(r['id']) for r in rows if r['arm']=='free'}|{'explain:'+str(i) for i in explain}
    assert set(reviews)==expected
    good=lambda r:r['valid'] and r['within_budget']
    results=[];case_rows=[]
    for arm in ['free','joint','individual']:
        leaf_pairs=[];region_pairs=[];outside_pairs=[];mover_tp=0;mover_n=0;missing=0;latencies=[];detect_seconds=[];leaf_correct=[];foreground_by_case={}
        for cat in cats:
            if arm=='free':
                r=next(r for r in rows if r['arm']==arm and r['catalog_key']==cat['key']);rev=reviews['free:'+str(r['id'])]
                pred={o['id']:('Y' if o['id'] in rev['changed_ids'] else 'N') for o in cat['items']};pred['X']='Y' if rev['outside'] else 'N'
                valid=good(r);correct=valid and rev['primary'];seconds=r['seconds'];first_seconds=seconds;called=1
            else:
                g=next(g for g in gates if g['arm']==arm and g['catalog_key']==cat['key']);pred=g['answers'];valid=g['valid'];first_seconds=g['seconds'];seconds=first_seconds;called=0
                if g['explanation_id'] is None:correct=valid and cases[cat['case']]['truth']['category']=='static'
                else:
                    e=explain[g['explanation_id']];rev=reviews['explain:'+str(e['id'])];correct=valid and good(e) and rev['primary'];seconds+=e['seconds'];called=1
            for k,truth in cat['truth'].items():
                value=pred.get(k)=='Y' if valid else None
                pair=(truth=='Y',value)
                if k=='X':outside_pairs.append(pair)
                elif k in cat['leaf_ids']:
                    leaf_pairs.append(pair);foreground_by_case.setdefault(cat['case'],[]).append(pair)
                else:region_pairs.append(pair)
            for k in cat['mover_ids']:
                if k is None:missing+=1
                else:mover_n+=1;mover_tp+=int(valid and pred.get(k)=='Y')
            latencies.append(seconds);detect_seconds.append(first_seconds)
            exact=valid and all(pred.get(k)==cat['truth'][k] for k in cat['leaf_ids'])
            leaf_correct.append(exact)
            case_rows.append({'arm':arm,'case':cat['case'],'repeat':cat['repeat'],'primary':bool(correct),'leaf_exact':bool(exact),'seconds':seconds,'first_stage_seconds':first_seconds,'explanation_called':called})
        selected=[r for r in case_rows if r['arm']==arm];categories={}
        for catname in ['motion','multi_motion','deformation','split','static','appearance']:
            rs=[r for r in selected if cases[r['case']]['truth']['category']==catname]
            categories[catname]={'correct':sum(r['primary'] for r in rs),'n':len(rs)}
        results.append({'arm':arm,'foreground':counts(leaf_pairs),'foreground_by_case':{k:counts(v) for k,v in foreground_by_case.items()},'other_entries':counts(region_pairs),'unlisted':counts(outside_pairs),
            'listed_movers':{'detected':mover_tp,'n':mover_n,'unlisted_movers':missing},'leaf_exact_cases':sum(leaf_correct),
            'end_to_end':categories,'median_first_stage_seconds':statistics.median(detect_seconds),'median_total_seconds':statistics.median(latencies),
            'mean_total_seconds':statistics.mean(latencies),'total_seconds':sum(latencies),
            'explanation_calls':sum(r['explanation_called'] for r in selected)})
    allrows=read_lines(out/'measurements.jsonl')+list(explain.values())
    result={'arms':results,'by_case':case_rows,'measured_requests_excluding_preflight':len(allrows),'valid':sum(good(r) for r in allrows),
        'max_seconds':max(r['seconds'] for r in allrows),'no_image_answers':[r.get('answer') for r in rows if r['arm']=='control'],
        'inventory':'Frozen previous static inventories reused; acquisition time excluded.',
        'single_completion_tokens':sorted({r['response']['usage']['completion_tokens'] for r in rows if r['arm']=='individual' and r.get('response')})}
    save_json(out/'summary.json',result);print(json.dumps({k:v for k,v in result.items() if k!='by_case'},ensure_ascii=False,indent=2))


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();summarize(a.output)
