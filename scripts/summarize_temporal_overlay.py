"""Structured recognition scores with a separate arm-hidden semantic audit."""
import argparse
from collections import Counter
import json
from pathlib import Path
import statistics
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.benchmark_parallel import digest,read_lines
from scripts.benchmark_recognition import save_json

def audit_key(r):
    return digest({'case':r['case'],'parsed':r.get('parsed'),'answer_if_unparsed':r.get('answer') if 'parsed' not in r else None})

def review(out):
    groups={}
    for r in read_lines(out/'measurements.jsonl'):
        if r['warmup']:continue
        key=audit_key(r)
        groups.setdefault(key,{'key':key,'case':r['case'],'observations':r.get('parsed',{}).get('observations',[]),'answer_if_unparsed':r.get('answer') if 'parsed' not in r else None})
    save_json(out/'audit-groups.json',list(groups.values()));print('Unique semantic outputs:',len(groups))

def summarize(out):
    rows=[r for r in read_lines(out/'measurements.jsonl') if not r['warmup']]
    cats={c['key']:c for c in json.loads((out/'catalogs.json').read_text())}
    audit={r['key']:r for r in json.loads((out/'semantic-review.json').read_text())['rows']}
    assert set(audit)=={audit_key(r) for r in rows}
    arms=[];detail=[]
    for arm in json.loads((out/'plan.json').read_text()).get('arms',['repeat','blend','outline','split']):
        rs=[r for r in rows if r['arm']==arm];cnt=Counter();lat=[];cases={}
        for r in rs:
            cat=cats[r['catalog_key']];valid=r['valid'] and r['within_budget'];a=audit[audit_key(r)]
            pred={x['id']:x for x in r.get('parsed',{}).get('observations',[])}
            rejected=set(a.get('rejected_ids',[]))
            def kind(k):return pred.get(k,{}).get('kind','uncertain') if valid else 'uncertain'
            def correct(k):return valid and (kind(k)==cat['truth'][k] or k in a.get('recovered_ids',[])) and k not in rejected
            positive=[k for k in cat['leaf_ids'] if cat['truth'][k]!='unchanged']
            negative=[k for k in cat['leaf_ids'] if cat['truth'][k]=='unchanged']
            cnt['changed_targets']+=len(positive);cnt['changed_detected']+=sum(kind(k) not in ['unchanged','uncertain'] for k in positive)
            cnt['unchanged_targets']+=len(negative);cnt['unchanged_evaluable']+=len(negative) if valid else 0
            cnt['false_positive_targets']+=sum(kind(k) not in ['unchanged','uncertain'] for k in negative)
            cnt['nonmoving_targets_called_moved']+=sum(kind(k)=='moved' for k in cat['leaf_ids'] if cat['truth'][k]!='moved')
            cnt['unlisted_false_positives']+=int(cat['truth']['X']=='unchanged' and kind('X') not in ['unchanged','uncertain'])
            cnt['moving_targets']+=len(cat['mover_ids']);cnt['moving_targets_correct']+=sum(correct(k) for k in cat['mover_ids'])
            cnt['moving_targets_labeled']+=sum(kind(k)=='moved' for k in cat['mover_ids'])
            cnt['valid']+=int(valid);cnt['artifact_hallucination_answers']+=int(a.get('artifact_hallucination',False))
            if cat['mover_ids']:
                primary=valid and all(correct(k) for k in cat['mover_ids']);category='motion'
            elif r['case']=='synthetic_static':
                primary=valid and all(kind(k)=='unchanged' for k in cat['leaf_ids']+['X']);category='static'
            else:
                targets=positive+(['X'] if cat['truth']['X']!='unchanged' else [])
                primary=valid and all(correct(k) for k in targets);category=r['case']
            unchanged_checks=negative+(['X'] if cat['truth']['X']=='unchanged' else [])
            coherent=primary and not a.get('contradiction',False) and not a.get('artifact_hallucination',False) and all(kind(k)=='unchanged' for k in unchanged_checks)
            cases.setdefault(r['case'],{'correct':0,'coherent':0,'n':0});cases[r['case']]['correct']+=int(primary);cases[r['case']]['coherent']+=int(coherent);cases[r['case']]['n']+=1
            cnt[category+'_n']+=1;cnt[category+'_correct']+=int(primary);cnt[category+'_coherent']+=int(coherent)
            lat.append(r['seconds']);detail.append({'id':r['id'],'case':r['case'],'arm':arm,'repeat':r['repeat'],'primary':bool(primary),'coherent':bool(coherent),'valid':bool(valid),'audit_key':audit_key(r)})
        arms.append({'arm':arm,'counts':dict(cnt),'by_case':cases,'median_seconds':statistics.median(lat),'mean_seconds':statistics.mean(lat)})
    result={'arms':arms,'rows':detail,'main_requests':len(rows),'warmups':4,'semantic_groups':len(audit),'all_seconds':sum(r['seconds'] for r in read_lines(out/'measurements.jsonl'))}
    save_json(out/'summary.json',result);print(json.dumps({k:v for k,v in result.items() if k!='rows'},ensure_ascii=False,indent=2))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('command',choices=['review','summarize']);ap.add_argument('--output',required=True,type=Path);a=ap.parse_args()
    if a.command=='review':review(a.output)
    else:summarize(a.output)
