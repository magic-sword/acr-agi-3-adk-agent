"""Exploratory host grouping of observed response signatures; never fuse IDs.

Only comparable independent trials with exactly equal BEFORE grids are grouped.
No game role or causal claim is inferred. Unknown correspondence remains unknown.
"""
from collections import defaultdict
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.benchmark_delta_recognition import verify,save,sha


def response_groups(trials,records):
    if len(trials)<2 or any(t['before']!=trials[0]['before'] for t in trials):
        return dict(status='incomparable_or_single_trial',groups=[])
    signatures=[];groups=[]
    for ident,box,colors in records[0]['before']['regions']:
        x,y,r,b=box;changed=[];translations=[]
        for record in records:
            changed.append(any(y<=row<=b and max(x,left)<=min(r,right)
                               for row,left,right,old,new in record['delta']['runs']))
            links=[v for v in record['tracking']['links'] if v['before']==box]
            translations.append(tuple(links[0]['delta_xy']) if len(links)==1 else None)
        signatures.append(dict(candidate_id=ident,bbox=box,colors=colors,changed=changed,translations=translations))
    for kind,key in [('co_response','changed'),('same_translation','translations')]:
        buckets=defaultdict(list)
        for s in signatures:
            values=s[key]
            if kind=='co_response' and not any(values):continue
            if kind=='same_translation' and (None in values or not any(v!=(0,0) for v in values)):continue
            buckets[tuple(values)].append(s['candidate_id'])
        for signature,ids in buckets.items():
            if len(ids)>1:groups.append(dict(kind=kind,members=ids,signature=signature,
                evidence_count=len(trials),claim='Observed matching response, not physical identity or causation.'))
    return dict(status='comparable',actions=[t['action'] for t in trials],signatures=signatures,groups=groups)


def main(source):
    cases,_=verify(source);e={(r['case'],r['trial']):r for r in json.loads((source/'evidence.json').read_text())}
    outputs=[];scores=[]
    for c in cases:
        start=time.perf_counter();result=response_groups(c['trials'],[e[c['name'],t['id']] for t in c['trials']]);elapsed=time.perf_counter()-start
        outputs.append(dict(case=c['name'],seconds=elapsed,**result))
        if c['group'] not in ['coupled_color','coupled_translation']:continue
        # Scoring only: map candidates back to the color-pair answer options.
        target_kind='co_response' if c['group']=='coupled_color' else 'same_translation'
        names={'9':'blue','12':'orange','14':'green'};byid={s['candidate_id']:s for s in result['signatures']}
        pairs=[]
        for g in result['groups']:
            if g['kind']!=target_kind:continue
            colors=[]
            for ident in g['members']:
                keys=list(byid[ident]['colors']);colors.append(names.get(keys[0]) if len(keys)==1 else None)
            if len(colors)==2 and None not in colors:pairs.append(frozenset(colors))
            else:pairs.append(frozenset(['unscorable_region_group']))
        truth=next(o['text'] for o in c['options'] if o['letter']==c['truth'])
        expected=[] if truth=='none of these pairs' else [frozenset(truth.split(' and '))]
        scores.append(dict(case=c['name'],kind=target_kind,correct=pairs==expected,
                           predicted_pairs=[sorted(p) for p in pairs],expected_pairs=[sorted(p) for p in expected],seconds=elapsed))
    report=dict(design='Exploratory analysis after seeing VLM results. Uses all static candidates, no gold filtering. Same BEFORE grid requirement supplies cross-trial alignment. Co-response uses raw delta intersection only. Translation uses existing tracker links. Scores are reused synthetic diagnostics, not held-out validation.',
        source=str(source),code_sha256=sha(Path(__file__)),outputs=outputs,scores=scores,
        summary={kind:dict(correct=sum(s['correct'] for s in scores if s['kind']==kind),
                           total=sum(s['kind']==kind for s in scores)) for kind in ['co_response','same_translation']})
    save(source/'host-response-groups.json',report);print(json.dumps(report['summary'],indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);a=p.parse_args();main(a.source)
