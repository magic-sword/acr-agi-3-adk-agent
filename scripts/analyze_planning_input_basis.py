"""Audit input reconstruction and score only the synthetic tasks with known answers."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re

import jsonschema

from probe_planning_input_basis import build
from probe_repeated_click_inputs import digest


def selected_button(answer):
    for key in ('target','action','next_step'):
        words=set(re.findall(r'\b(LEFT|RIGHT)\b', str(answer.get(key,'')), re.I))
        words={w.upper() for w in words}
        if len(words)==1:
            return next(iter(words))
        if len(words)>1:
            return None
    return None


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('output',type=Path)
    args=p.parse_args();out=args.output.resolve()
    _,expected=build()
    plan=json.loads((out/'plan.json').read_text())
    rows=json.loads((out/'results.json').read_text())
    assert len(expected)==plan['requests']==80
    records=[]
    for i,row in enumerate(rows):
        job=expected[i]
        assert all(row[k]==job[k] for k in ('case','variant','repetition','family','expected_status','expected_target'))
        payload=json.loads((out/row['request_file']).read_text())
        assert digest(payload)==row['request_sha256']==digest(job['payload'])==plan['jobs'][i]['request_sha256']
        answer=row.get('answer')
        errors=[]
        if isinstance(answer,dict):
            errors=[e.message for e in jsonschema.Draft202012Validator(
                payload['tools'][0]['function']['parameters']).iter_errors(answer)]
        else:
            errors=[row.get('error','Missing structured answer')];answer={}
        if row.get('tool')!='submit_next_step':errors.append('Missing/wrong tool name')
        button=selected_button(answer)
        correct=None
        if row['expected_status'] is not None:
            correct=(not errors and answer.get('status')==row['expected_status'] and
                     (row['expected_target'] is None or button==row['expected_target']))
        record={k:row[k] for k in ('case','variant','repetition','family','expected_status','expected_target','request_file')}
        record.update(schema_errors=errors,status=answer.get('status'),selected_button=button,
                      correct=correct,answer=answer,
                      target_choice_correct=(button==row['expected_target'] if row['expected_target'] is not None else None),
                      prompt_tokens=(row.get('response') or {}).get('usage',{}).get('prompt_tokens'),
                      finish_reason=((row.get('response') or {}).get('choices') or [{}])[0].get('finish_reason'))
        records.append(record)
    groups=defaultdict(list)
    for r in records:
        group=r['variant']
        if r['family']=='calibration' and r['variant']=='full':
            group='prerequisite' if r['case'].startswith('prerequisite') else 'lamp_achieved' if r['case'].endswith('achieved') else 'lamp_pending'
        groups[r['family'],group].append(r)
    aggregates=[]
    for (family,variant),selected in sorted(groups.items()):
        scored=[r for r in selected if r['correct'] is not None]
        target_scored=[r for r in selected if r['target_choice_correct'] is not None]
        aggregates.append(dict(family=family,variant=variant,n=len(selected),
                               statuses=dict(Counter(r['status'] for r in selected)),
                               correct=sum(r['correct'] for r in scored),scored=len(scored),
                               target_choice_correct=sum(r['target_choice_correct'] for r in target_scored),
                               target_choices_scored=len(target_scored)))
    result=dict(complete=len(rows)==len(expected),requests=len(rows),expected=len(expected),
                request_hashes_verified=len(rows),schema_valid=sum(not r['schema_errors'] for r in records),
                aggregates=aggregates,records=records,
                interpretation='Only supplied synthetic rules establish gold choices. Real-board actions need semantic '
                               'audit; changed text/status/target is not proof of useful or optimal planning.')
    (out/'audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    lines=['# Planning input basis: responses','','Only synthetic known-answer tasks are scored as correct/incorrect.','']
    for family in ('calibration','real'):
        for r in sorted((r for r in records if r['family']==family),key=lambda r:(r['case'],r['variant'],r['repetition'])):
            lines.extend([f"## {r['case']} / {r['variant']} / r{r['repetition']}",'',
                          f"Expected: {r['expected_status']} {r['expected_target']}; correct: {r['correct']}",'',
                          json.dumps(r['answer'],ensure_ascii=False),''])
    (out/'responses.md').write_text('\n'.join(lines))
    print(json.dumps({k:v for k,v in result.items() if k!='records'},ensure_ascii=False))


if __name__=='__main__':main()
