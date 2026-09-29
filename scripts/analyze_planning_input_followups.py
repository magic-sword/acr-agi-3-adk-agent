"""Audit exploratory goal/action-option probes; report choice separately from status."""
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path
import re

import jsonschema

from probe_repeated_click_inputs import digest
from probe_planning_input_basis import factual_history
import probe_planning_input_goal
import probe_planning_action_options
import probe_planning_minimal_choice
import probe_planning_repeat_goal
import probe_planning_action_table


def normal(text):
    text=re.sub(r'\s+',' ',str(text).lower()).strip(' .')
    return re.sub(r'^the ','',text)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('kind',choices=('goal','options','minimal','repeat','table'))
    p.add_argument('output',type=Path)
    args=p.parse_args();out=args.output.resolve()
    module={'goal':probe_planning_input_goal,'options':probe_planning_action_options,
            'minimal':probe_planning_minimal_choice,'repeat':probe_planning_repeat_goal,
            'table':probe_planning_action_table}[args.kind]
    _,jobs=module.build()
    plan=json.loads((out/'plan.json').read_text())
    rows=json.loads((out/'results.json').read_text());records=[]
    for i,row in enumerate(rows):
        job=jobs[i];request=json.loads((out/row['request_file']).read_text())
        assert digest(request)==digest(job['payload'])==row['request_sha256']==plan['jobs'][i]['request_sha256']
        assert all(row[k]==v for k,v in job.items() if k!='payload')
        answer=row.get('answer',{})
        errors=[e.message for e in jsonschema.Draft202012Validator(request['tools'][0]['function']['parameters']).iter_errors(answer)]
        if row.get('tool')!='submit_next_step':errors.append('Missing/wrong tool')
        assert isinstance(answer,dict)
        target=normal(answer.get('target',''))
        tested=job.get('tested_target')
        if args.kind=='goal':
            text=factual_history(job['case'])[0]
            tested=text.split('CLICK was sent at ',1)[1].split(' and acknowledged.',1)[0]
        untested=job.get('untested_targets',[])
        resolved_action_id=None
        if args.kind=='table':
            context=json.loads(request['messages'][0]['content'][1]['text'])
            last=context['last_completed_action_id']
            tested=job['action_targets'][last]
            untested=[t for k,t in job['action_targets'].items() if k!=last]
            for identity,description in job['action_targets'].items():
                if target==normal(description):resolved_action_id=identity
            if target in ('same target','same target as last completed action'):
                resolved_action_id=last
            for field in ('target','action','next_step'):
                found=re.findall(r'\baction ([AB])\b',answer.get(field,''))
                if len(set(found))==1:
                    resolved_action_id=found[0];break
        choice=('tested' if tested and target==normal(tested) else
                'untested_option' if target in [normal(t) for t in untested] else
                'none' if target in ('','none') else 'other_or_partial')
        if resolved_action_id is not None:
            choice='tested' if resolved_action_id==last else 'untested_option'
        region=lambda s: re.findall(r'\b(?:upper|lower)-(?:left|right)\b',str(s))
        old_regions=region(tested or '')
        new_regions=region(answer.get('target',''))
        same_arrangement=bool(old_regions and new_regions and old_regions==new_regions)
        records.append(dict(case=row['case'],variant=row['variant'],repetition=row['repetition'],
                            order=row.get('option_order'),choice=choice,status=answer.get('status'),
                            same_arrangement=same_arrangement,mentions_specific_arrangement=bool(new_regions),
                            resolved_action_id=resolved_action_id,
                            schema_errors=errors,answer=answer,request_file=row['request_file'],
                            prompt_tokens=row.get('response',{}).get('usage',{}).get('prompt_tokens'),
                            finish_reason=(row.get('response',{}).get('choices') or [{}])[0].get('finish_reason')))
    groups=defaultdict(list)
    for r in records:groups[r['variant']].append(r)
    summary={variant:dict(n=len(rs),choices=dict(Counter(r['choice'] for r in rs)),
                          statuses=dict(Counter(r['status'] for r in rs)),
                          same_arrangement=sum(r['same_arrangement'] for r in rs),
                          specific_arrangement=sum(r['mentions_specific_arrangement'] for r in rs))
             for variant,rs in sorted(groups.items())}
    result=dict(complete=len(rows)==len(jobs),requests=len(rows),expected=len(jobs),
                request_hashes_verified=len(rows),schema_valid=sum(not r['schema_errors'] for r in records),
                summary=summary,records=records,
                interpretation='Choice is literal normalized target match, separate from status. '
                               'Unknown/partial matches require manual review; no game-success score.')
    (out/'audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    lines=['# Input follow-up responses','']
    for r in sorted(records,key=lambda r:(r['variant'],r['case'],r['repetition'])):
        lines += [f"## {r['variant']} / {r['case']} / {r['repetition']}",'',
                  f"Choice: {r['choice']}; status: {r['status']}",'',json.dumps(r['answer'],ensure_ascii=False),'']
    (out/'responses.md').write_text('\n'.join(lines))
    print(json.dumps({k:v for k,v in result.items() if k!='records'},ensure_ascii=False))


if __name__=='__main__':main()
