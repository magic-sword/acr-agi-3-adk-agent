"""Candidate-count sweep with a unique goal-compatible target, all positions.

Synthetic trial facts, not new claims about ft09. Full production planning is
compared with selection-only output to avoid mistaking handoff burden for capacity.
"""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random

from probe_incremental_planning_inputs import execute
from probe_repeated_click_inputs import context_part

ROOT = Path(__file__).resolve().parents[1]
COUNTS = (2, 3, 4, 6, 8)
GOLD = 'the amber switch'
DISTRACTORS = ['the '+color+' switch' for color in ('red', 'blue', 'green', 'purple', 'cyan', 'white', 'gray')]


def build():
    log = next((ROOT/'outputs/focused-workflow-smoke-20260929-validated').glob('*/cognition/*.requests.jsonl'))
    original = next(r['request'] for r in map(json.loads,log.read_text().splitlines())
                    if r.get('work') == 'ground' and r['step'] == 1)
    jobs=[]
    for count in COUNTS:
        for position in range(count):
            targets=DISTRACTORS[:count-1]
            targets=targets[:position]+[GOLD]+targets[position:]
            candidates=[dict(id=f'c{i+1}',verb='activate',target_query=t,
                expected_effect='Observe whether activating this switch changes the board.',
                rationale='A visible switch that can be activated.',source='proposed',skill_name=None)
                for i,t in enumerate(targets)]
            facts=[dict(target_query=t,completed_activations=0 if t==GOLD else 1,
                        acknowledged=None if t==GOLD else True,
                        changed_cell_count=None if t==GOLD else 0)
                   for t in sorted(targets)]
            for mode in ('full_plan','selection_only'):
                payload=deepcopy(original)
                part,context=context_part(payload)
                context=dict(work='ground',observation_id='synthetic-count-test',
                    purpose=dict(observation_id='synthetic-count-test',
                        desired_state='Observe the effect of an activation not previously tested.',
                        target_query='A visible switch whose activation has not been tested.',intent='probe',
                        question='What happens when an untested switch is activated?',
                        expected_observation='One acknowledged activation followed by an observation.',
                        rationale='The next test must add an untested target to the completed tests.'),
                    candidates=candidates,
                    evidence=dict(current=dict(observation_id='synthetic-count-test',
                        scene='All offered switches are visible and can be activated with CLICK. These are complete trial records; '
                              'completed_activations=0 means untested, and 1 means already tested under these same conditions.'),
                        past_trials=facts,conditional_knowledge=[],available_actions=['CLICK']))
                part['text']=json.dumps(context,separators=(',',':'))
                schema=payload['tools'][0]['function']['parameters']
                for branch in schema['anyOf']:
                    branch['properties']['observation_id']['enum']=['synthetic-count-test']
                    if branch['properties']['candidate_id'].get('type')=='string':
                        branch['properties']['candidate_id']['enum']=[c['id'] for c in candidates]
                if mode=='selection_only':
                    payload['messages'][0]['content']=(
                        'Choose one offered candidate that satisfies purpose using evidence. '
                        'Return its candidate_id and a concise reason with submit_choice. Do not write an execution procedure.')
                    payload['tools']=[dict(type='function',function=dict(name='submit_choice',
                        description='Select the next semantic action.',parameters=dict(type='object',
                        additionalProperties=False,properties=dict(
                            candidate_id=dict(type='string',enum=[c['id'] for c in candidates]),
                            reason=dict(type='string',maxLength=240)),required=['candidate_id','reason'])))]
                gold_id=candidates[position]['id']
                assert next(f['completed_activations'] for f in facts if f['target_query']==GOLD)==0
                assert sum(f['completed_activations']==0 for f in facts)==1
                jobs.append(dict(case=f'n{count}-p{position+1}',variant=mode,repetition=0,
                    count=count,position=position+1,gold_id=gold_id,offered=deepcopy(candidates),payload=payload))
    random.Random(2026092909).shuffle(jobs)
    assert len(jobs)==46
    return jobs


def analyze(out):
    import jsonschema
    results=json.loads((out/'results.json').read_text())
    details=[]
    for r in results:
        request=json.loads((out/r['request_file']).read_text());answer=r.get('answer')
        try:
            jsonschema.validate(answer,request['tools'][0]['function']['parameters'])
            valid=r.get('tool')==request['tools'][0]['function']['name']
        except jsonschema.ValidationError:valid=False
        selected=answer.get('candidate_id') if isinstance(answer,dict) else None
        actions=[o['action'] for s in (answer.get('procedure') or {}).get('steps',[]) for o in s['options']] if valid else []
        details.append(dict(count=r['count'],position=r['position'],mode=r['variant'],valid=valid,
            correct=valid and selected==r['gold_id'],selected=selected,gold=r['gold_id'],
            first=valid and selected=='c1',reason=answer.get('reason') if isinstance(answer,dict) else None,
            actions=actions,request_file=r['request_file']))
    summary=[]
    for count in COUNTS:
        for mode in ('full_plan','selection_only'):
            rows=[r for r in details if r['count']==count and r['mode']==mode]
            nonfirst=[r for r in rows if r['position']!=1]
            summary.append(dict(count=count,mode=mode,n=len(rows),valid=sum(r['valid'] for r in rows),
                correct=sum(r['correct'] for r in rows),nonfirst_correct=sum(r['correct'] for r in nonfirst),
                nonfirst_n=len(nonfirst),wrong_first=sum(r['first'] for r in nonfirst),
                failed_positions=[r['position'] for r in rows if not r['correct']]))
    (out/'analysis.json').write_text(json.dumps(dict(summary=summary,details=details),ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--dry-run',action='store_true');p.add_argument('--analyze',action='store_true');a=p.parse_args()
    if a.analyze:return analyze(a.output)
    jobs=build()
    if a.dry_run:print('46 requests; unique-answer and position checks passed');return
    execute(jobs,a.output,dict(design='2/3/4/6/8 candidates, unique untested target in every possible position; '
        '23 contexts x full production grounding versus selection-only prompt/schema = 46 requests.',
        fixture='Synthetic complete factual trial counts, one untested amber switch; distractors have one acknowledged '
                'no-change activation in identical conditions. No images or real game actions.',
        fixed='Same goal, evidence format, target wording, model, temperature=0 and max_tokens; '
              'candidate and evidence count grow together. IDs assigned by presentation position.',
        limitations='Not an independent sample of games or a universal capacity bound. Count and input length covary. '
                    'Only one synthetic goal and target family. Selection-only changes instruction and schema together. '
                    'Above-three candidates are experimental; runtime provider limit remains three.',
        probe_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    analyze(a.output)


if __name__=='__main__':main()
