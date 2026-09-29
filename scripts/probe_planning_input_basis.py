"""Build inputs from zero: counterfactual capability controls and factual ARC probes.

No runtime, state-machine, acceptance, cursor, or game-rule changes.
"""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random

from probe_incremental_planning_inputs import execute, minimal_payload, read_cases
from probe_repeated_click_inputs import digest

ROOT = Path(__file__).resolve().parents[1]
QUESTION = 'What should you do next?'
TOOL = [{'type':'function', 'function':{
    'name':'submit_next_step', 'description':'Report your next step or missing information.',
    'parameters':{'type':'object', 'additionalProperties':False,
        'properties':{
            'status':{'type':'string', 'enum':['act','done','need_information']},
            'next_step':{'type':'string'},
            'action':{'type':'string', 'description':'Control to use, if acting.'},
            'target':{'type':'string', 'description':'Target in words, if acting; exact coordinates are unnecessary.'},
            'reason':{'type':'string', 'description':'A brief reason based on supplied information.'}},
        'required':['status','next_step']}}}]


def payload(original, fields, visual=None):
    content = [{'type':'text','text':QUESTION}]
    if fields:
        content.append({'type':'text','text':json.dumps(fields, ensure_ascii=False, separators=(',',':'))})
    if visual:
        content.extend([{'type':'text','text':'Current board'}, deepcopy(visual)])
    return dict(**minimal_payload(original), messages=[{'role':'user','content':content}],
                tools=deepcopy(TOOL), tool_choice='required', parallel_tool_calls=False)


def calibration_jobs(original):
    jobs = []
    for goal in ('ON','OFF'):
        for on_button in ('LEFT','RIGHT'):
            records = [dict(completed_test=i+1, initial_lamp='OFF', clicked_button=button,
                            observed_lamp='ON' if button==on_button else 'OFF')
                       for i,button in enumerate(('LEFT','RIGHT'))]
            for achieved in (False,True):
                state = goal if achieved else ('OFF' if goal=='ON' else 'ON')
                target = on_button if goal=='ON' else ('RIGHT' if on_button=='LEFT' else 'LEFT')
                fields = dict(
                    apparatus='Two buttons control one lamp. Each click deterministically sets the lamp state; '
                              'there are no hidden prerequisites, delays or cumulative effects. '
                              'The following are isolated completed tests, not current instructions.',
                    goal=f'Make the lamp {goal} using the fewest additional clicks.',
                    available_controls='CLICK the LEFT button or CLICK the RIGHT button.',
                    current_state={'lamp':state}, completed_tests=records)
                name=f'lamp-{goal}-on_{on_button}-'+('achieved' if achieved else 'pending')
                for rep in range(2):
                    jobs.append(dict(case=name,variant='full',repetition=rep,family='calibration',
                                     expected_status='done' if achieved else 'act',
                                     expected_target=None if achieved else target,
                                     payload=payload(original,fields)))
                if not achieved:
                    for key in ('goal','completed_tests','available_controls'):
                        reduced=deepcopy(fields);reduced.pop(key)
                        jobs.append(dict(case=name,variant='without_'+key,repetition=0,family='calibration',
                                         expected_status='act' if key=='available_controls' else None,
                                         expected_target=target if key=='available_controls' else None,
                                         payload=payload(original,reduced)))
    for open_button in ('LEFT','RIGHT'):
        start_button='RIGHT' if open_button=='LEFT' else 'LEFT'
        for cover in ('CLOSED','OPEN'):
            fields=dict(
                apparatus='A motor with a cover and two buttons; no hidden conditions or delayed effects.',
                goal='Make the motor RUNNING using the fewest additional clicks.',
                available_controls='CLICK the LEFT button or CLICK the RIGHT button.',
                observed_rules=[f'{open_button} opens the cover.',
                                f'{start_button} starts the motor only when the cover is OPEN; otherwise nothing changes.'],
                current_state=dict(cover=cover,motor='STOPPED'),
                completed_trial=(dict(clicked_button=start_button, observed_result='Cover stayed CLOSED; motor stayed STOPPED.')
                                 if cover=='CLOSED' else
                                 dict(clicked_button=open_button, observed_result='Cover became OPEN; motor stayed STOPPED.')))
            jobs.append(dict(case=f'prerequisite-open_{open_button}-{cover}',variant='full',repetition=0,
                             family='calibration',expected_status='act',
                             expected_target=open_button if cover=='CLOSED' else start_button,
                             payload=payload(original,fields)))
    assert len(jobs)==32
    return jobs


def factual_history(name):
    # Derived from recorded executed clicks, never from the model's target query.
    logs=next((ROOT/'outputs/memory-comparison-20260929'/name).glob('*/cognition'))
    observations=[json.loads(l) for l in next(logs.glob('*.observations.jsonl')).read_text().splitlines()]
    artifacts=[json.loads(l) for l in next(logs.glob('*.artifacts.jsonl')).read_text().splitlines()]
    feedback=[r['trial'] for r in artifacts if r['event']=='action_feedback' and r['step'] in (1,2)]
    assert len(feedback)==2
    by_id={r['observation_id']:r for r in observations}
    records=[]
    for i, trial in enumerate(feedback):
        assert trial['acknowledged'] and trial['changed_cell_count']==0
        assert by_id[trial['before_observation_id']]['grid']==by_id[trial['after_observation_id']]['grid']
        x,y=(trial['action'][key] for key in ('x','y'))
        assert (x,y) in ((15,15),(47,15))
        left=x<32
        location=('the right-hand portion of the bottom horizontal strip of the small central patch '
                  'in the upper-left 3-by-3 arrangement' if left else
                  'the bottom end of the leftmost vertical strip of the small central patch '
                  'in the upper-right 3-by-3 arrangement')
        box=[14,14,17,15] if left else [46,14,47,15]
        grid=by_id[trial['before_observation_id']]['grid']
        assert box[0]<=x<=box[2] and box[1]<=y<=box[3]
        assert all(grid[yy][xx]==grid[y][x] for yy in range(box[1],box[3]+1) for xx in range(box[0],box[2]+1))
        records.append(f'Test {i+1} is completed: CLICK was sent at {location} and acknowledged. '
                       'The before and after boards were identical: zero cells changed during the observed interval.')
    return records


def real_jobs(cases):
    jobs=[]
    common_goal='Investigate the unknown rules of this visual game.'
    common_controls='The available game control is CLICK.'
    first=next(iter(cases.values()))
    for rep in range(2):
        for variant,fields,visual in (
            ('Q',{},None),('G',{'goal':common_goal},None),
            ('GA',{'goal':common_goal,'available_controls':common_controls},None),
            ('GAI',{'goal':common_goal,'available_controls':common_controls},first['groups']['I'])):
            jobs.append(dict(case='shared-board',variant=variant,repetition=rep,family='real',
                             expected_status=None,expected_target=None,payload=payload(first['original'],fields,visual)))
        for name,case in cases.items():
            histories=factual_history(name)
            base=dict(goal=common_goal,available_controls=common_controls,completed_tests=histories)
            variants={
                'GAIH1':dict(base,completed_tests=histories[-1:]),
                'GAIH2':base,
                'GAH2':base,
                'GAIH2_P':dict(base,previous_model_plan_unverified=case['groups']['P']),
                'GAIH2_C':dict(base,current_candidates=case['groups']['C']),
                'GAIH2_K':dict(base,stored_model_claims_unverified=case['groups']['K']),
                'GAIH2_CPK':dict(base,current_candidates=case['groups']['C'],
                                previous_model_plan_unverified=case['groups']['P'],
                                stored_model_claims_unverified=case['groups']['K']),
                'GAIH2_result_withheld':dict(base,completed_tests=[s.split('The before')[0]+
                                                                  'The result is not supplied in this request.' for s in histories]),
                'AIH2':{k:v for k,v in base.items() if k!='goal'},
                'GIH2':{k:v for k,v in base.items() if k!='available_controls'},
            }
            for variant,fields in variants.items():
                jobs.append(dict(case=name,variant=variant,repetition=rep,family='real',
                                 expected_status=None,expected_target=None,
                                 payload=payload(case['original'],fields,None if variant=='GAH2' else case['groups']['I'])))
    assert len(jobs)==48
    return jobs


def build():
    cases=read_cases()
    jobs=calibration_jobs(next(iter(cases.values()))['original'])+real_jobs(cases)
    random.Random(2909202607).shuffle(jobs)
    return cases,jobs


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--prepare-only',action='store_true')
    args=p.parse_args()
    cases,jobs=build()
    metadata=dict(
        design='32 text-only synthetic capability/ablation controls and 48 incremental actual-board probes. '
               'Same short question, optional action/target fields, no system instruction, no mandatory CLICK. '
               'No causal hypotheses, candidate catalog or old model interpretation in the factual base.',
        factors='G goal; A available controls; I current image; H1/H2 one/two completed factual tests; '
                'P previous model plan; C candidates; K stored model claims. Result-withheld retains action/location.',
        calibration='Lamp: goal ON/OFF x reversed LEFT/RIGHT effects x achieved/pending, 2 repetitions. '
                    'Pending cases omit goal/history/control-list separately, once. '
                    'Prerequisite: reverse button roles x open/closed cover, once. '
                    'Synthetic rules are supplied as facts; this is not learning unknown game rules from vision.',
        real='Two ft09 histories sharing the same image. Actual locations are manually worded and checked '
             'against recorded click cells and homogeneous regions. No precise click coordinates in factual history. '
             'Old model prose may itself contain coordinates. No new actions, gold next move, or success-rate claim.',
        sources={n:dict(path=c['source'],sha256=digest(c['original'])) for n,c in cases.items()},
        question=QUESTION,probe_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        interpretation='A changed rationale alone is not a changed decision. Verbal reasons are not internal attention. '
                       'Calibration sufficiency is task/model/format-specific, not global input minimality.')
    if args.prepare_only:
        print(json.dumps(dict(requests=len(jobs),calibration=32,real=48,metadata=metadata),ensure_ascii=False))
        return
    execute(jobs,args.output.resolve(),metadata)


if __name__=='__main__':main()
