"""Small action-result table: bind history to explicit action IDs instead of prose joins."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random

from probe_incremental_planning_inputs import execute, read_cases
from probe_planning_input_basis import payload, factual_history
from probe_repeated_click_inputs import digest


def build():
    cases=read_cases();names=list(cases);locations={};jobs=[]
    for name in names:
        text=factual_history(name)[0]
        locations[name]=text.split('CLICK was sent at ',1)[1].split(' and acknowledged.',1)[0]
    for name,case in cases.items():
        tried='A' if name==names[0] else 'B'
        for reverse in (False,True):
            for descriptions in (False,True):
                for goal_kind,goal in (
                    ('explore','Investigate the unknown rules of this visual game.'),
                    ('repeat','Check whether repeating the last CLICK on the same target gives the same observable result.')):
                    actions=[]
                    for other,identity in zip(names,('A','B')):
                        row=dict(action_id=identity,control='CLICK',
                                 completed_tests=2 if identity==tried else 0,
                                 observed_effect='no board change' if identity==tried else 'not observed')
                        if descriptions:row['target']=locations[other]
                        actions.append(row)
                    if reverse:actions.reverse()
                    context=dict(goal=goal,available_actions=actions,last_completed_action_id=tried)
                    jobs.append(dict(case=name,variant=goal_kind+('_descriptions' if descriptions else '_ids'),
                                     repetition=int(reverse),order='reversed' if reverse else 'original',
                                     expected_action_id=tried if goal_kind=='repeat' else ('B' if tried=='A' else 'A'),
                                     action_targets=dict(A=locations[names[0]],B=locations[names[1]]),
                                     payload=payload(case['original'],context)))
    random.Random(2909202671).shuffle(jobs)
    assert len(jobs)==16
    return cases,jobs


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();cases,jobs=build()
    execute(jobs,args.output.resolve(),dict(
        design='16 exploratory requests: two actual no-change histories x reversed action row order x '
               'target descriptions present/absent x exploration/repetition objective. Two stable action IDs; '
               'trial counts and effects attached to each action row; last action explicitly referenced by ID.',
        limitation='Tests small action selection after host binding, not discovery of target semantics or optimal '
                   'gameplay. IDs-only cases deliberately require an external mapping for execution. '
                   'Multiple representation changes versus prose history; does not isolate naming alone.',
        probe_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        sources={n:dict(path=c['source'],sha256=digest(c['original'])) for n,c in cases.items()}))


if __name__=='__main__':main()
