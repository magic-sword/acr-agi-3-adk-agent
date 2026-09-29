"""Matched real histories: same two actions, swap which was actually tried; no image."""
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
    cases=read_cases();jobs=[];facts={};locations={}
    for name in cases:
        histories=factual_history(name)
        loc=histories[0].split('CLICK was sent at ',1)[1].split(' and acknowledged.',1)[0]
        locations[name]=loc
        facts[name]=(f'Two completed CLICK tests at {loc} were acknowledged. '
                     'Neither test changed any board cell during its observed interval.')
    names=list(cases)
    for reverse in (False,True):
        options=[dict(action='CLICK',target=locations[n]) for n in names]
        if reverse:options.reverse()
        for name,case in cases.items():
            other=next(n for n in names if n!=name)
            for variant,goal in (
                ('generic','Investigate the unknown rules of this visual game.'),
                ('new_effect','Identify an interaction whose effect has not yet been tested.'),
                ('repeatability','Check whether repeating the last interaction gives the same observable result.'),
                ('without_goal',None),
                ('result_withheld','Investigate the unknown rules of this visual game.')):
                fields=dict(available_action_options=deepcopy(options),completed_tests=[facts[name]])
                if goal is not None:fields=dict(goal=goal,**fields)
                if variant=='result_withheld':
                    fields['completed_tests']=[facts[name].split('Neither test')[0]+'The outcomes are not supplied in this request.']
                jobs.append(dict(case=name,variant=variant,repetition=int(reverse),
                                 option_order='reversed' if reverse else 'original',
                                 tested_target=locations[name],untested_targets=[locations[other]],
                                 payload=payload(case['original'],fields)))
        first=cases[names[0]]
        jobs.append(dict(case='shared-options',variant='without_history',repetition=int(reverse),
                         option_order='reversed' if reverse else 'original',tested_target=None,untested_targets=[],
                         payload=payload(first['original'],dict(goal='Investigate the unknown rules of this visual game.',
                                                              available_action_options=options))))
    random.Random(2909202647).shuffle(jobs)
    assert len(jobs)==22
    return cases,jobs


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();cases,jobs=build()
    execute(jobs,args.output.resolve(),dict(
        design='Exploratory confirmation: identical goal and two action descriptions across two real histories. '
               'Only which location was actually tested differs. Reverse option order as well. No image, '
               'separate control list, object inventory, old plan, causal hypothesis, or thought procedure.',
        variants='Generic exploration / untested-effect / repeatability goals; omit goal; withhold outcomes; '
                 'omit history (deduplicated across histories). 22 requests, no identical-payload repetitions.',
        limitation='The two options are supplied by the experimenter from existing executed locations. '
                   'Novelty selection does not prove either action advances the game. No autonomous candidate generation '
                   'or global minimality proof. Both histories have the same unchanged current board.',
        probe_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        sources={n:dict(path=c['source'],sha256=digest(c['original'])) for n,c in cases.items()}))


if __name__=='__main__':main()
