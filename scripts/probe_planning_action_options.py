"""Can a small explicit action set support failure-aware selection without object catalogs?"""
import argparse
import hashlib
import json
from pathlib import Path
import random

from probe_incremental_planning_inputs import execute, read_cases
from probe_planning_input_basis import payload, factual_history
from probe_repeated_click_inputs import digest


def build():
    cases=read_cases();jobs=[]
    for name,case in cases.items():
        history=factual_history(name)
        location=history[0].split('CLICK was sent at ',1)[1].split(' and acknowledged.',1)[0]
        fact=(f'Two completed CLICK tests at {location} were acknowledged. '
              'Neither test changed any board cell during its observed interval.')
        untested='the center of the small central patch in the lower-left 3-by-3 arrangement'
        other='the center of the small central patch in the lower-right 3-by-3 arrangement'
        fourth=('the center of the small central patch in the upper-right 3-by-3 arrangement'
                if name.startswith('r1') else 'the center of the small central patch in the upper-left 3-by-3 arrangement')
        for reverse in (False,True):
            for variant,count,visual,goal in (
                ('two_options',2,True,'Identify an interaction whose effect has not yet been tested.'),
                ('two_options_no_image',2,False,'Identify an interaction whose effect has not yet been tested.'),
                ('four_options',4,True,'Identify an interaction whose effect has not yet been tested.'),
                ('two_options_generic_goal',2,True,'Investigate the unknown rules of this visual game.'),
                ('two_options_repeat_goal',2,True,'Check whether repeating the last interaction gives the same observable result.')):
                choices=[dict(action='CLICK',target=t) for t in (location,untested,other,fourth)[:count]]
                if reverse:choices.reverse()
                fields=dict(goal=goal,available_controls='The available game control is CLICK.',
                            completed_tests=[fact],available_action_options=choices)
                jobs.append(dict(case=name,variant=variant,repetition=int(reverse),
                                 option_order='reversed' if reverse else 'tested_first',
                                 tested_target=location,untested_targets=[untested,other,fourth][:count-1],
                                 payload=payload(case['original'],fields,case['groups']['I'] if visual else None)))
    random.Random(2909202631).shuffle(jobs)
    return cases,jobs


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();cases,jobs=build()
    execute(jobs,args.output.resolve(),dict(
        design='20 exploratory requests: two real histories x five conditions x reversed option order. '
               'Two or four manually supplied high-level CLICK targets; exact tested location retained. '
               'No object catalog, target IDs, old plans, model claims, next-action validator, or forced choice.',
        limitation='Supplying alternatives is assistance from the experimenter. It tests selection, not autonomous '
                   'discovery of affordances. The untested-effect goal explicitly requests new evidence. '
                   'Different targets have no guaranteed effect. No game actions or success-rate claim.',
        probe_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        sources={n:dict(path=c['source'],sha256=digest(c['original'])) for n,c in cases.items()}))


if __name__=='__main__':main()
