"""Exploratory 2x2 factual-history compression x exploration objective, plus retry control."""
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
    cases=read_cases();jobs=[]
    for rep in range(2):
        block=[]
        for name,case in cases.items():
            history=factual_history(name)
            location=history[0].split('CLICK was sent at ',1)[1].split(' and acknowledged.',1)[0]
            compact=(f'Two completed CLICK tests at {location} were acknowledged. '
                     'Neither test changed any board cell during its observed interval.')
            for variant,goal,records in (
                ('base','Investigate the unknown rules of this visual game.',history),
                ('compact','Investigate the unknown rules of this visual game.',[compact]),
                ('new_effect','Identify an interaction whose effect has not yet been tested.',history),
                ('new_effect_compact','Identify an interaction whose effect has not yet been tested.',[compact]),
                ('repeatability','Check whether repeating the last interaction gives the same observable result.',history)):
                fields=dict(goal=goal,available_controls='The available game control is CLICK.',completed_tests=records)
                block.append(dict(case=name,variant=variant,repetition=rep,
                                  payload=payload(case['original'],fields,case['groups']['I'])))
        random.Random(2909202617+rep).shuffle(block);jobs.extend(block)
    return cases,jobs


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();cases,jobs=build()
    execute(jobs,args.output.resolve(),dict(
        design='Exploratory follow-up after 80-request basis probe. Two histories x five variants x two repetitions.',
        variants='2x2 unchanged generic goal / untested-effect goal x original two trial sentences / compact two-trial fact; '
                 'plus explicit repeatability objective control. Same facts, images, controls, question, output schema.',
        limitation='An explicit untested-effect goal changes the task objective, not just neutral formatting. '
                   'This tests goal responsiveness, not spontaneous discovery of the best exploration policy. '
                   'No game action or optimal-action oracle. Repetition of the real inputs checks local stability only.',
        probe_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        sources={n:dict(path=c['source'],sha256=digest(c['original'])) for n,c in cases.items()}))


if __name__=='__main__':main()
