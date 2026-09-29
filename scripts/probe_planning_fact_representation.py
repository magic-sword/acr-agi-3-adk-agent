"""Exploratory refinement of the additive probe; no game actions."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import random

from probe_incremental_planning_inputs import additive_payload, execute, read_cases


def build_refinement_jobs(cases, repeats):
    jobs = []
    for repetition in range(repeats):
        block = []
        for name, case in cases.items():
            groups = case['groups']
            action = groups['F']['action']
            facts = {k: v for k, v in groups['F'].items() if k != 'action'}
            sentence = (f"The previous CLICK at x={action['x']}, y={action['y']} was acknowledged. "
                        'The before and after boards were identical: zero cells changed in that observed interval.')
            knowledge = [{'subject': k['subject'],
                          'observed_fact': f"The tested click on {k['subject']} produced no visible change. "
                                           'Whether this target responds under other conditions is unknown.'}
                         for k in groups['K']]
            specs = [
                ('P_action', (), {'previous_planned_action': groups['P']['planned_action']}),
                ('P_description', (), {'previous_description': {k: groups['P'][k] for k in ('baseline', 'targets')}}),
                ('F_action', (), {'last_actual_action': action}),
                ('F_result', (), {'last_actual_result': facts}),
                ('IF_sentence', ('I',), {'last_actual_result': sentence}),
                ('IK_factual', ('I',), {'past_observations': knowledge}),
                ('ICF', ('I', 'C', 'F'), None),
                ('ICF_task', ('I', 'C', 'F'), None),
            ]
            for variant, selected, extra in specs:
                payload = additive_payload(case, selected)
                if extra is not None:
                    payload['messages'][0]['content'].append({
                        'type': 'text', 'text': json.dumps(deepcopy(extra), separators=(',', ':'))})
                if variant == 'ICF_task':
                    payload['messages'][0]['content'][0]['text'] = (
                        'You are exploring a visual game whose rules are not yet known. '
                        'The available control is CLICK. What should you do next? '
                        'Use the observed result of the previous action when deciding.')
                block.append(dict(case=name, variant=variant, groups=list(selected),
                                  repetition=repetition, payload=payload))
        random.Random(29092027 + repetition).shuffle(block)
        jobs += block
    return jobs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--repeats', default=3, type=int)
    args = parser.parse_args()
    cases = read_cases()
    execute(build_refinement_jobs(cases, args.repeats), args.output.resolve(), dict(
        phase='refinement', repeats=args.repeats,
        design='Exploratory follow-up selected after observing first additive block, not preregistered confirmation. '
               'Eight conditions, two source histories, three interleaved repetitions.',
        variants={
            'P_action': 'Only the previous planned action, no image or result',
            'P_description': 'Only previous baseline and target descriptions, no planned action, image or result',
            'F_action': 'Only the actual previous action and coordinates',
            'F_result': 'Only the measured unchanged result, without action or coordinates',
            'IF_sentence': 'Current image and same actual no-change result expressed as a past-tense sentence',
            'IK_factual': 'Current image and prior target knowledge rewritten as a limited negative observation',
            'ICF': 'Current image, current candidates and structured actual previous result',
            'ICF_task': 'Same ICF evidence with explicit game exploration, available control and use-result instruction'},
        limitations='P subfields are correlated target descriptions. Rewriting K changes wording/status/metadata '
                    'together, not a one-word negation test. No execution or task-success claim. '
                    'Two histories share the board. Repetition measures local stability, not independent games.'))


if __name__ == '__main__':
    main()
