"""Independent, additive input probes of recorded planning failures; no game actions.

The six evidence groups are explicitly constructed by allowlist. No conversation
history, runtime context wrapper, or previous answer is implicitly carried over.
"""
import argparse
from copy import deepcopy
import hashlib
from itertools import combinations
import json
import os
from pathlib import Path
import random
import time
import urllib.request

from probe_repeated_click_inputs import context_part, digest

ROOT = Path(__file__).resolve().parents[1]
GROUPS = ('I', 'C', 'F', 'P', 'K', 'L')
QUESTION = 'What should you do next?'
SIMPLE_SYSTEM = 'Decide the next step from the supplied information. State when information is insufficient.'
SIMPLE_TOOLS = [{
    'type': 'function', 'function': {
        'name': 'submit_next_step',
        'description': 'Report the next step, or the information needed to decide.',
        'parameters': {
            'type': 'object', 'additionalProperties': False,
            'properties': {
                'status': {'type': 'string', 'enum': ['act', 'need_information']},
                'next_step': {'type': 'string'},
                'target': {'type': 'string'},
                'candidate_refs': {'type': 'array', 'items': {'type': 'string'}},
            },
            'required': ['status', 'next_step'],
        },
    },
}]


def read_cases():
    cases = {}
    for name in ('r1-ft09-separated', 'r2-ft09-separated'):
        path = ROOT / 'outputs/repeated-click-xai-20260929' / f'{name}-baseline_start.request.json'
        original = json.loads(path.read_text())
        _, context = context_part(original)
        measurement = next(iter(context['measurements'].values()))
        trial = context['last_completed_trial']
        images = [p for m in original['messages'] if isinstance(m['content'], list)
                  for p in m['content'] if p['type'] == 'image_url']
        result = trial['actual_result']
        groups = {
            'I': deepcopy(images[-1]),
            'C': {
                'candidate_index_columns': measurement['candidate_index_columns'][:3],
                'candidate_index': [r[:3] for r in measurement['candidate_index']],
                'color_names': measurement['color_names'],
                'note': 'Regions may overlap or include background.',
            },
            'F': {k: deepcopy(result[k]) for k in (
                'action', 'acknowledged', 'frame_changed', 'changed_cell_count',
                'changed_cells', 'change_bounds')},
            'P': {k: deepcopy(trial[k]) for k in (
                'baseline', 'expected_effect', 'targets', 'planned_action')},
            'K': deepcopy(context['long_term_knowledge']),
            'L': deepcopy(measurement['target_correspondence']),
        }
        groups['F']['action'] = {k: result['action'][k] for k in ('action', 'x', 'y')}
        cases[name] = dict(original=original, groups=groups, source=str(path.relative_to(ROOT)))
    return cases


def minimal_payload(original):
    return {k: deepcopy(original[k]) for k in ('model', 'temperature', 'max_tokens', 'stream')}


def additive_payload(case, selected, free=False):
    payload = minimal_payload(case['original'])
    content = [{'type': 'text', 'text': QUESTION}]
    labels = {'C': 'current_candidates', 'F': 'last_actual_result',
              'P': 'previous_plan_unverified', 'K': 'long_term_knowledge',
              'L': 'previous_target_correspondence'}
    for group in GROUPS:
        if group not in selected:
            continue
        if group == 'I':
            content += [{'type': 'text', 'text': 'Current board'}, deepcopy(case['groups']['I'])]
        else:
            content.append({'type': 'text', 'text': json.dumps(
                {labels[group]: case['groups'][group]}, ensure_ascii=False, separators=(',', ':'))})
    payload['messages'] = [{'role': 'user', 'content': content}]
    if not free:
        payload.update(tools=deepcopy(SIMPLE_TOOLS), tool_choice='required', parallel_tool_calls=False)
    return payload


def build_jobs(cases, repeats, phase):
    variants = [()] + [(g,) for g in GROUPS] + list(combinations(GROUPS, 2))
    variants += [GROUPS] + [tuple(g for g in GROUPS if g != omitted) for omitted in GROUPS]
    jobs = []
    for repetition in range(repeats):
        block = []
        if phase in ('all', 'additive'):
            for name, case in cases.items():
                for selected in variants:
                    block.append(dict(case=name, variant=''.join(selected) or 'empty',
                                      groups=list(selected), repetition=repetition,
                                      payload=additive_payload(case, selected)))
        if phase in ('all', 'bridge'):
            for name, case in cases.items():
                for prompt in ('original', 'simple'):
                    for schema in ('original', 'simple'):
                        payload = deepcopy(case['original'])
                        if prompt == 'simple':
                            payload['messages'][0] = {'role': 'system', 'content': SIMPLE_SYSTEM}
                        if schema == 'simple':
                            payload['tools'] = deepcopy(SIMPLE_TOOLS)
                            payload['messages'][0]['content'] = payload['messages'][0]['content'].replace(
                                'submit_one_action', 'submit_next_step')
                        block.append(dict(case=name, variant=f'bridge_{prompt}_{schema}',
                                          groups=None, repetition=repetition, payload=payload))
            # The two empty cases are identical. Only one free-text control is needed.
            name = next(iter(cases))
            block.append(dict(case=name, variant='question_only_free', groups=[],
                              repetition=repetition, payload=additive_payload(cases[name], (), free=True)))
        random.Random(29092026 + repetition).shuffle(block)
        jobs += block
    return jobs


def execute(jobs, out, metadata):
    out.mkdir(parents=True, exist_ok=False)
    plan = dict(metadata, requests=len(jobs),
                script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                jobs=[{k: v for k, v in job.items() if k != 'payload'} |
                      {'request_sha256': digest(job['payload'])} for job in jobs])
    (out / 'plan.json').write_text(json.dumps(plan, ensure_ascii=False, indent=2) + '\n')
    api = os.getenv('VLM_API_BASE', 'http://vlm:8080/v1').rstrip('/')
    with urllib.request.urlopen(api.removesuffix('/v1') + '/props', timeout=15) as response:
        (out / 'server-props.json').write_text(json.dumps(json.load(response), indent=2) + '\n')
    results = []
    for index, job in enumerate(jobs):
        row = {k: v for k, v in job.items() if k != 'payload'}
        payload = job['payload']
        row['request_sha256'] = digest(payload)
        filename = f"{job['case']}-{job['variant']}-r{job['repetition']}.request.json"
        row['request_file'] = filename
        (out / filename).write_text(json.dumps(payload, ensure_ascii=False) + '\n')
        start = time.monotonic()
        try:
            request = urllib.request.Request(api + '/chat/completions',
                data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'})
            with urllib.request.urlopen(request, timeout=90) as response:
                row['response'] = json.load(response)
            message = row['response']['choices'][0]['message']
            if message.get('tool_calls'):
                call = message['tool_calls'][0]['function']
                row['tool'] = call['name']
                row['answer'] = json.loads(call['arguments'])
            else:
                row['answer'] = message.get('content')
        except Exception as exc:
            row['error'] = f'{type(exc).__name__}: {exc}'
        row['seconds'] = time.monotonic() - start
        results.append(row)
        (out / 'results.json').write_text(json.dumps(results, ensure_ascii=False, indent=2) + '\n')
        print(f"{index + 1}/{len(jobs)} {job['case']} {job['variant']} r{job['repetition']}",
              json.dumps(row.get('answer', row.get('error')), ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--phase', choices=('all', 'additive', 'bridge'), default='all')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('--repeats must be positive')
    cases = read_cases()
    jobs = build_jobs(cases, args.repeats, args.phase)
    metadata = dict(
        groups={'I': 'Current image only, no previous image',
                'C': 'Current candidates: ID, bounding box, colors and palette only; no history or tracking',
                'F': 'Previous actual action and measured result; no prediction or assessment',
                'P': 'Previous plan: baseline, prediction, targets, planned action',
                'K': 'Stored conditional knowledge, including status and provenance',
                'L': 'Previous target to current candidate correspondence'},
        sources={n: {'file': c['source'], 'sha256': digest(c['original'])} for n, c in cases.items()},
        design='Empty, all six singles, all 15 pairs, full set and all six leave-one-out sets. '
               'Three repetitions in shuffled blocks; each is a new independent request. '
               'Bridge: 2x2 original/simple system instruction and output schema, original full user input fixed. '
               'Literal question-only free text control. No game execution.',
        limitations='Two selected histories of the same board, not independent games. Minimality only among '
                    'tested groups under fixed question/schema/serialization. Model stochasticity, prompt '
                    'sensitivity and missing-input distribution shift remain. Need-information is allowed '
                    'in simple schema. L without C has unresolved references. No runtime accuracy claim.',
        phase=args.phase, repeats=args.repeats)
    if args.dry_run:
        print(json.dumps(dict(metadata, requests=len(jobs)), ensure_ascii=False, indent=2))
        return
    execute(jobs, args.output.resolve(), metadata)


if __name__ == '__main__':
    main()
