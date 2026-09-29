"""Paired four-arm requests at identical recorded A-arm replanning states."""
import argparse
import base64
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agent.cognition.action_update_comparison import ActionUpdateRuntime, UPDATE_INSTRUCTION
from agent.cognition.memory_comparison import SeparatedMemoryRuntime
from probe_incremental_planning_inputs import execute
from probe_repeated_click_inputs import context_part, digest


def rows(path):
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def make_cases(source):
    cases = []
    for directory in sorted(source.glob('r*-*-A')):
        logs = next(directory.glob('*/cognition'))
        requests = rows(next(logs.glob('*.requests.jsonl')))
        artifacts = rows(next(logs.glob('*.artifacts.jsonl')))
        observations = rows(next(logs.glob('*.observations.jsonl')))
        eligible = [r for r in requests if r['work'] == 'ground' and
                    any(t['event'] == 'update_trial' and t['sequence'] < r['sequence'] for t in artifacts)]
        if not eligible:
            continue
        # Fixed rule: use the second replan if available (two failures), else first.
        request = eligible[min(1, len(eligible)-1)]
        payload = deepcopy(request['request'])
        for m in payload['messages']:
            if not isinstance(m['content'], list):
                continue
            for p in m['content']:
                if p['type'] != 'image_url':
                    continue
                info = p['image_url']
                data = (logs/info['path']).read_bytes()
                assert hashlib.sha256(data).hexdigest() == info['sha256']
                p['image_url'] = {'url': f"data:{info['mime_type']};base64,"+base64.b64encode(data).decode()}
                if info.get('detail') is not None:
                    p['image_url']['detail'] = info['detail']
        assert digest(payload) == request['request_sha256']
        _, context = context_part(payload)
        observation = next(o for o in observations if o['observation_id'] == context['observation_id'])
        perception = next(r['measurement'] for r in artifacts if r['event'] == 'objects_measured'
                          and r['observation_id'] == context['observation_id'])
        trials = [r['trial'] for r in artifacts if r['event'] == 'update_trial' and r['sequence'] < request['sequence']]
        cases.append(dict(name=directory.name, source=str(logs.relative_to(ROOT)),
                          payload=payload, context=context, observation=observation,
                          perception=perception, trials=trials))
    return cases


def variant(case, arm):
    payload = deepcopy(case['payload'])
    if arm in 'BD':
        runtime = object.__new__(ActionUpdateRuntime)
        runtime.update_arm = arm
        runtime.obs = case['observation']
        runtime.perception = case['perception']
        runtime.trial_ledger = case['trials']
        with patch.object(SeparatedMemoryRuntime, '_context', return_value=deepcopy(case['context'])):
            context = runtime._context('ground')
        part, _ = context_part(payload)
        part['text'] = json.dumps(context, ensure_ascii=False, separators=(',', ':'))
    if arm in 'CD':
        assert payload['messages'][0]['role'] == 'system'
        payload['messages'][0]['content'] += UPDATE_INSTRUCTION
    assert payload['tools'] == case['payload']['tools']
    return payload


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--prepare-only', action='store_true')
    args = p.parse_args()
    cases = make_cases(args.source.resolve())
    jobs = []
    for rep in range(args.repeats):
        block = [dict(case=c['name'], variant=a, repetition=rep, payload=variant(c, a))
                 for c in cases for a in 'ABCD']
        random.Random(29102026+rep).shuffle(block)
        jobs.extend(block)
    metadata = dict(design='Identical recorded A-arm state, history, before/current images and output schema; '
                          'production B/D representation transform and C/D instruction. No action execution.',
                    selection='For each A run: second replan after completed review if available, otherwise first; '
                              'runs without completed review/replan are excluded and reported.',
                    sources={c['name']: dict(path=c['source'], request_sha256=digest(c['payload']),
                                            observation_id=c['observation']['observation_id'],
                                            previous_trial=c['trials'][-1]) for c in cases},
                    repetitions=args.repeats, independent_games=False)
    if args.prepare_only:
        print(json.dumps(dict(cases=len(cases), requests=len(jobs), names=[c['name'] for c in cases])))
        return
    execute(jobs, args.output.resolve(), metadata)


if __name__ == '__main__':
    main()
