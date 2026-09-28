"""Exploratory, shorter event/relationship representation on the frozen tasks.

Added after seeing early verbose-arm errors. No change to tasks or answer key.
Uses only previously measured facts, no simulator rule or functional role labels.
"""
import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import shutil
import statistics
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.benchmark_rule_inference import INSTRUCTION, run, validate_answer
from scripts.benchmark_attention_selection import now
from scripts.benchmark_parallel import digest, read_lines
from scripts.benchmark_recognition import save_json
from scripts.benchmark_official_relations import sha

SOURCE = Path('outputs/rule-inference-20260928-final')
COLORS = ['white', 'light gray', 'gray', 'dark gray', 'charcoal', 'black', 'magenta', 'pink', 'red', 'blue', 'cyan', 'yellow', 'orange', 'maroon', 'green', 'purple']


def ref(obj):
    colors = sorted(map(int, obj['colors']))
    return dict(concept=obj['class_name'], colors=[COLORS[c] for c in colors], top_left=obj['bbox'][:2])


def state(s):
    result = []
    for obj in s['instances']:
        r = ref(obj)
        r['size_pixels'] = obj['size']
        if obj['class_name'] == 'regular_grid_3x3': r['cells'] = [[COLORS[v] for v in row] for row in obj['pattern']]
        result.append(r)
    return result


def equal_pairs(s):
    result = []
    for i, a in enumerate(s['instances']):
        for b in s['instances'][i+1:]:
            if a['appearance_signature'] == b['appearance_signature']:
                result.append([ref(a), ref(b)])
    return result


def brief_trial(t):
    old = {o['id']: o for o in t['before']['instances']}
    new = {o['id']: o for o in t['after']['instances']}
    result = dict(id=t['id'], action=t['action'], changed=[], unchanged=[], identical_patterns_after=equal_pairs(t['after']))
    for e in t['changes']['changes']:
        r = dict(change=e['kind'], before=ref(old[e['before_id']]) if e['before_id'] else None,
            after=ref(new[e['after_id']]) if e['after_id'] else None)
        if 'displacement_pixels' in e: r['displacement_pixels'] = e['displacement_pixels']
        if 'changed_cells' in e:
            r['changed_cells'] = [dict(row=c['row'], column=c['column'], before=COLORS[c['before']], after=COLORS[c['after']]) for c in e['changed_cells']]
        if 'candidates' in e: r['alternative_correspondences'] = [ref(new[i]) for i in e['candidates']]
        result['changed'].append(r)
    for a, b, reason in t['changes']['correspondences']:
        if reason == 'same_observed_state': result['unchanged'].append(ref(old[a]))
    return result


def request_for(c, m):
    ctx = dict(concepts=m['current']['classes'],
        note='Records are summaries derived from independently measured pixels. References use appearance and location; they are not confirmed roles or identities. Empty changed means no measured change in the extracted objects. All instances remain separate even when their concepts or patterns match.',
        trials=[brief_trial(t) for t in m['trials']], current=state(m['current']),
        query_action=c['query_action'], prediction_options=[dict(id=o['id'], state=state(o['state'])) for o in m['options']],
        target=state(m['target']), available_actions=['A', 'B', 'WAIT'])
    return dict(model='qwen3-vl-4b-instruct', messages=[dict(role='system', content=INSTRUCTION),
        dict(role='user', content=json.dumps(ctx, separators=(',', ':')))], temperature=0,
        max_tokens=650, stream=False, cache_prompt=False, seed=280928)


def prepare(out):
    out.mkdir(parents=True, exist_ok=False); (out/'sources').mkdir()
    cases = json.loads((SOURCE/'cases.json').read_text()); measured = json.loads((SOURCE/'measured.json').read_text())
    jobs = []
    for c in cases:
        p = request_for(c, measured[c['name']])
        jobs.append(dict(id=len(jobs), case=c['name'], arm='brief', payload=p, payload_digest=digest(p)))
    save_json(out/'cases.json', cases); save_json(out/'jobs.json', jobs)
    paths = [Path(__file__), Path('scripts/benchmark_rule_inference.py')]
    for p in paths: shutil.copyfile(p, out/'sources'/p.name)
    save_json(out/'plan.json', dict(created_at=now(), requests=len(jobs), warmups=1, arms=['brief'],
        jobs_digest=digest(jobs), cases_digest=digest(cases), sources={str(p): sha(p) for p in paths},
        source=str(SOURCE), source_measurements_digest=digest(measured),
        exploratory=True, rationale='Added after inspecting early verbose-arm answers. Compressed changed/unchanged and equal-pattern relationships, readable color names, no repeated full past state matrices. Current/options/target retain measured detail. Same question, choices, answer key and system instruction. Representation/content volume changes jointly; no attribution to brevity alone. No tuning after this run.'))
    print('Prepared exploratory', len(jobs), 'requests')


def analyze(out):
    cases = {c['name']: c for c in json.loads((out/'cases.json').read_text())}
    rows = [r for r in read_lines(out/'responses.jsonl') if not r['warmup']]; assert len(rows) == len(cases)
    scored = []
    for r in rows:
        c = cases[r['case']]; p = r.get('parsed', {})
        prediction = bool(r['valid'] and p.get('prediction_choice') == c['truth']['prediction_choice'])
        action = bool(r['valid'] and p.get('action_choice') == c['truth']['action_choice'])
        scored.append({k: r.get(k) for k in ['case', 'arm', 'valid', 'seconds', 'parsed', 'answer', 'error']} |
            dict(family=c['family'], prediction=prediction, action=action, joint=prediction and action,
                truth=c['truth'], prompt_tokens=r.get('response', {}).get('usage', {}).get('prompt_tokens')))
    main = [r for r in scored if r['family'] != 'uncertain']; controls = [r for r in scored if r['family'] == 'uncertain']
    arm = dict(arm='brief', valid=sum(r['valid'] for r in scored), total=len(scored), main_cases=len(main),
        prediction=sum(r['prediction'] for r in main), action=sum(r['action'] for r in main), joint=sum(r['joint'] for r in main),
        uncertainty_joint=sum(r['joint'] for r in controls), uncertainty_cases=len(controls), median_seconds=statistics.median(r['seconds'] for r in scored),
        max_prompt_tokens=max(r['prompt_tokens'] or 0 for r in scored),
        families={f: {k: sum(r[k] for r in main if r['family'] == f) for k in ['prediction','action','joint']} for f in ['counter','orbit','clock','panels']})
    save_json(out/'summary.json', dict(arms=[arm], rows=scored))
    print(json.dumps(arm, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('command', choices=['prepare','run','analyze']); p.add_argument('--output', required=True, type=Path)
    a = p.parse_args(); globals()[a.command](a.output)
