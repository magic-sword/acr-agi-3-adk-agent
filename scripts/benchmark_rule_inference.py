"""Frozen synthetic interventions: pixel measurements -> local rule induction.

No generated rule, simulator identity, future truth, or semantic role reaches the
model. All three arms share per-frame observations and the same question/options.
Only computed temporal differences and then within-frame relations are added.
"""
import argparse
from copy import deepcopy
import itertools
import json
from pathlib import Path
import random
import shutil
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.instance_state import extract, compare
from scripts.benchmark_instance_state import render, PALETTE
from scripts.benchmark_attention_selection import http, now
from scripts.benchmark_parallel import digest, read_lines
from scripts.benchmark_recognition import save_json
from scripts.benchmark_official_relations import sha

ARMS = ['states', 'changes', 'relations']
SOURCE = Path('outputs/instance-state-20260928-final/cases.json')
INSTRUCTION = '''Infer a provisional local game rule from the observed trials, then answer two questions.
Each trial starts from its own BEFORE state: trials are independent interventions, not a continuous movie. Each action is followed by one equal-duration observation interval. WAIT means no button is pressed, but time still passes. A and B are opaque button names with NO directional meaning.
The geometric classes describe shared concepts, not functional roles. IDs are local to a snapshot, not persistent identities. Object grouping and correspondence are hypotheses. Multiple objects may respond differently to the same action. Coordinates are original pixels, x rightwards and y downwards. Repeated panels share a concept but remain distinct instances with internal cells.
All measurements are computed from pixels. Changes/relations, when provided, are computed summaries, not known game rules. No player/goal/hint role or winning condition is supplied. Do not claim such roles are confirmed.
Use the simplest consistent local rule supported by repeated and contrasting trials to predict a NEW state. This is provisional extrapolation, not a universal proof. If distinct plausible rules remain untested and predict different answers, choose uncertain. For ambiguous identity, do not silently prefer proximity. Compare evidence across trials; do not assume an unchanged appearance means an unchanged position.
prediction_choice: choose the option ID whose state best predicts CURRENT after the query action (or uncertain).
action_choice: choose A, B, or WAIT that best achieves the specified target from CURRENT (or uncertain).
Return only JSON: {"prediction_choice":"option ID or uncertain","action_choice":"A/B/WAIT/uncertain","rule_hypothesis":"brief rule, or why not identifiable","evidence_trials":["trial IDs"],"limitation":"what remains unproven","next_test":"one discriminating intervention"}.
Keep each text field to at most two short sentences. Do not invent success feedback.
''' + PALETTE


def board():
    return [[5] * 64 for _ in range(64)]


def objects(positions, colors):
    g = board()
    for (x, y), color in zip(positions, colors):
        for dy in range(3):
            for dx in range(3):
                g[y + dy][x + dx] = color
    return g


def panels(patterns):
    g = board()
    for (x0, y0), matrix in zip([(8, 8), (40, 8), (8, 40), (40, 40)], patterns):
        for y in range(3):
            for x in range(3):
                for dy in range(3):
                    for dx in range(3):
                        g[y0 + 5*y + dy][x0 + 5*x + dx] = matrix[y][x]
    return g


def observe(grid, label, seed):
    state = extract(grid, label)
    rng = random.Random(seed)
    tags = rng.sample(range(100, 999), len(state['instances']))
    for obj, tag in zip(state['instances'], tags):
        obj['id'] = f'{label}_{tag}'
    rng.shuffle(state['instances'])
    return state


def compact(state):
    return {'concepts': state['classes'], 'instances': [
        {k: o[k] for k in ('id', 'class_name', 'bbox', 'pattern', 'role')}
        for o in state['instances']]}


def relations(state):
    result = []
    for a, b in itertools.combinations(state['instances'], 2):
        aa, bb = a['bbox'], b['bbox']
        result.append(dict(instances=[a['id'], b['id']],
            same_concept=a['class_name'] == b['class_name'],
            same_pattern=a['appearance_signature'] == b['appearance_signature'],
            a_relative_to_b=[-1 if aa[2] < bb[0] else 1 if aa[0] > bb[2] else 0,
                             -1 if aa[3] < bb[1] else 1 if aa[1] > bb[3] else 0],
            center_offset_twice=[aa[0]+aa[2]-bb[0]-bb[2], aa[1]+aa[3]-bb[1]-bb[3]]))
    return result


def changes(before, after):
    result = compare(before, after)
    old = {o['id']: o for o in before['instances']}
    new = {o['id']: o for o in after['instances']}
    for e in result['changes']:
        a, b = old.get(e['before_id']), new.get(e['after_id'])
        if a is None or b is None:
            continue
        if e['kind'] == 'moved':
            e['displacement_pixels'] = dict(dx=b['bbox'][0]-a['bbox'][0], dy=b['bbox'][1]-a['bbox'][1])
        if a['class_name'] == b['class_name'] == 'regular_grid_3x3':
            e['changed_cells'] = [dict(row=y, column=x, before=a['pattern'][y][x], after=b['pattern'][y][x])
                for y in range(3) for x in range(3) if a['pattern'][y][x] != b['pattern'][y][x]]
    # Within-frame pattern equality belongs exclusively to the relations arm.
    result.pop('same_pattern_after')
    result.pop('shared_classes')
    return result


def make_case(name, family, trials, current, candidates, correct_index, target, correct_action, query_action, seed, note=''):
    rng = random.Random(seed)
    order = list(range(len(candidates)))
    rng.shuffle(order)
    options = [dict(id=f'P{i+1}', pixels=candidates[j]) for i, j in enumerate(order)]
    correct = 'uncertain' if correct_index is None else options[order.index(correct_index)]['id']
    return dict(name=name, family=family, trials=[dict(id=f'T{i+1}', action=a, before=b, after=c)
        for i, (a, b, c) in enumerate(trials)], current=current, options=options,
        query_action=query_action, target=target,
        truth=dict(prediction_choice=correct, action_choice=correct_action), note=note)


def build_cases():
    cases = []
    for v in range(3):
        axis = v % 2
        sign = -1 if v == 1 else 1
        goal_action = ['A', 'B', 'WAIT'][v]
        colors = [[9, 12], [14, 8], [15, 10]][v]
        def shift(pos, direction):
            q = deepcopy(pos)
            q[0][axis] += direction
            q[1][axis] -= direction
            return q
        def step(pos, action):
            return shift(pos, sign if action == 'A' else -sign if action == 'B' else 0)
        trials = []
        for action, pos in [('A', [[14, 18], [40, 40]]), ('B', [[20, 22], [38, 36]]), ('WAIT', [[24, 18], [42, 40]]), ('A', [[16, 20], [36, 38]])]:
            trials.append((action, objects(pos, colors), objects(step(pos, action), colors)))
        pos = [[18, 24], [40, 36]]
        wrong = deepcopy(pos)
        for p in wrong: p[axis] += sign
        candidates = [objects(step(pos, 'B'), colors), objects(step(pos, 'A'), colors), objects(pos, colors), objects(wrong, colors)]
        cases.append(make_case(f'counter_{v}', 'counter', trials, objects(pos, colors), candidates, 0,
            objects(step(pos, goal_action), colors), goal_action, 'B', 810+v,
            'Two separately segmented objects respond in opposite directions; 1 pixel steps.'))

        corners = [(14, 14), (40, 14), (40, 40), (14, 40)]
        def orbit(indices, action):
            d = sign if action == 'A' else -sign if action == 'B' else 0
            return [(i+d) % 4 for i in indices]
        def orb(indices): return objects([corners[i] for i in indices], colors)
        trials = [(a, orb(q), orb(orbit(q, a))) for a, q in [('A', [0, 2]), ('A', [1, 3]), ('B', [2, 0]), ('WAIT', [0, 2])]]
        pos = [3, 1]
        candidates = [orb(orbit(pos, 'A')), orb(orbit(pos, 'B')), orb(pos), orb([(pos[0]+sign) % 4, pos[1]])]
        cases.append(make_case(f'orbit_{v}', 'orbit', trials, orb(pos), candidates, 0,
            orb(orbit(pos, goal_action)), goal_action, 'A', 910+v,
            'Position-dependent coupled changes around four corners; unseen query arrangement.'))

        def clock_step(pos, action):
            q = deepcopy(pos)
            q[0][0] += 1  # autonomous, also on WAIT
            q[1][axis] += sign if action == 'A' else -sign if action == 'B' else 0
            return q
        trials = [(a, objects(p, colors), objects(clock_step(p, a), colors)) for a, p in
            [('A', [[12, 12], [32, 36]]), ('WAIT', [[14, 12], [32, 36]]), ('B', [[16, 12], [36, 32]]), ('WAIT', [[18, 12], [36, 32]])]]
        pos = [[20, 12], [38, 38]]
        candidates = [objects(clock_step(pos, 'WAIT'), colors), objects(pos, colors), objects(clock_step(pos, 'A'), colors), objects(clock_step(pos, 'B'), colors)]
        cases.append(make_case(f'clock_{v}', 'clock', trials, objects(pos, colors), candidates, 0,
            objects(clock_step(pos, goal_action), colors), goal_action, 'WAIT', 1010+v,
            'Time-driven object versus button-dependent object, with WAIT controls.'))

        rng = random.Random(301+v)
        source_a, source_b = [(0, 1), (2, 0), (1, 2)][v]
        used_patterns = set()
        def patterns():
            while True:
                p = [[[rng.choice(colors) for _ in range(3)] for _ in range(3)] for _ in range(4)]
                hashes = {digest(x) for x in p}
                if len(hashes) == 4 and not hashes.intersection(used_patterns):
                    used_patterns.update(hashes)
                    return p
        def copied(p, action):
            q = deepcopy(p)
            if action != 'WAIT': q[3] = deepcopy(q[source_a if action == 'A' else source_b])
            return q
        trials = []
        for a in ['A', 'B', 'A', 'B', 'WAIT']:
            p = patterns()
            trials.append((a, panels(p), panels(copied(p, a))))
        p = patterns()
        other = deepcopy(p); other[3] = deepcopy(p[next(i for i in range(3) if i not in [source_a, source_b])])
        candidates = [panels(copied(p, 'A')), panels(copied(p, 'B')), panels(p), panels(other)]
        cases.append(make_case(f'panels_{v}', 'panels', trials, panels(p), candidates, 0,
            panels(copied(p, goal_action)), goal_action, 'A', 1110+v,
            'New internal patterns at query: infer copying a reference instance, not memorizing colors.'))

    g0 = objects([[16, 20]], [9]); g1 = objects([[17, 20]], [9]); current = objects([[24, 20]], [9])
    cases.append(make_case('insufficient_action', 'uncertain', [('A', g0, g1)], current,
        [current, objects([[25, 20]], [9]), objects([[23, 20]], [9])], None,
        current, 'uncertain', 'WAIT', 201,
        'A single action-response pair cannot distinguish button effect from autonomous drift; B and WAIT unobserved.'))
    old = objects([[16, 20], [40, 20]], [0, 0]); new = objects([[24, 20], [32, 20]], [0, 0])
    cases.append(make_case('ambiguous_identity', 'uncertain', [('A', old, new)], new,
        [old, objects([[28, 20], [36, 20]], [0, 0]), new], None,
        old, 'uncertain', 'B', 202,
        'Identical copies have ambiguous correspondence; reverse button mapping was never observed.'))
    return cases


def measure_case(case, seed):
    measured = dict(trials=[])
    for i, t in enumerate(case['trials']):
        before = observe(t['before'], f'T{i+1}b', seed+i*2)
        after = observe(t['after'], f'T{i+1}a', seed+i*2+1)
        measured['trials'].append(dict(id=t['id'], action=t['action'], before=before, after=after,
            changes=changes(before, after), relations=dict(before=relations(before), after=relations(after))))
    measured['current'] = observe(case['current'], 'C', seed+100)
    measured['options'] = [dict(id=o['id'], state=observe(o['pixels'], o['id'], seed+200+i)) for i, o in enumerate(case['options'])]
    measured['target'] = observe(case['target'], 'G', seed+300)
    return measured


def request_for(case, measured, arm):
    trials = []
    for t in measured['trials']:
        row = dict(id=t['id'], action=t['action'], before=compact(t['before']), after=compact(t['after']))
        if arm in ['changes', 'relations']: row['computed_changes'] = t['changes']
        if arm == 'relations': row['computed_relations'] = t['relations']
        trials.append(row)
    context = dict(trials=trials, current=compact(measured['current']), query_action=case['query_action'],
        prediction_options=[dict(id=o['id'], state=compact(o['state'])) for o in measured['options']],
        target=compact(measured['target']), available_actions=['A', 'B', 'WAIT'])
    if arm == 'relations':
        context['current_relations'] = relations(measured['current'])
        context['option_relations'] = {o['id']: relations(o['state']) for o in measured['options']}
        context['target_relations'] = relations(measured['target'])
    return dict(model='qwen3-vl-4b-instruct', messages=[dict(role='system', content=INSTRUCTION),
        dict(role='user', content=json.dumps(context, separators=(',', ':')))], temperature=0,
        max_tokens=650, stream=False, cache_prompt=False, seed=280928)


def prepare(out):
    out.mkdir(parents=True, exist_ok=False)
    (out/'sources').mkdir()
    palette = json.loads(SOURCE.read_text())[0]['palette']
    cases = build_cases()
    measurements = {}
    jobs = []
    for i, case in enumerate(cases):
        m = measure_case(case, 5000+i*1000)
        measurements[case['name']] = m
        for arm in ARMS:
            p = request_for(case, m, arm)
            jobs.append(dict(id=len(jobs), case=case['name'], arm=arm, payload=p, payload_digest=digest(p)))
        images = {'current': case['current'], 'target': case['target']}
        for t in case['trials']:
            images[t['id']+'-before'] = t['before']; images[t['id']+'-after'] = t['after']
        for o in case['options']: images[o['id']] = o['pixels']
        for label, pixels in images.items(): render(pixels, palette).save(out/f'{case["name"]}-{label}.png')
    save_json(out/'cases.json', cases)
    save_json(out/'measured.json', measurements)
    save_json(out/'jobs.json', jobs)
    paths = [Path(__file__), Path('scripts/instance_state.py')]
    for p in paths: shutil.copyfile(p, out/'sources'/p.name)
    save_json(out/'plan.json', dict(created_at=now(), arms=ARMS, requests=len(jobs), warmups=1,
        cases_digest=digest(cases), measured_digest=digest(measurements), jobs_digest=digest(jobs),
        sources={str(p): sha(p) for p in paths},
        primary='Paired prediction-choice and target-action accuracy on 12 answerable synthetic episodes; separate abstention on 2 insufficient-evidence episodes. Joint correctness requires both choices correct.',
        design='4 rule families x 3 parameterizations; plus 2 uncertainty controls. No image inputs. Same states/questions/instruction across 3 arms. Add measured differences, then deterministic same-concept/pattern/spatial relations. Local IDs/order randomized; candidate order randomized. Query panel patterns and motion positions differ from demonstrations.',
        limitations='Synthetic tasks with known simulator truth, constrained choices and simple rules; not independent games or actual play. Three variants per family are correlated, not stochastic repetitions. More evidence tokens and preprocessing in augmented arms; not equal compute. Rule is inductive hypothesis; renderer/extractor geometry is deliberately in scope. No confirmed semantic roles or winning feedback.'))
    print('Prepared', len(jobs), 'requests', flush=True)


def validate_answer(p, case):
    assert isinstance(p, dict)
    assert p['prediction_choice'] in [o['id'] for o in case['options']] + ['uncertain']
    assert p['action_choice'] in ['A', 'B', 'WAIT', 'uncertain']
    for k in ['rule_hypothesis', 'limitation', 'next_test']: assert isinstance(p[k], str)
    assert isinstance(p['evidence_trials'], list)
    assert all(x in [t['id'] for t in case['trials']] for x in p['evidence_trials'])


def run(out):
    load = lambda n: json.loads((out/n).read_text())
    plan, jobs = load('plan.json'), load('jobs.json')
    assert digest(jobs) == plan['jobs_digest']
    for path, h in plan['sources'].items(): assert sha(path) == h
    cases = {c['name']: c for c in load('cases.json')}
    assert not (out/'responses.jsonl').exists()
    save_json(out/'server-before.json', http('http://127.0.0.1:8080/props'))
    order = jobs.copy(); random.Random(7028).shuffle(order)
    with (out/'responses.jsonl').open('w') as log:
        for job, warm in [(jobs[0], True)] + [(j, False) for j in order]:
            r = {k: job[k] for k in ['id', 'case', 'arm', 'payload_digest']}
            r.update(warmup=warm, started_at=now()); start = time.monotonic()
            try:
                response = http('http://127.0.0.1:8080/v1/chat/completions', job['payload'], 240)
                choice = response['choices'][0]
                r.update(response=response, answer=choice['message']['content'])
                r['parsed'] = json.loads(r['answer'])
                validate_answer(r['parsed'], cases[job['case']])
                r['valid'] = choice['finish_reason'] == 'stop'
            except Exception as e:
                r.update(valid=False, error=f'{type(e).__name__}: {e}')
            r['seconds'] = time.monotonic()-start
            log.write(json.dumps(r, ensure_ascii=False)+'\n'); log.flush()
            print(job['id'], job['case'], job['arm'], r['valid'], round(r['seconds'], 2), r.get('error', ''), flush=True)
    save_json(out/'server-after.json', http('http://127.0.0.1:8080/props'))
    save_json(out/'health-after.json', http('http://127.0.0.1:8080/health'))


def analyze(out):
    cases = {c['name']: c for c in json.loads((out/'cases.json').read_text())}
    rows = [r for r in read_lines(out/'responses.jsonl') if not r['warmup']]
    assert len(rows) == len(cases)*len(ARMS)
    scored = []
    for r in rows:
        c = cases[r['case']]; p = r.get('parsed', {})
        scores = {k: bool(r['valid'] and p.get(k) == c['truth'][k]) for k in ['prediction_choice', 'action_choice']}
        scored.append({k: r.get(k) for k in ['case', 'arm', 'valid', 'seconds', 'parsed', 'answer', 'error']} |
            dict(family=c['family'], prediction=scores['prediction_choice'], action=scores['action_choice'], joint=all(scores.values()), truth=c['truth'],
                 prompt_tokens=r.get('response', {}).get('usage', {}).get('prompt_tokens')))
    arms = []
    for arm in ARMS:
        rr = [r for r in scored if r['arm'] == arm]; main = [r for r in rr if r['family'] != 'uncertain']; controls = [r for r in rr if r['family'] == 'uncertain']
        arms.append(dict(arm=arm, valid=sum(r['valid'] for r in rr), total=len(rr),
            main_cases=len(main), prediction=sum(r['prediction'] for r in main), action=sum(r['action'] for r in main), joint=sum(r['joint'] for r in main),
            uncertainty_joint=sum(r['joint'] for r in controls), uncertainty_cases=len(controls), median_seconds=statistics.median(r['seconds'] for r in rr),
            max_prompt_tokens=max((r['prompt_tokens'] or 0) for r in rr),
            families={f: {k: sum(r[k] for r in main if r['family'] == f) for k in ['prediction', 'action', 'joint']} for f in ['counter', 'orbit', 'clock', 'panels']}))
    save_json(out/'summary.json', dict(arms=arms, rows=scored))
    print(json.dumps(arms, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('command', choices=['prepare', 'run', 'analyze']); p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); globals()[a.command](a.output)
