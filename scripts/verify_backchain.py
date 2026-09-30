"""V6: can Qwen chain skills backward from a goal, and which input format does it need?

Each task has a symbolic state, goal predicates, and grounded skills (preconditions and
effects). The program's breadth-first planner gives ground truth; the model's chain is
scored by simulating it, so any valid plan counts. Tasks whose goal no skill can reach
expect the model to name the missing link and probe an untested item instead.

Formats
  F1  natural language: scene sentences and skill descriptions (like understand + L1 today)
  F2  structured: state facts, goal predicates, skill cards {requires, effect, evidence}
  F3  F2 plus program-computed unsatisfied goals and the skills whose effects touch them

usage (dev container): python scripts/verify_backchain.py
"""
from collections import deque
import json
import os
from pathlib import Path
import time
import urllib.request

API = os.getenv('VLM_API_BASE', 'http://vlm:8080/v1') + '/chat/completions'
OUT = Path(__file__).resolve().parents[1] / 'outputs/v6-backchain'
SAMPLES = [0.0, 0.8, 0.8]


def op(ident, nl, requires, effect, pre, eff, evidence='measured 6/6'):
    return dict(id=ident, nl=nl, card=dict(id=ident, requires=requires, effect=effect, evidence=evidence), pre=pre, eff=eff)


def upd(s, **kw):
    return {**s, **{k.replace('__', '.'): v for k, v in kw.items()}}


TASKS = []

# T1 (from vc33): repeat count from a measured shift.
TASKS.append(dict(id='T1-repeat-shift', kind='chain',
    state={'x(yellow marker)': 50, 'x(yellow segment)': 38},
    goal=[('x(yellow marker) == x(yellow segment)', lambda s: s['x(yellow marker)'] == s['x(yellow segment)'])],
    nl_state='A yellow marker is at column 50. A yellow segment in the black bar is at column 38.',
    nl_goal='The yellow marker is in the same column as the yellow segment.',
    ops=[op('S1', 'Clicking the lower blue button moves the yellow marker 4 columns left (measured 6/6).', [],
            ['x(yellow marker) -= 4'], lambda s: True, lambda s: upd(s, **{'x(yellow marker)': s['x(yellow marker)']-4})),
         op('S2', 'Clicking the upper blue button moves the yellow marker 4 columns right (measured 5/5).', [],
            ['x(yellow marker) += 4'], lambda s: True, lambda s: upd(s, **{'x(yellow marker)': s['x(yellow marker)']+4}))],
    untested=[]))

# T2 (from ls20): attribute must match, no measured skill changes it -> missing link, probe the untested object.
TASKS.append(dict(id='T2-missing-attribute', kind='missing',
    state={'at(block)': 'start', 'shape(block)': 'L', 'shape(target mark)': 'T'},
    goal=[('at(block) == framed box', lambda s: s['at(block)'] == 'framed box'),
          ('shape(block) == shape(target mark)', lambda s: s['shape(block)'] == s['shape(target mark)'])],
    nl_state='The orange-blue block has an L shape. The framed box at the top holds a T-shaped mark. A white cross lies to the left.',
    nl_goal='The block is inside the framed box and its shape equals the mark in the box.',
    ops=[op('S1', 'Direction keys move the orange-blue block to a reachable object (the program can execute it; 30/31 predictions).',
            ['reachable(X)'], ['at(block) = X'], lambda s: True, None)],
    move_targets=['framed box', 'white cross'],
    untested=[dict(id='U1', what='the white cross (never touched)'), dict(id='U2', what='the ACT button (never pressed)')],
    expected_probe={'U1', 'U2'}))

# T3: same as T2 but the shapes already match -> direct move.
TASKS.append(dict(id='T3-attribute-already-met', kind='chain',
    state={'at(block)': 'start', 'shape(block)': 'T', 'shape(target mark)': 'T'},
    goal=TASKS[-1]['goal'],
    nl_state='The orange-blue block has a T shape. The framed box at the top holds a T-shaped mark. A white cross lies to the left.',
    nl_goal=TASKS[-1]['nl_goal'], ops=TASKS[-1]['ops'], move_targets=['framed box', 'white cross'],
    untested=[dict(id='U1', what='the white cross (never touched)')]))

# T4: key and door, a three-link chain.
TASKS.append(dict(id='T4-key-door', kind='chain',
    state={'at(player)': 'start', 'has(key)': False, 'door': 'closed'},
    goal=[('at(player) == exit', lambda s: s['at(player)'] == 'exit')],
    nl_state='The player stands at the start. A key lies in the corner. A closed door blocks the corridor to the exit.',
    nl_goal='The player reaches the exit.',
    ops=[op('S1', 'Moving the player to the key picks it up (measured 2/2).', [], ['at(player) = key', 'has(key) = true'],
            lambda s: True, lambda s: upd(s, **{'at(player)': 'key', 'has(key)': True})),
         op('S2', 'Moving the player to the door opens it when the player holds the key (measured 1/1); without the key it does not open.',
            ['has(key)'], ['at(player) = door', 'door = open'],
            lambda s: s['has(key)'], lambda s: upd(s, **{'at(player)': 'door', 'door': 'open'})),
         op('S3', 'Moving the player to the exit works once the door is open (measured 1/1).', ['door == open'],
            ['at(player) = exit'], lambda s: s['door'] == 'open', lambda s: upd(s, **{'at(player)': 'exit'}))],
    untested=[]))

# T5 / T6 (ft09-like): colour cycles blue -> red -> green -> blue on each click.
CYCLE = {'blue': 'red', 'red': 'green', 'green': 'blue'}
for ident, start in (('T5-cycle-once', 'blue'), ('T6-cycle-twice', 'green')):
    TASKS.append(dict(id=ident, kind='chain',
        state={'color(tile A)': start, 'color(tile B)': 'blue'},
        goal=[('color(tile A) == red', lambda s: s['color(tile A)'] == 'red'),
              ('color(tile B) == blue', lambda s: s['color(tile B)'] == 'blue')],
        nl_state=f'Tile A is {start}. Tile B is blue.',
        nl_goal='Tile A is red and tile B is blue.',
        ops=[op('S1', 'Clicking tile A changes its colour blue->red->green->blue (measured 4/4).', [],
                ['color(tile A) = next(blue->red->green->blue)'], lambda s: True,
                lambda s: upd(s, **{'color(tile A)': CYCLE[s['color(tile A)']]})),
             op('S2', 'Clicking tile B changes its colour blue->red->green->blue (measured 3/3).', [],
                ['color(tile B) = next(blue->red->green->blue)'], lambda s: True,
                lambda s: upd(s, **{'color(tile B)': CYCLE[s['color(tile B)']]}))],
        untested=[]))

# T7: no skill touches the goal and no object is untested except a control -> probe the control.
TASKS.append(dict(id='T7-missing-untried-control', kind='missing',
    state={'color(lamp)': 'gray', 'at(player)': 'start'},
    goal=[('color(lamp) == yellow', lambda s: s['color(lamp)'] == 'yellow')],
    nl_state='A gray lamp hangs on the wall. The player stands at the start.',
    nl_goal='The lamp is yellow.',
    ops=[op('S1', 'Direction keys move the player to a reachable object (measured 12/12).', ['reachable(X)'],
            ['at(player) = X'], lambda s: True, None)],
    move_targets=['lamp'],
    untested=[dict(id='U1', what='the ACT button (never pressed)')], expected_probe={'U1'}))

# T8: push a box twice after walking behind it; distractor skills present.
TASKS.append(dict(id='T8-push-with-distractors', kind='chain',
    state={'at(player)': 'start', 'x(box)': 3, 'x(pad)': 5},
    goal=[('x(box) == x(pad)', lambda s: s['x(box)'] == s['x(pad)'])],
    nl_state='A box is at column 3 and a pad at column 5 in the same row. The player is elsewhere. There is also a lever and a red button.',
    nl_goal='The box is on the pad.',
    ops=[op('S1', 'Moving the player to the left side of the box (measured 3/3).', [], ['at(player) = left of box'],
            lambda s: True, lambda s: upd(s, **{'at(player)': 'left of box'})),
         op('S2', 'Pressing RIGHT while the player is on the left side of the box pushes the box 1 column right (measured 4/4).',
            ['at(player) == left of box'], ['x(box) += 1'], lambda s: s['at(player)'] == 'left of box',
            lambda s: upd(s, **{'x(box)': s['x(box)']+1})),
         op('S3', 'Pulling the lever turns the ceiling light on (measured 2/2).', [], ['light = on'],
            lambda s: True, lambda s: upd(s, light='on')),
         op('S4', 'Clicking the red button makes a sound; nothing moves (measured 3/3).', [], [],
            lambda s: True, lambda s: s)],
    untested=[]))

# T9: vertical alignment by repeated moves in one direction.
TASKS.append(dict(id='T9-align-rows', kind='chain',
    state={'y(piece)': 10, 'y(slot)': 4},
    goal=[('y(piece) == y(slot)', lambda s: s['y(piece)'] == s['y(slot)'])],
    nl_state='A piece is at row 10. A slot is at row 4.',
    nl_goal='The piece is in the same row as the slot.',
    ops=[op('S1', 'UP moves the piece 2 rows up (row number decreases by 2; measured 5/5).', [], ['y(piece) -= 2'],
            lambda s: True, lambda s: upd(s, **{'y(piece)': s['y(piece)']-2})),
         op('S2', 'DOWN moves the piece 2 rows down (measured 5/5).', [], ['y(piece) += 2'],
            lambda s: True, lambda s: upd(s, **{'y(piece)': s['y(piece)']+2}))],
    untested=[]))

# T10: one conjunct already true; only the other needs work.
TASKS.append(dict(id='T10-partly-satisfied', kind='chain',
    state={'at(player)': 'start', 'color(target)': 'green'},
    goal=[('at(player) == target', lambda s: s['at(player)'] == 'target'),
          ('color(target) == green', lambda s: s['color(target)'] == 'green')],
    nl_state='The player is at the start. The target square is green.',
    nl_goal='The player stands on the target and the target is green.',
    ops=[op('S1', 'Direction keys move the player to a reachable object (measured 9/9).', ['reachable(X)'],
            ['at(player) = X'], lambda s: True, None),
         op('S2', 'Clicking the target cycles its colour green->blue->green (measured 2/2).', [],
            ['color(target) = next(green->blue->green)'], lambda s: True,
            lambda s: upd(s, **{'color(target)': 'blue' if s['color(target)'] == 'green' else 'green'}))],
    move_targets=['target'], untested=[]))


def ground(task):
    """Expand parameterised move skills into one grounded step per destination."""
    steps = []
    for o in task['ops']:
        if o['eff'] is None:
            mover = o['card']['effect'][0].split(')')[0] + ')'
            for t in task.get('move_targets', []):
                steps.append((o['id'], t, (lambda s, o=o: True), (lambda s, m=mover, t=t: upd(s, **{m: t}))))
        else:
            steps.append((o['id'], None, o['pre'], o['eff']))
    return steps


def satisfied(task, s):
    return all(f(s) for _, f in task['goal'])


def plan(task, depth=8):
    steps = ground(task)
    start = tuple(sorted(task['state'].items()))
    prev, queue = {start: None}, deque([start])
    while queue:
        key = queue.popleft()
        s = dict(key)
        if satisfied(task, s):
            out = []
            while prev[key] is not None:
                key, step = prev[key]
                out.append(step)
            return out[::-1]
        for ident, arg, pre, eff in steps:
            if pre(s):
                n = tuple(sorted(eff(s).items()))
                if n not in prev and len(prev) < 5000:
                    prev[n] = (key, (ident, arg))
                    queue.append(n)
    return None


def simulate(task, chain):
    steps = {(i, a): (pre, eff) for i, a, pre, eff in ground(task)}
    s = dict(task['state'])
    for item in chain:
        key = (item.get('skill'), item.get('argument') or None)
        if key not in steps:
            key = (item.get('skill'), None)
        if key not in steps:
            return False, f"unknown step {item}"
        pre, eff = steps[key]
        for _ in range(max(1, int(item.get('repeat') or 1))):
            if not pre(s):
                return False, f"precondition failed at {item}"
            s = eff(s)
    return satisfied(task, s), s


def prompt(task, fmt):
    moves = task.get('move_targets')
    if fmt == 'F1':
        skills = '\n'.join(f"{o['id']}: {o['nl']}" for o in task['ops'])
        body = f"Scene: {task['nl_state']}\nGoal: {task['nl_goal']}\nSkills:\n{skills}"
    else:
        data = dict(state=[f'{k} = {v}' for k, v in task['state'].items()],
                    goal=[g for g, _ in task['goal']], skills=[o['card'] for o in task['ops']])
        if fmt == 'F3':
            s = task['state']
            data['unsatisfied_goals'] = [g for g, f in task['goal'] if not f(s)]
            touched = {}
            for g in data['unsatisfied_goals']:
                subject = g.split(' ')[0].split('(')[0]
                touched[g] = [o['id'] for o in task['ops'] if any(e.split(' ')[0].split('(')[0] == subject for e in o['card']['effect'])]
            data['skills_affecting_unsatisfied'] = touched
        body = json.dumps(data, indent=1)
    if moves:
        body += f"\nMove destinations (argument for the move skill): {moves}"
    if task['untested']:
        body += '\nUntested items: ' + '; '.join(f"{u['id']}: {u['what']}" for u in task['untested'])
    return body


SCHEMA = {'type': 'object', 'properties': {
    'unsatisfied': {'type': 'array', 'items': {'type': 'string', 'maxLength': 120}, 'maxItems': 4},
    'chain': {'type': 'array', 'maxItems': 6, 'items': {'type': 'object', 'properties': {
        'skill': {'type': 'string', 'maxLength': 8}, 'argument': {'type': ['string', 'null'], 'maxLength': 40},
        'repeat': {'type': 'integer', 'minimum': 1, 'maximum': 12}}, 'required': ['skill', 'argument', 'repeat']}},
    'missing_link': {'type': ['string', 'null'], 'maxLength': 160},
    'probe': {'type': ['string', 'null'], 'maxLength': 8}},
    'required': ['unsatisfied', 'chain', 'missing_link', 'probe']}

INSTRUCTION = ('Work backwards from the goal. List the goal conditions not yet true. For each, find a skill whose '
               'effect makes it true; if that skill has requirements that are not true, chain further back to a skill '
               'that makes them true. Return the chain in execution order (skill id, argument for a move skill, '
               'repeat count). If some condition cannot be reached by any skill, set missing_link to that condition, '
               'keep the chain empty, and set probe to the untested item id most likely to reveal it. Otherwise set '
               'missing_link and probe to null.')


def ask(body, temperature, seed):
    payload = {'model': 'qwen3-vl-4b-instruct', 'temperature': temperature, 'seed': seed, 'max_tokens': 500,
               'response_format': {'type': 'json_schema', 'json_schema': {'name': 'backchain', 'schema': SCHEMA}},
               'messages': [{'role': 'system', 'content': INSTRUCTION}, {'role': 'user', 'content': body}]}
    started = time.monotonic()
    r = json.loads(urllib.request.urlopen(urllib.request.Request(API, json.dumps(payload).encode(),
                                                                 {'Content-Type': 'application/json'}), timeout=300).read())
    return json.loads(r['choices'][0]['message']['content']), r.get('usage', {}), time.monotonic()-started


def score(task, answer):
    probe = (answer['probe'] or '').replace(':', ' ').split(' ')[0] or None  # "U1: the cross" names U1
    if task['kind'] == 'missing':
        return dict(correct=bool(answer['missing_link']) and not answer['chain'] and probe in task['expected_probe'],
                    flagged_missing=bool(answer['missing_link']), probe_ok=probe in task['expected_probe'])
    ok, detail = simulate(task, answer['chain'])
    return dict(correct=ok and not answer['missing_link'], flagged_missing=bool(answer['missing_link']),
                chain_valid=ok, detail=str(detail)[:120])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for task in TASKS:
        truth = plan(task)
        assert (truth is None) == (task['kind'] == 'missing'), task['id']
        for fmt in ('F1', 'F2', 'F3'):
            body = prompt(task, fmt)
            for i, t in enumerate(SAMPLES):
                answer, usage, seconds = ask(body, t, i)
                rows.append(dict(task=task['id'], format=fmt, temperature=t, truth=truth, answer=answer,
                                 prompt_tokens=usage.get('prompt_tokens'), completion_tokens=usage.get('completion_tokens'),
                                 seconds=round(seconds, 2), **score(task, answer)))
                print(json.dumps(rows[-1])[:300])
    (OUT / 'trials.json').write_text(json.dumps(rows, indent=1))
    print('\n|format|correct (all)|chain tasks correct|missing-link tasks correct|false missing-link claims|prompt tokens (mean)|seconds (mean)|')
    print('|---|---:|---:|---:|---:|---:|---:|')
    for fmt in ('F1', 'F2', 'F3'):
        rs = [r for r in rows if r['format'] == fmt]
        chain = [r for r in rs if TASK_KIND[r['task']] == 'chain']
        miss = [r for r in rs if TASK_KIND[r['task']] == 'missing']
        print(f"|{fmt}|{sum(r['correct'] for r in rs)}/{len(rs)}|{sum(r['correct'] for r in chain)}/{len(chain)}|"
              f"{sum(r['correct'] for r in miss)}/{len(miss)}|{sum(r['flagged_missing'] for r in chain)}/{len(chain)}|"
              f"{sum(r['prompt_tokens'] or 0 for r in rs)/len(rs):.0f}|{sum(r['seconds'] for r in rs)/len(rs):.1f}|")
    print('\n|task|F1|F2|F3|')
    print('|---|---|---|---|')
    for task in TASKS:
        cells = [f"{sum(r['correct'] for r in rows if r['task'] == task['id'] and r['format'] == f)}/{len(SAMPLES)}"
                 for f in ('F1', 'F2', 'F3')]
        print(f"|{task['id']}|" + '|'.join(cells) + '|')


TASK_KIND = {t['id']: t['kind'] for t in TASKS}

def rescore():
    rows = json.loads((OUT / 'trials.json').read_text())
    by_id = {t['id']: t for t in TASKS}
    for r in rows:
        r.update(score(by_id[r['task']], r['answer']))
    (OUT / 'trials.json').write_text(json.dumps(rows, indent=1))
    return rows


def report(rows):
    print('|format|correct (all)|chain tasks correct|missing-link tasks correct|probe item correct (missing tasks)|false missing-link claims|prompt tokens (mean)|seconds (mean)|')
    print('|---|---:|---:|---:|---:|---:|---:|---:|')
    for fmt in ('F1', 'F2', 'F3'):
        rs = [r for r in rows if r['format'] == fmt]
        chain = [r for r in rs if TASK_KIND[r['task']] == 'chain']
        miss = [r for r in rs if TASK_KIND[r['task']] == 'missing']
        print(f"|{fmt}|{sum(r['correct'] for r in rs)}/{len(rs)}|{sum(r['correct'] for r in chain)}/{len(chain)}|"
              f"{sum(r['correct'] for r in miss)}/{len(miss)}|{sum(r.get('probe_ok', False) for r in miss)}/{len(miss)}|"
              f"{sum(r['flagged_missing'] for r in chain)}/{len(chain)}|"
              f"{sum(r['prompt_tokens'] or 0 for r in rs)/len(rs):.0f}|{sum(r['seconds'] for r in rs)/len(rs):.1f}|")
    print('\n|task|F1|F2|F3|')
    print('|---|---|---|---|')
    for task in TASKS:
        cells = [f"{sum(r['correct'] for r in rows if r['task'] == task['id'] and r['format'] == f)}/{len(SAMPLES)}"
                 for f in ('F1', 'F2', 'F3')]
        print(f"|{task['id']}|" + '|'.join(cells) + '|')


if __name__ == '__main__':
    import sys
    if sys.argv[1:] == ['rescore']:
        report(rescore())
    else:
        main()
