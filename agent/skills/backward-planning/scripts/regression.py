"""Bounded goal regression over grounded actions with explicit true/false effects."""
from collections import deque
import time


def literals(state):
    return {(p, True) for p in state.get('true', [])} | {(p, False) for p in state.get('false', [])}


def state_of(values):
    return {name: sorted(p for p, truth in values if truth == value)
            for name, value in (('true', True), ('false', False))}


def consistent(values):
    return not any((p, not v) in values for p, v in values)


def backward_plan(current, goal, rules, *, allow_candidates=False, max_depth=12,
                  max_nodes=2000, seconds=0.5):
    initial, target = literals(current), frozenset(literals(goal))
    eligible, excluded = [], []
    for rule in rules:
        if rule['status'] == 'supported' or (allow_candidates and rule['status'] == 'candidate'):
            eligible.append(rule)
        else:
            excluded.append({'rule_id': rule['id'], 'reason': rule['status'],
                             'preconditions': rule['preconditions'], 'effects': rule['effects']})
    # Do not pick the favorable outcome of competing models for the same concrete input.
    ambiguous = set()
    for i, a in enumerate(eligible):
        for b in eligible[i + 1:]:
            if (a['actions'] == b['actions'] and a['effects'] != b['effects']
                    and consistent(literals(a['preconditions']) | literals(b['preconditions']))):
                ambiguous.update((a['id'], b['id']))
    excluded += [{'rule_id': r['id'], 'reason': 'overlapping_action_models',
                  'preconditions': r['preconditions'], 'effects': r['effects']}
                 for r in eligible if r['id'] in ambiguous]
    eligible = [r for r in eligible if r['id'] not in ambiguous]
    prepared = [(r, literals(r['preconditions']), literals(r['effects'])) for r in eligible]
    queue = deque([(target, [])])
    visited = {target}
    frontier = []
    expanded, limited = 0, False
    deadline = time.monotonic() + seconds
    while queue:
        if expanded >= max_nodes or time.monotonic() >= deadline:
            limited = True
            break
        required, suffix = queue.popleft()
        expanded += 1
        if required <= initial:
            state, proof = set(initial), []
            for r in suffix:
                pre, effect = literals(r['preconditions']), literals(r['effects'])
                if not pre <= state:
                    raise AssertionError('regression returned a plan with unmet preconditions')
                before = state_of(state)
                state.difference_update((p, not v) for p, v in effect)
                state.update(effect)
                proof.append({'rule_id': r['id'], 'action_name': r['action_name'], 'actions': r['actions'],
                              'before': before, 'after': state_of(state), 'status': r['status'],
                              'evidence': r.get('evidence', {})})
            if not target <= state:
                raise AssertionError('regression plan failed forward validation')
            return {'status': 'tentative_plan' if any(r['status'] != 'supported' for r in suffix) else 'plan',
                    'plan': proof, 'next_action': suffix[0]['actions'][0] if suffix else None,
                    'expanded_nodes': expanded, 'excluded_rules': excluded,
                    'qualification': 'valid under recorded models and frame assumptions; execute one step then observe'}
        missing = set(required) - initial
        conflicts, predecessors = [], []
        for r, pre, effects in prepared:
            if not effects & required:
                continue
            destroyed = {(p, v) for p, v in required if (p, not v) in effects}
            previous = (set(required) - effects) | pre
            if destroyed or not consistent(previous):
                conflicts.append({'rule_id': r['id'], 'reason': 'conflicting_effect_or_precondition',
                                  'destroyed_requirements': state_of(destroyed)})
                continue
            predecessors.append((frozenset(previous), [r] + suffix))
        unvisited = [(p, path) for p, path in predecessors if p not in visited]
        if len(suffix) >= max_depth and unvisited:
            limited = True
        elif predecessors:
            for previous, path in predecessors:
                if previous not in visited:
                    if len(visited) >= max_nodes:
                        limited = True
                        continue
                    visited.add(previous)
                    queue.append((previous, path))
        # Keep leaf/cycle frontiers as conditional questions, not new facts.
        if not unvisited or len(suffix) >= max_depth:
            producers = set().union(*(effects for _, _, effects in prepared)) if prepared else set()
            frontier.append({'required': state_of(required), 'missing': state_of(missing),
                             'unknown_current': state_of({(p, v) for p, v in missing
                                                          if (p, not v) not in initial}),
                             'no_known_producer': state_of(missing - producers),
                             'suffix_rule_ids': [r['id'] for r in suffix], 'conflicts': conflicts})
    # Prefer frontiers reached by a useful chain over simply restating the original goal.
    frontier.sort(key=lambda f: (len(literals(f['missing'])), -len(f['suffix_rule_ids'])))
    return {'status': 'search_limit' if limited else 'knowledge_gap', 'plan': [],
            'expanded_nodes': expanded, 'frontier': frontier[:8], 'excluded_rules': excluded,
            'search_complete': not limited,
            'qualification': ('search was truncated; a plan or missing rule is not established' if limited else
                              'no plan in the supplied finite model; gaps are experiment/observation questions, not proof of impossibility')}
