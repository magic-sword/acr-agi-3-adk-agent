"""Finite, grounded conjunction learning from explicit positive/negative transitions.

No arbitrary code, Prolog execution, closed-world negation or causal-proof claims.
"""
from itertools import combinations
import hashlib
import json
import time


def literals(state):
    return {(p, True) for p in state.get('true', [])} | {(p, False) for p in state.get('false', [])}


def state_of(values):
    return {name: sorted(p for p, truth in values if truth == value)
            for name, value in (('true', True), ('false', False))}


def signature(item):
    return (item['action_name'], json.dumps(item['actions'], sort_keys=True),
            item['scope'], item.get('level'))


def outcome(example, effects):
    """An unknown or already-satisfied result is not positive intervention evidence."""
    before, after = literals(example['before']), literals(example['after'])
    if any((p, not v) in after for p, v in effects):
        return 'negative'
    if effects <= after and any((p, not v) in before for p, v in effects):
        return 'positive'
    return 'unknown'


def assess(rule, examples):
    pre, effects = literals(rule['preconditions']), literals(rule['effects'])
    evidence = {'positive': [], 'negative': [], 'unknown': []}
    rule_signature = signature(rule)
    for e in examples:
        if signature(e) != rule_signature or e.get('confounded'):
            continue
        before = literals(e['before'])
        if any((p, not v) in before for p, v in pre):
            continue
        result = outcome(e, effects) if pre <= before else 'unknown'
        evidence[result].append(e['id'])
    status = 'contradicted' if evidence['negative'] else 'supported' if evidence['positive'] else 'candidate'
    return {**rule, 'status': status, 'evidence': {k: v[:16] for k, v in evidence.items()},
            'evidence_counts': {k: len(v) for k, v in evidence.items()},
            'qualification': 'consistent with recorded symbolic interpretations; not proven causation'}


def induce(examples, *, max_preconditions=4, max_candidates=2000, seconds=0.5):
    """Find a smallest shared conjunction separating each observed effect from negatives.

    Restricted propositional ILP: ground symbols only, no invented predicates or
    variable unification. Unknown negative-example conditions cannot exclude an example.
    """
    deadline = time.monotonic() + seconds
    groups = {}
    for e in examples:
        if not e.get('confounded'):
            groups.setdefault(signature(e), []).append(e)
    rules, unresolved = [], []
    examined = 0
    limited = False
    for group in groups.values():
        effects_seen = {frozenset((p, v) for p, v in literals(e['after'])
                                 if (p, not v) in literals(e['before'])) for e in group}
        for effects in sorted(effects_seen, key=lambda v: sorted(v)):
            if time.monotonic() >= deadline:
                return {'rules': rules, 'unresolved': unresolved, 'search_complete': False,
                        'reason': 'induction_budget', 'candidates_examined': examined}
            if not effects:
                continue
            positive = [e for e in group if outcome(e, effects) == 'positive']
            negative = [e for e in group if outcome(e, effects) == 'negative']
            pool = sorted(set.intersection(*(literals(e['before']) for e in positive)))
            if len(pool) > 32:
                unresolved.append({'action_name': group[0]['action_name'], 'reason': 'literal_budget',
                                   'effects': state_of(effects)})
                limited = True
                continue
            # With positive evidence only, retain observed context instead of inventing an unconditional law.
            chosen = tuple(pool) if not negative else None
            for size in ([] if chosen is not None else range(min(len(pool), max_preconditions) + 1)):
                for candidate in combinations(pool, size):
                    examined += 1
                    if examined > max_candidates or time.monotonic() >= deadline:
                        return {'rules': rules, 'unresolved': unresolved, 'search_complete': False,
                                'reason': 'induction_budget', 'candidates_examined': examined - 1}
                    # Every observed negative must explicitly contradict a precondition.
                    if all(any((p, not v) in literals(e['before']) for p, v in candidate) for e in negative):
                        chosen = candidate
                        break
                if chosen is not None:
                    break
            if chosen is None:
                unresolved.append({'action_name': group[0]['action_name'], 'effects': state_of(effects),
                                   'positive_examples': [e['id'] for e in positive],
                                   'negative_examples': [e['id'] for e in negative],
                                   'reason': 'no_separating_conjunction_within_language',
                                   'question': 'Which unobserved condition, mode, target or delay distinguishes these trials?'})
                continue
            rule = {k: group[0][k] for k in ('action_name', 'actions', 'scope', 'level')}
            rule.update(preconditions=state_of(chosen), effects=state_of(effects), origin='induced')
            rule['id'] = 'learned-' + hashlib.sha256(json.dumps(rule, sort_keys=True).encode()).hexdigest()[:20]
            rules.append(rule)
    return {'rules': rules, 'unresolved': unresolved, 'search_complete': not limited,
            'candidates_examined': examined}
