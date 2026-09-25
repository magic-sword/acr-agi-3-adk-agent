"""Deterministic routing; no prose classification or LLM orchestration."""


def after_goal(decision):
    return {'continue': 'design_experiment', 'completed': 'select_goal',
            'replace': 'select_goal', 'deferred': 'stop'}[decision]


def recovery_route(reason, target_checked, attempts, limit=2):
    if attempts > limit:
        return 'stop'
    if reason == 'same_test_unchanged_conditions' and not target_checked:
        return 'inspect_target'
    if reason == 'target_mismatch':
        return 'design_experiment'
    return 'assess_goal'
