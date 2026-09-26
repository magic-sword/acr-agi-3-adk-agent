"""Deterministic routing; no prose classification or LLM orchestration."""

from .machine import destination

def after_goal(decision):
    return destination({'continue':'goal_continue','completed':'goal_replace',
                        'replace':'goal_replace','deferred':'goal_deferred'}[decision])


def recovery_route(reason, attempts, limit=2):
    if attempts > limit:
        return destination('recover_exhausted')
    if reason == 'target_mismatch':
        return destination('recover_target')
    return destination('recover_goal')
