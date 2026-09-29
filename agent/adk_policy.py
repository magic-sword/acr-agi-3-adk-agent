"""Synchronous entrypoints for the persistent ADK 2.0 cognitive workflow."""
from __future__ import annotations

import os
from .cognition.workflow import CognitiveRuntime
from .cognition.focused_workflow import FocusedRuntime
from .cognition.simple_workflow import SimpleRuntime


def create_runtime(game_id: str, *, candidate_generator=None) -> CognitiveRuntime:
    variant = os.getenv('COGNITION_PLANNING_COMPARISON', '')
    runtime = SimpleRuntime if candidate_generator is None else FocusedRuntime
    if variant:
        from .cognition.planning_comparison import ComparisonRuntime, OneActionRuntime
        if variant not in ('A', 'B'):
            raise ValueError('COGNITION_PLANNING_COMPARISON must be A or B')
        runtime = ComparisonRuntime if variant == 'A' else OneActionRuntime
    memory_variant = os.getenv('COGNITION_MEMORY_COMPARISON', '')
    if memory_variant:
        if variant != 'B' or memory_variant not in ('mixed', 'separated'):
            raise ValueError('COGNITION_MEMORY_COMPARISON requires planning B and mixed/separated')
        from .cognition.memory_comparison import MemoryComparisonRuntime, SeparatedMemoryRuntime
        runtime = MemoryComparisonRuntime if memory_variant == 'mixed' else SeparatedMemoryRuntime
    update_variant = os.getenv('COGNITION_ACTION_UPDATE_COMPARISON', '')
    if update_variant:
        if variant != 'B' or memory_variant != 'separated' or update_variant not in ('A', 'B', 'C', 'D'):
            raise ValueError('COGNITION_ACTION_UPDATE_COMPARISON requires planning B, separated memory and A/B/C/D')
        from .cognition.action_update_comparison import ActionUpdateRuntime
        runtime = ActionUpdateRuntime
    if candidate_generator is not None and runtime is not FocusedRuntime:
        raise ValueError("candidate_generator is supported by the default focused workflow only")
    options = {"candidate_generator": candidate_generator} if runtime is FocusedRuntime else {}
    return runtime(
        game_id, os.getenv("ADK_MODEL") or None, **options,
        repair_attempts=int(os.getenv("COGNITION_REPAIR_ATTEMPTS", "1")),
        decision_seconds=float(os.getenv("COGNITION_DECISION_SECONDS", "45")),
        max_resets=int(os.getenv("COGNITION_MAX_RESETS", "2")),
        seconds=float(os.getenv("COGNITION_SECONDS", "600")),
        log_dir=os.getenv("COGNITION_LOG_DIR") or None,
    )


def decide(observation: dict, runtime: CognitiveRuntime | None = None) -> dict:
    """Pass a game-owned runtime to retain memory. One-shot use remains supported."""
    if runtime is not None:
        return runtime.decide(observation)
    runtime = create_runtime(observation["game_id"])
    try:
        return runtime.decide(observation)
    finally:
        runtime.close()
