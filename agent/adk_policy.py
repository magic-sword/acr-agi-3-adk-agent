"""Synchronous entrypoints for the persistent ADK 2.0 cognitive workflow."""
from __future__ import annotations

import os
from .cognition.workflow import CognitiveRuntime
from .cognition.focused_workflow import FocusedRuntime
from .cognition.goal_runtime import GoalRuntime


def create_runtime(game_id: str, *, candidate_generator=None) -> CognitiveRuntime:
    retired = ('COGNITION_PLANNING_COMPARISON', 'COGNITION_MEMORY_COMPARISON', 'COGNITION_ACTION_UPDATE_COMPARISON')
    if any(os.getenv(key) for key in retired):
        raise ValueError('Legacy comparison runtimes were removed; unset COGNITION_*_COMPARISON')
    # goal: hypotheses + program backward chaining (docs/backchain-review-ja.md §6.4);
    # stages: the per-step understand/backchain/candidates/ground/reconcile pipeline.
    pipeline = os.getenv("COGNITION_PIPELINE", "goal")
    if pipeline not in ("goal", "stages"):
        raise ValueError("COGNITION_PIPELINE must be goal or stages")
    runtime_class = GoalRuntime if pipeline == "goal" else FocusedRuntime
    return runtime_class(
        game_id, os.getenv("ADK_MODEL") or None, candidate_generator=candidate_generator,
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
