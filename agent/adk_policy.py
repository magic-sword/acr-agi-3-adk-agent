"""Synchronous entrypoints for the persistent ADK 2.0 cognitive workflow."""
from __future__ import annotations

import os
from .cognition.validation import parse_json, validate_action
from .cognition.workflow import CognitiveRuntime


def _parse_model_answer(text: str, observation: dict) -> dict:
    """Compatibility helper for simple action clients; never grants implicit RESET."""
    return validate_action(parse_json(text), observation).model_dump(exclude_none=True)


def create_runtime(game_id: str) -> CognitiveRuntime:
    return CognitiveRuntime(
        game_id, os.getenv("ADK_MODEL") or None,
        max_calls=int(os.getenv("COGNITION_MAX_CALLS", "3")),
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
