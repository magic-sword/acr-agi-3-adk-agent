"""Synchronous entrypoints for the persistent ADK 2.0 cognitive workflow."""
from __future__ import annotations

import os
from .cognition.workflow import CognitiveRuntime


def create_runtime(game_id: str) -> CognitiveRuntime:
    return CognitiveRuntime(
        game_id, os.getenv("ADK_MODEL") or None,
        max_calls=int(os.getenv("COGNITION_MAX_CALLS", "4")),
        max_http_requests=int(os.getenv("COGNITION_MAX_HTTP_REQUESTS", "8")),
        max_resets=int(os.getenv("COGNITION_MAX_RESETS", "2")),
        seconds=float(os.getenv("COGNITION_SECONDS", "600")),
        log_dir=os.getenv("COGNITION_LOG_DIR") or None,
        learning=os.getenv("COGNITION_LEARNING", "1") == "1",
        skill_library=os.getenv("COGNITION_SKILL_LIBRARY") or None,
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
