"""An offline Google ADK policy with an optional local model backend.

The default policy uses ADK's BaseAgent and Runner without making model calls.
Set ADK_MODEL to a LiteLLM model identifier to try a locally served model.
No external API calls are needed for the default Kaggle submission.
"""
from __future__ import annotations

import asyncio
import base64
import json
import os
import re
from typing import Any

from google.adk.agents import BaseAgent, LlmAgent
from google.adk.events import Event
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

APP_NAME = "arc_agi_3_adk"
_USER = "player"
_ACTIONS = ("ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5")


def baseline_action(observation: dict[str, Any]) -> dict[str, Any]:
    """Return a bounded, repeatable action from one game observation."""
    if observation["state"] in ("NOT_PLAYED", "GAME_OVER"):
        return {"action": "RESET", "reason": "start or restart the game"}
    allowed = [a for a in observation.get("available_actions", []) if a in _ACTIONS]
    if not allowed:
        allowed = list(_ACTIONS)
    step = int(observation["step"])
    # Fixed game-specific offset; unlike Python's hash(), stable across processes.
    offset = sum(ord(c) for c in observation["game_id"]) % len(allowed)
    return {"action": allowed[(step + offset) % len(allowed)], "reason": "offline baseline"}


class OfflinePolicy(BaseAgent):
    """Small ADK agent. Replace this method with a stronger policy later."""

    async def _run_async_impl(self, ctx):
        observation = ctx.session.state["observation"]
        answer = baseline_action(observation)
        yield Event(
            invocation_id=ctx.invocation_id,
            author=self.name,
            content=types.Content(role="model", parts=[types.Part(text=json.dumps(answer))]),
        )


def _model_agent(model: str) -> LlmAgent:
    # Keep imports optional: the offline ADK baseline does not require LiteLLM.
    from google.adk.models.lite_llm import LiteLlm

    return LlmAgent(
        name="arc_model_policy",
        model=LiteLlm(model=model),
        instruction=(
            "You select ARC-AGI-3 game actions. Read the user message's JSON "
            "observation. Reply with exactly one JSON object containing "
            "action (an allowed ACTION1..ACTION5 or RESET) and reason. "
            "On NOT_PLAYED or GAME_OVER choose RESET. Never invent action names."
        ),
    )


def _parse_model_answer(text: str, observation: dict[str, Any]) -> dict[str, Any]:
    # Some local models surround JSON with prose/fences; bound the extraction.
    match = re.search(r"\{[^{}]{1,1000}\}", text)
    if match is None:
        raise ValueError(f"Model did not return an action object: {text[:160]!r}")
    answer = json.loads(match.group())
    allowed = set(observation.get("available_actions") or _ACTIONS)
    allowed.add("RESET")
    if answer.get("action") not in allowed:
        raise ValueError(f"Invalid model action: {answer.get('action')!r}")
    return {"action": answer["action"], "reason": str(answer.get("reason", "model"))[:200]}


async def _decide_async(observation: dict[str, Any], model: str | None) -> dict[str, Any]:
    if model and not os.getenv("OPENAI_API_BASE") and model.startswith("openai/"):
        raise ValueError("Set OPENAI_API_BASE to your locally served model URL")
    agent = _model_agent(model) if model else OfflinePolicy(name="arc_offline_policy")
    service = InMemorySessionService()
    session_id = "single_step"
    await service.create_session(
        app_name=APP_NAME,
        user_id=_USER,
        session_id=session_id,
        state={"observation": observation},
    )
    runner = Runner(agent=agent, app_name=APP_NAME, session_service=service)
    summary = {key: value for key, value in observation.items() if key != "image_png_base64"}
    parts = [types.Part(text=json.dumps(summary))]
    if model and observation.get("image_png_base64"):
        parts.append(types.Part.from_bytes(
            data=base64.b64decode(observation["image_png_base64"]), mime_type="image/png"
        ))
    message = types.Content(role="user", parts=parts)
    response = None
    async for event in runner.run_async(user_id=_USER, session_id=session_id, new_message=message):
        if event.is_final_response() and event.content:
            response = "".join(p.text or "" for p in event.content.parts or [])
    if response is None:
        raise RuntimeError("ADK agent returned no final response")
    if model:
        return _parse_model_answer(response, observation)
    return json.loads(response)


def decide(observation: dict[str, Any]) -> dict[str, Any]:
    """Synchronous adapter for the ARC framework's choose_action method."""
    model = os.getenv("ADK_MODEL") or None
    return asyncio.run(_decide_async(observation, model))
