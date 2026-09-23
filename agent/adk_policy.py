"""Google ADK policy: deterministic fallback or a local vision model.

The default policy uses ADK's BaseAgent and Runner without making model calls.
Set ADK_MODEL=local/qwen3-vl-4b-instruct to use the local vision server.
The competition notebook starts its own server and never calls an outside API.
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

from agent.local_vlm import LocalVisionLlm

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
    if model != "local/qwen3-vl-4b-instruct":
        raise ValueError(f"Unsupported ADK_MODEL: {model}. Use local/qwen3-vl-4b-instruct")
    return LlmAgent(
        name="arc_model_policy",
        model=LocalVisionLlm(
            model="qwen3-vl-4b-instruct",
            api_base=os.getenv("VLM_API_BASE", "http://vlm:8080/v1"),
        ),
        instruction=(
            "You play an unknown visual puzzle game. Compare the image with "
            "the recent action history and look for the game's goal, objects, "
            "and controls. Explore when the goal is unclear. Read the JSON "
            "observation for legal actions and progress. Reply with exactly "
            "one short JSON object containing "
            "action (one of the available ACTION1..ACTION7, or RESET only to restart) "
            "and reason. For ACTION6 (point/click), include integer x and y "
            "coordinates in [0,63]. Use the exact ACTION name, not its numeric ID. "
            "Never invent action names. Avoid repeating actions without progress."
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
    raw = answer.get("action")
    if isinstance(raw, int) and not isinstance(raw, bool):
        raw = str(raw)
    if not isinstance(raw, str):
        raise ValueError(f"Invalid model action: {raw!r}; allowed: {sorted(allowed)}")
    name = raw.strip().upper()
    if name.isdecimal():
        name = "RESET" if int(name) == 0 else f"ACTION{int(name)}"
    if name not in allowed:
        raise ValueError(f"Invalid model action: {raw!r}; allowed: {sorted(allowed)}")
    result = {"action": name, "reason": str(answer.get("reason", "model"))[:200]}
    if name == "ACTION6":
        for coordinate in ("x", "y"):
            value = answer.get(coordinate)
            if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 63:
                raise ValueError(f"ACTION6 requires integer {coordinate} in [0,63]: {value!r}")
            result[coordinate] = value
    return result


async def _decide_async(observation: dict[str, Any], model: str | None) -> dict[str, Any]:
    if observation["state"] in ("NOT_PLAYED", "GAME_OVER"):
        return baseline_action(observation)
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
