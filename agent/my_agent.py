"""ARC framework adapter; policy decisions are made by agent/adk_policy.py."""
from __future__ import annotations

import base64
import io
import os
from typing import Any

from arcengine import FrameData, GameAction, GameState
from agents.agent import Agent

from agent.adk_policy import decide


class MyAgent(Agent):
    # The upstream loop uses <=, so this produces at most 80 actions.
    MAX_ACTIONS = 79

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        return latest_frame.state is GameState.WIN

    def choose_action(self, frames: list[FrameData], latest_frame: FrameData) -> GameAction:
        observation = {
            "game_id": self.game_id,
            "state": latest_frame.state.name,
            "step": self.action_counter,
            "levels_completed": latest_frame.levels_completed,
            "available_actions": [
                getattr(a, "name", str(a)) for a in (latest_frame.available_actions or [])
            ],
        }
        if os.getenv("ADK_MODEL") and latest_frame.frame:
            # A local vision model can receive the rendered observation.
            import numpy as np
            from PIL import Image

            pixels = np.asarray(latest_frame.frame[0], dtype=np.uint8)
            if pixels.ndim == 2:
                image = Image.fromarray(pixels, mode="L")
            elif pixels.ndim == 3 and pixels.shape[-1] in (3, 4):
                image = Image.fromarray(pixels)
            else:
                raise ValueError(f"Unsupported frame shape: {pixels.shape}")
            image = image.resize((512, 512), resample=Image.Resampling.NEAREST)
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            observation["image_png_base64"] = base64.b64encode(buffer.getvalue()).decode("ascii")
        result = decide(observation)
        action = GameAction[result["action"]]
        action.reasoning = {"policy": "adk", "reason": result["reason"]}
        return action
