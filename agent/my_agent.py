"""ARC adapter: actual observations and exactly one grounded action per iteration."""
from __future__ import annotations

import base64
import io
import logging
import os
import time
from typing import Any

from arcengine import FrameData, GameAction, GameState
from agents.agent import Agent
from agents.tracing import trace_agent_session

from agent.adk_policy import create_runtime


class PolicyStopped(RuntimeError):
    """Control signal to the driver, never a fabricated GameAction."""


def available_action_names(actions: list[Any] | None) -> list[str]:
    names = []
    for raw in actions or []:
        if isinstance(raw, GameAction):
            action = raw
        elif isinstance(raw, int) and not isinstance(raw, bool):
            action = GameAction.from_id(raw)
        elif isinstance(raw, str) and raw.strip().isdigit():
            action = GameAction.from_id(int(raw.strip()))
        else:
            action = GameAction[str(raw).strip().upper()]
        names.append(action.name)
    return names


def frame_observation(game_id: str, frame: FrameData, step: int, remaining: int,
                      *, with_image: bool) -> dict:
    observation = {
        "game_id": game_id, "state": frame.state.name, "step": step,
        "levels_completed": frame.levels_completed,
        "available_actions": available_action_names(frame.available_actions),
        "remaining_actions": remaining, "full_reset": bool(frame.full_reset),
    }
    if frame.frame:
        import numpy as np
        from PIL import Image
        from arc_agi.rendering import COLOR_MAP, hex_to_rgb

        # An action can produce multiple animation frames; use the final real frame.
        pixels = np.asarray(frame.frame[-1])
        if pixels.ndim == 2:
            if not np.issubdtype(pixels.dtype, np.integer) or pixels.min() < 0 or pixels.max() > 15:
                raise ValueError("invalid color IDs")
            observation["grid"] = pixels.tolist()
            palette = np.asarray([hex_to_rgb(COLOR_MAP[i]) for i in range(16)], dtype=np.uint8)
            image = Image.fromarray(palette[pixels])
        elif pixels.ndim == 3 and pixels.shape[-1] in (3, 4):
            image = Image.fromarray(pixels.astype(np.uint8))
        else:
            raise ValueError(f"Unsupported frame shape: {pixels.shape}")
        observation["width"], observation["height"] = image.size
        if with_image:
            # Palette colors are faithfully rendered; action coordinates stay in original pixels.
            image = image.resize((image.width * 4, image.height * 4), Image.Resampling.NEAREST)
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            observation["image_png_base64"] = base64.b64encode(buffer.getvalue()).decode("ascii")
    return observation


class MyAgent(Agent):
    MAX_ACTIONS = 79  # Preserve the original inclusive external configuration.

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.action_counter = 0
        self.evaluation_max_levels: int | None = None
        self._runtime = create_runtime(self.game_id)
        self._sent_decisions: set[str] = set()
        self._stopped = False

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        return self._stopped or latest_frame.state is GameState.WIN

    def _observation(self, frame):
        observation = frame_observation(self.game_id, frame, self.action_counter,
                                        max(0, self.MAX_ACTIONS + 1 - self.action_counter),
                                        with_image=bool(os.getenv("ADK_MODEL")))
        if self.evaluation_max_levels is not None and frame.levels_completed >= self.evaluation_max_levels:
            observation["evaluation_stop_reason"] = "level_limit"
        return observation

    def choose_action(self, frames: list[FrameData], latest_frame: FrameData) -> GameAction:
        result = self._runtime.decide(self._observation(latest_frame))
        if result["status"] != "action":
            self._stopped = True
            raise PolicyStopped(result["reason"])
        if result["decision_id"] in self._sent_decisions:
            self._stopped = True
            raise PolicyStopped("duplicate decision delivery")
        action = GameAction[result["action"]]
        # GameAction is a mutable enum singleton; clear data left by previous users.
        data = {"game_id": self.game_id}
        if action is GameAction.ACTION6:
            data.update(x=result["x"], y=result["y"])
        action.set_data(data)
        action.reasoning = {"policy": "adk-cognition", "reason": result["reason"],
                            "decision_id": result["decision_id"]}
        self._sent_decisions.add(result["decision_id"])
        return action

    @trace_agent_session
    def main(self) -> None:
        """Own the stop boundary because the upstream loop always expects an action."""
        self.timer = time.time()
        try:
            while True:
                latest = self._convert_raw_frame_data(self.arc_env.observation_space)
                if self.action_counter == 0:
                    self.frames[0] = latest
                try:
                    action = self.choose_action(self.frames, latest)
                except PolicyStopped as exc:
                    logging.info("%s: %s", self.game_id, exc)
                    break
                frame = self.take_action(action)
                self.action_counter += 1
                if frame is None:
                    raise RuntimeError("action outcome unavailable; refusing automatic resend")
                self.append_frame(frame)
                logging.info("%s - %s: count %s, levels completed %s", self.game_id,
                             action.name, self.action_counter, frame.levels_completed)
                # The next iteration records the final observation even at WIN or budget exhaustion.
        finally:
            self._runtime.close()
            self.cleanup()
