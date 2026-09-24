"""The ARC boundary and generated offline notebook must exercise the new runtime."""
from __future__ import annotations

import base64
import io
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "vendor/ARC-AGI-3-Agents"))
from arcengine import FrameData, FrameDataRaw, GameState
import numpy as np
from PIL import Image

from agent.my_agent import MyAgent, frame_observation
from scripts.build_notebook import build, SOURCES, ADK_WHEELS


class DriverTests(unittest.TestCase):
    def test_palette_and_last_animation_frame_preserve_coordinates(self):
        f = FrameData(frame=[[[5, 5]], [[8, 9]]], state=GameState.NOT_FINISHED)
        obs = frame_observation("x", f, 0, 2, with_image=True)
        self.assertEqual(obs["grid"], [[8, 9]])
        self.assertEqual((obs["width"], obs["height"]), (2, 1))
        image = Image.open(io.BytesIO(base64.b64decode(obs["image_png_base64"])))
        self.assertEqual(image.getpixel((0, 0)), (249, 60, 49))
        self.assertEqual(image.getpixel((4, 0)), (30, 147, 255))

    def test_driver_budget_stops_without_extra_action(self):
        f = FrameDataRaw(frame=[np.asarray([[0]], dtype=np.int8)], state=GameState.NOT_FINISHED, available_actions=[1])
        class Env:
            observation_space = f
            calls = 0
            def step(self, action, **kwargs):
                self.calls += 1
                return f
        env = Env()
        with patch.dict("os.environ", {"ADK_MODEL": ""}):
            player = MyAgent(card_id="test", game_id="unknown", agent_name="test", ROOT_URL="",
                             record=False, arc_env=env)
        player.MAX_ACTIONS = 1
        player.main()
        self.assertEqual(env.calls, 2)
        self.assertEqual(player._runtime.memory.stop_reason, "budget_exhausted")
        self.assertEqual(player._runtime.memory.revision, 3)
        self.assertTrue(player._runtime.closed)

    def test_notebook_contains_all_modules_and_offline_adk(self):
        notebook = build()
        for i, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] == "code":
                compile(cell["source"], f"cell_{i}", "exec")
        self.assertIn("agent/cognition/workflow.py", SOURCES)
        self.assertIn("agent/cognition/skills.py", SOURCES)
        self.assertTrue(all(w.is_file() for w in ADK_WHEELS))
        self.assertIn("google-adk==2.0.0", str(notebook))
