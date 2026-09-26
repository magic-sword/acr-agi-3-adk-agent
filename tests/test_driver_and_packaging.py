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
        self.assertEqual(image.getpixel((32, 44)), (249, 60, 49))
        self.assertEqual(image.getpixel((38, 44)), (30, 147, 255))

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
        with patch.object(player._runtime, 'record_execution', wraps=player._runtime.record_execution) as events:
            player.main()
        self.assertEqual([c.args[0] for c in events.call_args_list],
                         ['action_dispatched', 'action_acknowledged'] * 2)
        self.assertEqual(env.calls, 2)
        self.assertEqual(player._runtime.memory.stop_reason, "budget_exhausted")
        self.assertEqual(player._runtime.memory.revision, 3)
        self.assertTrue(player._runtime.closed)

    def test_execution_failure_is_recorded_without_acknowledgement_or_resend(self):
        f = FrameDataRaw(frame=[np.asarray([[0]], dtype=np.int8)],
                         state=GameState.NOT_FINISHED, available_actions=[1])
        class Env:
            observation_space = f
        with patch.dict('os.environ', {'ADK_MODEL': ''}):
            player = MyAgent(card_id='test', game_id='unknown', agent_name='test', ROOT_URL='',
                             record=False, arc_env=Env())
        with patch.object(player, 'take_action', side_effect=RuntimeError('response lost')) as take, \
             patch.object(player._runtime, 'record_execution', wraps=player._runtime.record_execution) as events:
            with self.assertRaisesRegex(RuntimeError, 'response lost'):
                player.main()
        take.assert_called_once()
        self.assertEqual([c.args[0] for c in events.call_args_list],
                         ['action_dispatched', 'action_outcome_unknown'])
        self.assertTrue(player._runtime.closed)

    def test_notebook_contains_all_modules_and_offline_adk(self):
        notebook = build()
        for i, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] == "code":
                compile(cell["source"], f"cell_{i}", "exec")
        self.assertIn("agent/cognition/workflow.py", SOURCES)
        self.assertIn("agent/cognition/library.py", SOURCES)
        self.assertIn("agent/cognition/skills.py", SOURCES)
        for module in ('attention',):
            self.assertIn(f'agent/cognition/{module}.py', SOURCES)
        self.assertFalse(any(path.startswith('tests/') for path in SOURCES))
        self.assertTrue(all(w.is_file() for w in ADK_WHEELS))
        self.assertIn("google-adk==2.0.0", str(notebook))
