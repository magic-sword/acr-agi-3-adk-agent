"""Regression checks for ARC's integer action IDs and ADK model replies."""
from __future__ import annotations

import json
import os
import sys
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "vendor/ARC-AGI-3-Agents"))

from arcengine import GameAction

from agent.adk_policy import _parse_model_answer, decide
from agent.my_agent import available_action_names
from scripts.wait_model import PIXEL


class ActionContractTests(unittest.TestCase):
    def test_framework_ids_become_canonical_names(self) -> None:
        self.assertEqual(
            available_action_names([1, "2", GameAction.ACTION3, 6, 7]),
            ["ACTION1", "ACTION2", "ACTION3", "ACTION6", "ACTION7"],
        )

    def test_model_ids_respect_available_actions(self) -> None:
        observation = {"available_actions": ["ACTION1", "ACTION3"]}
        for raw in ('{"action":"1"}', '{"action":1}', '{"action":"action1"}'):
            with self.subTest(raw=raw):
                answer = _parse_model_answer(raw, observation)
                self.assertIs(GameAction[answer["action"]], GameAction.ACTION1)
        with self.assertRaisesRegex(ValueError, "Invalid model action"):
            _parse_model_answer('{"action":"2"}', observation)
        answer = _parse_model_answer(
            '{"action":6,"x":12,"y":35}', {"available_actions": ["ACTION6"]}
        )
        self.assertEqual((answer["action"], answer["x"], answer["y"]),
                         ("ACTION6", 12, 35))

    def test_adk_runner_accepts_numeric_reply_for_image_observation(self) -> None:
        received = []

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                received.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
                body = b'{"choices":[{"message":{"content":"{\\"action\\":\\"1\\"}"}}]}'
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args: object) -> None:
                pass

        server = HTTPServer(("127.0.0.1", 0), Handler)
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        try:
            with patch.dict(os.environ, {
                "ADK_MODEL": "local/qwen3-vl-4b-instruct",
                "VLM_API_BASE": f"http://127.0.0.1:{server.server_port}/v1",
            }):
                answer = decide({
                    "game_id": "ls20", "state": "IN_PROGRESS", "step": 1,
                    "available_actions": available_action_names([1, 2]),
                    "image_png_base64": PIXEL,
                })
            self.assertIs(GameAction[answer["action"]], GameAction.ACTION1)
            self.assertTrue(any(
                part.get("type") == "image_url"
                for message in received[0]["messages"]
                for part in message["content"] if isinstance(message["content"], list)
            ))
        finally:
            server.shutdown()
            server.server_close()
            worker.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
