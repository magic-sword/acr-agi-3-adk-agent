"""Current-vs-history, exact cursor coordinates and actual vision tool transport."""
import base64
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'vendor/ARC-AGI-3-Agents'))
import numpy as np
from PIL import Image
from arcengine import FrameData, GameState
from google.genai import types
from google.adk.models.llm_request import LlmRequest

from agent.my_agent import frame_observation
from agent.observation import _renderer
from agent.cognition.engine import CognitiveTurn
from agent.cognition.workflow import CognitiveRuntime
from agent.local_vlm import LocalVisionLlm


class VisualObservationTests(unittest.TestCase):
    def observation(self):
        frames = [np.full((64, 64), c).tolist() for c in (5, 8, 9, 14, 11)]
        return frame_observation('visual', FrameData(frame=frames, state=GameState.NOT_FINISHED,
                                available_actions=[1, 6]), 1, 10, with_image=True,
                                source_action={'action': 'ACTION1'})

    def test_current_is_final_and_cursor_does_not_change_game_evidence(self):
        obs = self.observation()
        self.assertEqual(obs['grid'][0][0], 11)
        image = Image.open(io.BytesIO(base64.b64decode(obs['image_png_base64'])))
        self.assertEqual(image.getpixel((32, 44)), (255, 220, 0))
        runtime = CognitiveRuntime('visual')
        self.addCleanup(runtime.close)
        runtime.turn = CognitiveTurn(runtime.memory, obs)
        runtime.turn.observe()
        before = runtime.turn.obs['observation_id'], runtime.turn.obs['frame_hash']
        for x, y in [(0, 0), (63, 63), (12, 35)]:
            result = runtime.move_observation_cursor(x, y)
            self.assertEqual(result['cursor'], {'x': x, 'y': y})
        self.assertEqual(before, (runtime.turn.obs['observation_id'], runtime.turn.obs['frame_hash']))
        self.assertIn('error', runtime.move_observation_cursor(64, 0))
        self.assertEqual(runtime.observe_current()['view'], 'current_final_frame')
        self.assertNotIn('next_start_frame', runtime.observe_current())

    def test_history_pagination_and_reread_are_same_evidence(self):
        obs = self.observation()
        runtime = CognitiveRuntime('visual')
        self.addCleanup(runtime.close)
        runtime.turn = CognitiveTurn(runtime.memory, obs)
        runtime.turn.observe()
        event = obs['animation']['event_id']
        page = runtime.observe_animation(event)
        self.assertFalse(page['time_advanced'])
        self.assertEqual(page, runtime.observe_animation(event))
        self.assertEqual(page['next_start_frame'], 4)
        self.assertIsNone(runtime.observe_animation(event, 4)['next_start_frame'])
        self.assertIn('error', runtime.observe_animation('stale'))
        self.assertIn('error', runtime.observe_animation(event, -1))
        self.assertNotIn('_visual_frames', runtime._context()['observation'])
        self.assertNotIn('image_png_base64', json.dumps(runtime._context()))
        self.assertEqual(runtime.observe_current()['animation']['event_id'], event)

    def test_single_play_gif_and_cli(self):
        import subprocess
        obs = self.observation()
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / 'event.json'
            source.write_text(json.dumps({'frames': obs['_visual_frames'],
                              'available_actions': obs['available_actions'], 'cursor': obs['cursor'],
                              **obs['animation']}))
            for mode, name in [('current', 'current.png'), ('replay', 'history.gif')]:
                target = Path(tmp) / name
                subprocess.run([sys.executable, _renderer.__file__, mode, str(source), str(target)], check=True)
                with Image.open(target) as image:
                    if mode == 'replay':
                        self.assertNotIn('loop', image.info)
                        self.assertEqual(image.n_frames, 5)
                        image.seek(4)
                        self.assertEqual(image.convert('RGB').getpixel((32, 44)), (255, 220, 0))
                    else:
                        self.assertEqual(image.format, 'PNG')

    def test_adk_executes_visual_tools_and_archives_source_frames(self):
        from unittest.mock import patch
        obs = self.observation()
        payloads = []

        def respond(model, payload):
            payloads.append(payload)
            n = len(payloads)
            calls = [('move_observation_cursor', {'x': 12, 'y': 35}),
                     ('observe_animation', {'event_id': obs['animation']['event_id']}),
                     ('observe_current', {})]
            if n <= len(calls):
                name, args = calls[n-1]
                message = {'tool_calls': [{'id': f'visual{n}', 'type': 'function',
                           'function': {'name': name, 'arguments': json.dumps(args)}}]}
            else:
                context = next(json.loads(part['text']) for m in payload['messages']
                               if m['role'] == 'user' and isinstance(m['content'], list)
                               for part in m['content'] if part.get('type') == 'text')
                reply = {'observation_id': context['observation_id'], 'memory_revision': context['memory_revision']}
                if n == 4:
                    reply.update(facts=[], unknowns=['goal'], goal='finish')
                else:
                    reply.update(purpose='plan', plan=[{'id': 'go', 'subgoal': 'finish',
                        'action': {'action': 'ACTION1'}, 'completion': [{'kind': 'state', 'value': 'WIN'}],
                        'effects': [{'kind': 'state', 'value': 'WIN'}]}])
                message = {'content': json.dumps(reply)}
            return {'choices': [{'message': message}]}

        with tempfile.TemporaryDirectory() as tmp, patch.object(LocalVisionLlm, '_complete', respond):
            runtime = CognitiveRuntime('visual', 'local/qwen3-vl-4b-instruct', log_dir=tmp)
            self.addCleanup(runtime.close)
            result = runtime.decide(obs)
            self.assertEqual(result['action'], 'ACTION1')
            self.assertEqual(runtime.cursor, {'x': 12, 'y': 35})
            self.assertNotIn('_visual_frames', runtime.memory.last_observation)
            archive = next((Path(tmp) / 'frames').glob('*.json'))
            self.assertEqual(json.loads(archive.read_text())['frames'], obs['_visual_frames'])
            tool_results = [json.loads(m['content']) for m in payloads[3]['messages'] if m['role'] == 'tool']
            self.assertEqual([r['view'] for r in tool_results],
                             ['current_final_frame', 'historical_replay_not_live', 'current_final_frame'])
            images = [p for m in payloads[3]['messages'] if isinstance(m['content'], list)
                      for p in m['content'] if p.get('type') == 'image_url']
            self.assertEqual(len(images), 4)  # Initial frame plus three tool images.

    def test_tool_image_reaches_vision_transport_without_base64_text(self):
        obs = self.observation()
        model = LocalVisionLlm(model='qwen3-vl-4b-instruct', api_base='http://unused')
        request = LlmRequest(contents=[
            types.Content(role='model', parts=[types.Part(function_call=types.FunctionCall(
                id='view', name='observe_animation', args={}))]),
            types.Content(role='user', parts=[types.Part(function_response=types.FunctionResponse(
                id='view', name='observe_animation', response={'view': 'historical_replay_not_live',
                '_image_png_base64': obs['image_png_base64']}))])])
        payload = model._payload(request)
        self.assertEqual(payload['messages'][1]['role'], 'tool')
        self.assertNotIn('_image_png_base64', payload['messages'][1]['content'])
        self.assertEqual(payload['messages'][2]['content'][1]['type'], 'image_url')


if __name__ == '__main__':
    unittest.main()
