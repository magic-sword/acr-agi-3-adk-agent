"""Evaluation limits, official gateway contract and diagnostics without a leaderboard submission."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'vendor/ARC-AGI-3-Agents'))
import numpy as np
from arcengine import FrameDataRaw, GameState
from agent.my_agent import MyAgent
from scripts.eval_reporting import diagnostics, percentile, write_report
from scripts.benchmark_local import recover_interrupted


class EvaluationLimits(unittest.TestCase):
    def test_level_limit_stops_after_clear_without_another_action(self):
        initial = FrameDataRaw(frame=[np.asarray([[0]], dtype=np.int8)], state=GameState.NOT_FINISHED,
                               levels_completed=0, available_actions=[1])
        cleared = FrameDataRaw(frame=[np.asarray([[1]], dtype=np.int8)], state=GameState.NOT_FINISHED,
                               levels_completed=1, available_actions=[1])
        class Env:
            observation_space = initial
            calls = 0
            def step(self, action, **kwargs):
                self.calls += 1
                self.observation_space = cleared
                return cleared
        env = Env()
        with patch.dict('os.environ', {'ADK_MODEL': ''}):
            player = MyAgent(card_id='test', game_id='unknown', agent_name='eval', ROOT_URL='',
                             record=False, arc_env=env)
        player.evaluation_max_levels = 1
        player.main()
        self.assertEqual(env.calls, 1)
        self.assertEqual(player._runtime.memory.stop_reason, 'level_limit')
        self.assertEqual(player._runtime.memory.last_observation['levels_completed'], 1)
        self.assertEqual(player._runtime.memory.model_calls, 0)


class EvaluationReports(unittest.TestCase):
    def test_native_tool_metrics_and_duplicate_representation(self):
        from scripts.eval_reporting import tool_requests
        normalized = [{'id': 'x', 'name': 'list_observations', 'arguments': {}}]
        exchange = {'response': {}, 'normalized_tool_calls': normalized}
        self.assertEqual(tool_requests(exchange), [
            {'function': {'name': 'list_observations', 'arguments': {}}}])
        exchange['response']['tool_calls'] = tool_requests(exchange)
        self.assertEqual(len(tool_requests(exchange)), 1)

    def test_missing_score_is_not_zero_or_omitted_from_suite_mean(self):
        with tempfile.TemporaryDirectory() as d:
            summary = write_report(Path(d), [
                {'game_id': 'a', 'sdk_score_full_game': 20, 'stop_reason': 'level_limit'},
                {'game_id': 'b', 'sdk_score_full_game': None, 'stop_reason': 'hard_timeout'},
            ])
            self.assertIsNone(summary['mean_sdk_score_full_game'])
            self.assertEqual(summary['mean_available_score'], 20)
            self.assertEqual(summary['scored_games'], 1)

    def test_diagnostics_separate_schema_errors_and_repeated_actions(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            turns = [{'trace': ['DECIDE', 'RUN'], 'action': {'status': 'action', 'action': 'ACTION1'},
                      'frame_hash': 'same', 'errors': [], 'verification': [], 'decision_seconds': 3}] * 2
            (root / 'run.jsonl').write_text('\n'.join(map(json.dumps, turns)))
            calls = [{'schema_valid': True, 'seconds': 2, 'usage': {'prompt_tokens': 8, 'completion_tokens': 2},
                      'http_requests': 2, 'exchanges': [{'response': {'tool_calls': [
                          {'function': {'name': 'submit_grounding'}}]}}]},
                     {'error': 'invalid JSON', 'seconds': 4}]
            (root / 'run.model.jsonl').write_text('\n'.join(map(json.dumps, calls)))
            (root / 'run.observations.jsonl').write_text('{}\n')
            result = diagnostics(root)
            self.assertEqual(result['model_calls'], 2)
            self.assertEqual(result['schema_valid_rate'], .5)
            self.assertEqual(result['identical_frame_action_repeats'], 1)
            self.assertEqual(result['model_latency_p50'], 3)
            self.assertEqual(result['tokens']['prompt_tokens'], 8)
            self.assertEqual(result['model_http_requests'], 3)
            self.assertEqual(result['tool_calls'], {'submit_grounding': 1})
            self.assertTrue(result['hints'])

    def test_hard_timeout_counts_acknowledged_actions_not_planned_actions(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            events = [{'http_status': 200, 'response': {'levels_completed': 0, 'state': 'NOT_FINISHED'}},
                      {'http_status': 200, 'response': {'levels_completed': 1, 'state': 'NOT_FINISHED'}}]
            (root / 'gateway.jsonl').write_text('\n'.join(map(json.dumps, events)) + '\n{"incomplete":')
            r = recover_interrupted(root, 'game', 20)
            self.assertEqual(r['actions'], 1)
            self.assertEqual(r['levels_completed'], 1)
            self.assertIsNone(r['sdk_score_full_game'])

    def test_empty_latency_is_unknown(self):
        self.assertIsNone(percentile([], .95))
        self.assertEqual(percentile([4], .95), 4)


    def test_fast_slow_diagnostics_separate_deliberation_and_execution(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            calls=[{'work':'ground','attempt':1,'seconds':4,'http_requests':1},
                   {'work':'execute_step','seconds':.4,'http_requests':1},
                   {'work':'aim','seconds':.7,'http_requests':1}]
            (root/'r.model.jsonl').write_text('\n'.join(map(json.dumps,calls)))
            events=[{'event':'plan_created'},{'event':'action_feedback','trial':{'frame_changed':False}},
                    {'event':'reconciliation_requested'},
                    {'event':'cursor_adjusted'}, {'event':'cursor_confirmed','seconds':2.1}]
            (root/'r.artifacts.jsonl').write_text('\n'.join(map(json.dumps,events)))
            report=diagnostics(root)['fast_slow']
            self.assertEqual(report['repairs'],1)
            self.assertEqual(report['no_visible_effect'],1)
            self.assertEqual(report['reconsiderations'],1)
            self.assertEqual(report['by_work']['execute_step']['latency_p50'],.4)
            self.assertEqual(report['by_work']['aim']['latency_p50'],.7)
            self.assertEqual(report['cursor_latency_p50'],2.1)
            self.assertEqual(report['cursor_adjustments'],1)
            self.assertEqual(report['cursor_confirmations'],1)
