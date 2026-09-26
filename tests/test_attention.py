"""Image-led production policy through real ADK dispatch and scripted HTTP replies."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from agent.cognition.workflow import CognitiveRuntime
from agent.local_vlm import LocalVisionLlm
from test_decide_run import call, context, ack
from test_skill_learning import obs


def answer_for(c, x=0, **updates):
    answer = dict(observation_id=c['observation_id'], interpretation='Its response is still uncertain.',
                  focus='The visible cell at the left', action={'action': 'CLICK', 'x': x, 'y': 0},
                  expectation='Observe any local or remote effect, including no change.',
                  notes='Click effects remain tentative.')
    answer.update(updates)
    return 'submit_attention', answer


class AttentionTests(unittest.TestCase):
    def runtime(self, **kwargs):
        r = CognitiveRuntime('test', 'local/qwen3-vl-4b-instruct', **kwargs)
        self.addCleanup(r.close)
        return r

    def test_one_request_per_action_without_object_inventory_or_review_calls(self):
        r = self.runtime(); requests = []
        def answer(model, p):
            requests.append(p)
            return call(*answer_for(context(p), len(requests)-1))
        with patch.object(LocalVisionLlm, '_complete', answer):
            first = r.decide(obs()); ack(r)
            second = r.decide(obs(1))
        self.assertEqual((first['x'], second['x']), (0, 1))
        self.assertEqual(len(requests), 2)
        for p in requests:
            self.assertEqual(p['tool_choice'], 'required')
            self.assertEqual([t['function']['name'] for t in p['tools']], ['submit_attention'])
            self.assertEqual(context(p)['work'], 'attend')
            self.assertNotIn('components', context(p))
            self.assertNotIn('world', context(p))
            self.assertNotIn('notebook', context(p))
            self.assertFalse(any(m['role'] == 'tool' for m in p['messages']))
        c = context(requests[1])
        self.assertFalse(c['last_result']['frame_changed'])
        self.assertEqual(c['unchanged_state_trials'][0]['attempts'], 1)
        self.assertEqual(c['working_notes_hypotheses'], 'Click effects remain tentative.')
        self.assertIsNone(r.experiments.active)

    def test_same_no_effect_action_is_repaired_once_without_dispatch(self):
        r = self.runtime(); requests = []
        def answer(model, p):
            c = context(p); requests.append(c)
            return call(*answer_for(c, 1 if c['correction'] else 0))
        with patch.object(LocalVisionLlm, '_complete', answer):
            r.decide(obs()); ack(r); result = r.decide(obs(1))
        self.assertEqual(result['x'], 1)
        self.assertEqual(r.calls, 2)
        self.assertEqual(len(requests), 3)
        self.assertIn('exhausted', requests[-1]['correction'])
        self.assertEqual(r.memory.pending['action']['x'], 1)

    def test_prose_or_new_repeat_allowance_cannot_reopen_failed_action(self):
        r = self.runtime()
        def answer(model, p):
            return call(*answer_for(context(p), repeat_limit=1 if r.obs['step'] == 0 else 3,
                                   repeat_reason='A new wording of a delay hypothesis', focus='Different name'))
        with patch.object(LocalVisionLlm, '_complete', answer):
            r.decide(obs()); ack(r); result = r.decide(obs(1))
        self.assertEqual(result['status'], 'stop')
        self.assertEqual(result['reason'], 'attention_output_invalid')
        self.assertEqual(r.calls, 2)
        self.assertIsNone(r.memory.pending)

    def test_predeclared_repeat_has_frozen_total_limit(self):
        r = self.runtime()
        def answer(model, p):
            return call(*answer_for(context(p), repeat_limit=2, repeat_reason='Test a two-press delay'))
        with patch.object(LocalVisionLlm, '_complete', answer):
            r.decide(obs()); ack(r)
            self.assertEqual(r.decide(obs(1))['status'], 'action'); ack(r)
            self.assertEqual(r.decide(obs(2))['status'], 'stop')
        self.assertEqual(r.unchanged_trials[0]['attempts'], 2)

    def test_effect_elsewhere_is_global_feedback_and_same_action_can_continue(self):
        r = self.runtime(); inputs = []
        def answer(model, p):
            inputs.append(context(p)); return call(*answer_for(context(p)))
        with patch.object(LocalVisionLlm, '_complete', answer):
            r.decide(obs(grid=[[0, 1]])); ack(r)
            result = r.decide(obs(1, [[0, 2]]))
        self.assertEqual(result['status'], 'action')
        feedback = inputs[-1]['last_result']
        self.assertEqual(feedback['changed_cell_count'], 1)
        self.assertEqual(feedback['change_bounds'], {'x': 1, 'y': 0, 'width': 1, 'height': 1})
        self.assertEqual(feedback['changed_cells'][0]['x'], 1)
        self.assertEqual(inputs[-1]['unchanged_state_trials'], [])

    def test_boundary_clears_trial_restrictions_and_hypotheses(self):
        r = self.runtime(); inputs = []
        def answer(model, p):
            inputs.append(context(p)); return call(*answer_for(context(p)))
        with patch.object(LocalVisionLlm, '_complete', answer):
            r.decide(obs()); ack(r)
            result = r.decide({**obs(1), 'levels_completed': 1})
        self.assertEqual(result['status'], 'action')
        self.assertEqual(inputs[-1]['unchanged_state_trials'], [])
        self.assertEqual(inputs[-1]['working_notes_hypotheses'], '')
        self.assertEqual(inputs[-1]['recent_trials'], [])

    def test_invalid_coordinates_and_stale_answers_get_only_one_repair(self):
        for updates in ({'action': {'action': 'CLICK', 'x': 63, 'y': 63}},
                        {'observation_id': 'stale'}, {'repeat_limit': 3}):
            with self.subTest(updates=updates):
                r = self.runtime()
                with patch.object(LocalVisionLlm, '_complete', lambda m, p: call(*answer_for(context(p), **updates))) as http:
                    result = r.decide(obs())
                self.assertEqual(result['reason'], 'attention_output_invalid')
                self.assertEqual(r.memory.model_calls, 2)
                self.assertIsNone(r.memory.pending)

    def test_deadline_prevents_action_even_if_model_returns_valid_decision(self):
        r = self.runtime()
        def answer(model, p):
            r.deadline = 0
            return call(*answer_for(context(p)))
        with patch.object(LocalVisionLlm, '_complete', answer):
            result = r.decide(obs())
        self.assertEqual(result['reason'], 'budget_exhausted')
        self.assertIsNone(r.memory.pending)

    def test_no_receipt_prevents_resend_and_duplicate_observation_is_idempotent(self):
        r = self.runtime()
        with patch.object(LocalVisionLlm, '_complete', lambda m, p: call(*answer_for(context(p)))) as http:
            first = r.decide(obs())
            self.assertEqual(r.decide(obs()), first)
            result = r.decide(obs(1))
        self.assertEqual(r.memory.model_calls, 1)
        self.assertEqual(result['reason'], 'execution_outcome_unknown')

    def test_terminal_observation_records_result_without_another_inference(self):
        with tempfile.TemporaryDirectory() as directory:
            r = self.runtime(log_dir=directory)
            with patch.object(LocalVisionLlm, '_complete', lambda m, p: call(*answer_for(context(p)))) as http:
                r.decide(obs()); ack(r)
                result = r.decide({**obs(1), 'state': 'WIN'})
            self.assertEqual(result['reason'], 'win')
            self.assertEqual(r.memory.model_calls, 1)
            records = [json.loads(s) for s in (Path(directory)/(r.session_id+'.artifacts.jsonl')).read_text().splitlines()]
            self.assertTrue(any(e['event'] == 'attention_feedback' for e in records))
            self.assertTrue(any(e['event'] == 'attention_selected' for e in records))

    def test_images_show_before_only_when_changed_and_no_object_scan_runs(self):
        from agent.observation import attach_visuals
        r = self.runtime(); requests = []
        def answer(model, p):
            requests.append(p)
            return call(*answer_for(context(p), 1 if len(requests) == 3 else 0))
        observations = [obs(grid=[[0, 1]]), obs(1, [[0, 2]]), obs(2, [[0, 2]])]
        for o in observations:
            attach_visuals(o, [o['grid']], None)
        with patch('agent.cognition.world.components', side_effect=AssertionError('no object scan')), \
             patch.object(LocalVisionLlm, '_complete', answer):
            for o in observations:
                self.assertEqual(r.decide(o)['status'], 'action'); ack(r)
        images = [sum(part.get('type') == 'image_url' for m in p['messages']
                      if isinstance(m.get('content'), list) for part in m['content']) for p in requests]
        self.assertEqual(images, [1, 2, 1])
        last_parts = requests[-1]['messages'][-1]['content']
        self.assertIn('already_tried_without_effect_do_not_repeat', last_parts[-1]['text'])

    def test_repairs_can_be_disabled_and_missing_visuals_do_not_invoke_model(self):
        r = self.runtime(repair_attempts=0)
        with patch.object(LocalVisionLlm, '_complete', lambda m, p: call(*answer_for(context(p), observation_id='stale'))):
            self.assertEqual(r.decide(obs())['reason'], 'attention_output_invalid')
        self.assertEqual(r.memory.model_calls, 1)
        r = self.runtime(); observation = obs(); observation.pop('grid')
        self.assertEqual(r.decide(observation)['reason'], 'observation_unavailable')
        self.assertEqual(r.memory.model_calls, 0)

    def test_attention_replay_shows_current_output_without_future_notes(self):
        from scripts.agent_monitor import attention_html
        base = {'context': {'work': 'attend', 'working_notes_hypotheses': 'tentative'},
                'current': {'step': 1}, 'response': {'step': 2, 'schema_valid': True,
                    'response': json.dumps({'focus': 'future focus', 'notes': '<script>bad</script>'})}}
        self.assertNotIn('future focus', attention_html(base))
        base['current']['step'] = 2
        self.assertIn('future focus', attention_html(base))
        self.assertNotIn('<script>', attention_html(base))


if __name__ == '__main__':
    unittest.main()
