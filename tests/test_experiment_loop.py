"""Causal-experiment boundaries: predictions precede actions, reviews precede reuse."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from agent.cognition.state import Decision, ExperimentReview
from agent.cognition.workflow import CognitiveRuntime
from agent.local_vlm import LocalVisionLlm
from scripts.agent_monitor import Timeline, Run, dashboard_html
from scripts.eval_reporting import diagnostics
from test_decide_run import act, ack, call, context, experiment, revised
from test_skill_learning import obs


class ExperimentLoopTests(unittest.TestCase):
    def runtime(self, **kw):
        r = CognitiveRuntime('test', 'local/qwen3-vl-4b-instruct', **kw)
        self.addCleanup(r.close)
        return r

    def test_missing_plan_cannot_reach_execution(self):
        with self.assertRaisesRegex(ValueError, 'experiment plan'):
            Decision(kind='act', action={'action': 'UP'}, purpose='Explore')

    def reviewed_runtime(self):
        r = self.runtime()
        with patch.object(LocalVisionLlm, '_complete', return_value=call('submit_decision', act())):
            r.decide(obs()); ack(r)
        r._receive(obs(1))
        r._finish_review(r.experiments.automatic_review(), source='host')
        return r

    def test_revision_requires_current_review_and_actual_declared_change(self):
        r = self.reviewed_runtime(); c = r._context()
        with self.assertRaisesRegex(ValueError, 'must cite notebook.handoff'):
            r.validate_job(Decision.model_validate(act(1)))
        job = revised(act(1), c)
        for field, value in [('experiment_id', 'unknown'), ('review_revision', 1)]:
            bad = deepcopy(job); bad['experiment']['revision'][field] = value
            with self.assertRaisesRegex(ValueError, 'current handoff'):
                r.validate_job(Decision.model_validate(bad))
        bad = deepcopy(job); bad['experiment']['revision']['change'] = 'context'
        with self.assertRaisesRegex(ValueError, 'change context is absent'):
            r.validate_job(Decision.model_validate(bad))
        r.validate_job(Decision.model_validate(job))
        self.assertEqual(c['notebook']['handoff']['observed_fact'],
                         r.notebook.pages['experiment-1']['data']['review']['finding'])

    def test_scope_revision_and_corrected_target_are_allowed_but_prose_only_is_not(self):
        r = self.reviewed_runtime(); c = r._context()
        # Same click, but observe remote effects instead of only the clicked pixel.
        broader = revised(act(), c, 'observation_scope')
        broader['experiment']['expected']['region']['width'] = 3
        r.validate_job(Decision.model_validate(broader))
        # Same supposed object: corrected placement is a changed action, not a coordinate ban.
        corrected = revised(act(1), c)
        corrected['experiment']['revision']['reconsidered_assumption'] = 'The prior click may have missed the intended object.'
        r.validate_job(Decision.model_validate(corrected))
        prose = revised(act(), c, 'observation_scope')
        prose['experiment']['expected']['kind'] = 'semantic'
        prose['experiment']['expected']['description'] = 'A newly worded prediction'
        with self.assertRaisesRegex(ValueError, 'change observation_scope is absent'):
            r.validate_job(Decision.model_validate(prose))

    def test_observed_condition_change_and_predeclared_retry_preserve_revision(self):
        r = self.reviewed_runtime(); c = r._context()
        r.obs['grid'][0][0] = 1
        same = revised(act(), c, 'context')
        r.validate_job(Decision.model_validate(same))
        r.obs['grid'][0][0] = 0
        changed = revised(act(1), c)
        changed['experiment'].update(max_attempts=2, repeat_reason='Predeclared delayed response test')
        decision = Decision.model_validate(changed)
        action = {'action': 'ACTION6', 'x': 1, 'y': 0}
        key = r.experiments.start(decision.experiment, action, r.obs)
        r.experiments.observe(r.obs, {'acknowledged': True, 'boundary': '', 'experience_id': 'experience-2'})
        r._finish_review(r.experiments.automatic_review(), source='host')
        retry = deepcopy(changed); retry['experiment']['retry_of'] = key
        r.validate_job(Decision.model_validate(retry))
        retry['experiment']['revision']['reason'] = 'Rewrite the frozen reasoning'
        with self.assertRaisesRegex(ValueError, 'preserve the original'):
            r.validate_job(Decision.model_validate(retry))

    def test_handoff_does_not_invent_target_failure_and_expires_at_boundary(self):
        r = self.reviewed_runtime()
        handoff = r.notebook.handoff()
        self.assertIn('unverified', handoff['understanding']['reconsider_assumption'])
        handoff['understanding']['open_question'] = 'tampered'
        self.assertNotEqual(r.notebook.handoff()['understanding']['open_question'], 'tampered')
        r.notebook.begin_segment('level')
        self.assertIsNone(r.notebook.handoff())
        with self.assertRaisesRegex(ValueError, 'current handoff'):
            job = revised(act(1), {'notebook': {'handoff': handoff}})
            r.validate_job(Decision.model_validate(job))

    def test_revised_plan_and_review_understanding_survive_journal_and_rendering(self):
        from scripts.notebook_view import notebook_html, page_html
        with tempfile.TemporaryDirectory() as directory:
            r = self.runtime(log_dir=directory)
            def answer(model, payload):
                c = context(payload)
                return call('submit_decision', act() if c['observation']['step'] == 0 else revised(act(1), c))
            with patch.object(LocalVisionLlm, '_complete', answer):
                r.decide(obs()); ack(r); result = r.decide(obs(1))
            self.assertEqual(result['x'], 1)
            page = r.experiments.active
            self.assertEqual(page['data']['plan']['revision']['review_revision'], 3)
            self.assertIn('再設計の根拠', page_html(page))
            t = Timeline(Run(Path(directory), r.session_id)).load()
            initial = next(i for i,e in enumerate(t.events) if e['event'] == 'notebook_opened')
            self.assertNotIn('見直す前提', notebook_html(t.snapshot(initial)))
            final = max(i for i,e in enumerate(t.events) if e['event'] == 'notebook_opened')
            self.assertIn('残る疑問', notebook_html(t.snapshot(final)))

    def test_changed_status_strip_does_not_support_target_or_allow_reworded_repeat(self):
        r = self.runtime(); seen = []
        def answer(model, payload):
            c = context(payload); seen.append(c)
            if c['observation']['step'] == 0:
                job = act(2); job['action']['y'] = 1
                job['experiment'] = experiment(2, 1)
                return call('submit_decision', job)
            self.assertEqual(c['work'], 'experiment_design')  # Host checks pixels without another LLM call.
            page = c['notebook']['latest_review']; d = page['data']
            self.assertEqual(d['review']['verdict'], 'unsupported')
            self.assertEqual(d['measurement']['changed_cells_in_region'], 0)
            self.assertNotIn('before_pixels', d)
            repeated = act(2); repeated['action']['y'] = 1
            repeated['experiment'] = experiment(2, 1)
            repeated['experiment']['question'] = 'Rephrased same test'
            with self.assertRaisesRegex(ValueError, 'same action'):
                r.validate_job(Decision.model_validate(repeated))
            return call('submit_decision', revised(act(3), c))
        with patch.object(LocalVisionLlm, '_complete', answer):
            r.decide(obs(0, [[0]*6, [0]*6])); ack(r)
            result = r.decide(obs(1, [[1,0,0,0,0,0], [0]*6]))
        self.assertEqual(result['x'], 3)
        self.assertEqual(len(seen), 2)
        self.assertEqual(r.memory.history[-1]['trace'], ['DECIDE','RUN','DECIDE','RUN'])

    def test_semantic_review_has_only_review_completion_and_can_complete_negative_question(self):
        r = self.runtime(); works = []
        def answer(model, payload):
            c = context(payload); works.append(c['work'])
            if c['observation']['step'] == 0:
                job = act(); job['experiment']['expected']['kind'] = 'semantic'
                return call('submit_decision', job)
            if c['work'] == 'experiment_review':
                names = [t['function']['name'] for t in payload['tools']]
                self.assertIn('submit_review', names)
                for name in ('submit_decision', 'propose_skill', 'write_note'):
                    self.assertNotIn(name, names)
                page = c['notebook']['active_experiment']
                review = {'experiment_id': page['id'], 'verdict': 'unsupported',
                          'finding': 'Target did not move', 'evidence_ids': ['experience-1'],
                          'subgoal_status': 'completed', 'next_step': 'revise',
                          'understanding': {'reconsider_assumption': 'An immediate response at this cell is unsupported.',
                                            'open_question': 'Does an adjacent cell respond?',
                                            'subgoal_reason': 'The immediate response question was answered negatively.'},
                          'update': 'The immediate response question is answered negatively; inspect a different target'}
                with self.assertRaisesRegex(ValueError, 'actual resulting experience'):
                    r.validate_job(ExperimentReview.model_validate(dict(review, evidence_ids=['fake'])))
                return call('submit_review', review)
            self.assertEqual(c['notebook']['latest_review']['data']['review']['verdict'], 'unsupported')
            self.assertEqual(c['notebook']['goal_path'][-1]['data']['status'], 'completed')
            return call('submit_decision', revised(act(1), c))
        with patch.object(LocalVisionLlm, '_complete', answer):
            r.decide(obs()); ack(r); result = r.decide(obs(1))
        self.assertEqual(result['x'], 1)
        self.assertEqual(works, ['experiment_design','experiment_review','experiment_design'])
        self.assertEqual(r.calls, 2)

    def test_frozen_plan_cannot_be_edited_and_bounded_delay_cannot_extend_limit(self):
        r = self.runtime(); plan = act()
        plan['experiment'].update(max_attempts=2, repeat_reason='Test a two-action response delay')
        calls = []
        def answer(model, payload):
            c = context(payload); calls.append(c)
            if c['observation']['step'] == 0:
                return call('submit_decision', plan)
            previous = c['notebook']['latest_review']
            retry = deepcopy(plan)
            retry['experiment']['retry_of'] = previous['id']
            if c['observation']['step'] == 1:
                return call('submit_decision', retry)
            with self.assertRaisesRegex(ValueError, 'attempt limit'):
                r.validate_job(Decision.model_validate(retry))
            return call('submit_decision', revised(act(1), c))
        with patch.object(LocalVisionLlm, '_complete', answer):
            r.decide(obs()); frozen = deepcopy(r.experiments.active)
            with self.assertRaisesRegex(ValueError, 'host-owned'):
                r.notebook.erase(frozen['id'], 1, 'Change criteria after acting')
            with self.assertRaisesRegex(ValueError, 'automatically'):
                r.notebook.bookmark('active_experiment', 'goal')
            ack(r); r.decide(obs(1)); ack(r); r.decide(obs(2))
        self.assertEqual(r.notebook.pages[frozen['id']]['data']['plan'], frozen['data']['plan'])
        self.assertEqual(r.notebook.pages['experiment-2']['data']['attempt'], 2)

    def test_last_action_is_reviewed_at_budget_or_win_before_stop(self):
        for state, reason in [('NOT_FINISHED', 'budget_exhausted'), ('WIN', 'win')]:
            r = self.runtime()
            with patch.object(LocalVisionLlm, '_complete', return_value=call('submit_decision', act())) as model:
                r.decide(obs()); ack(r)
                data = obs(1, [[1,0,0,0,0,0]]); data.update(state=state, remaining_actions=0)
                result = r.decide(data)
            self.assertEqual(result['reason'], reason)
            self.assertIsNone(r.experiments.active)
            self.assertEqual(r.notebook.pages['experiment-1']['data']['review']['verdict'], 'supported')
            self.assertEqual(model.call_count, 1)

    def test_boundary_and_unknown_receipt_never_become_false_negative(self):
        for boundary in ('level', 'reset', 'unknown'):
            r = self.runtime()
            with patch.object(LocalVisionLlm, '_complete', return_value=call('submit_decision', act())):
                r.decide(obs())
                if boundary != 'unknown': ack(r)
                data = obs(1); data['remaining_actions'] = 0
                if boundary == 'level': data['levels_completed'] = 1
                if boundary == 'reset': data['full_reset'] = True
                r.decide(data)
            page = r.notebook.pages['experiment-1']
            self.assertEqual(page['data']['review']['verdict'], 'inconclusive')
            if boundary != 'unknown':
                self.assertEqual(r.notebook.segment, 1)
                self.assertIsNone(r.notebook.opening()['active_experiment'])
            else:
                self.assertEqual(r.result['reason'], 'execution_outcome_unknown')

    def test_level_expectation_is_reviewed_before_new_segment(self):
        r = self.runtime(); job = act()
        job['experiment']['expected'] = {'kind':'level_increased','description':'The level counter increases'}
        with patch.object(LocalVisionLlm, '_complete', return_value=call('submit_decision', job)):
            r.decide(obs()); ack(r)
            data = obs(1); data.update(levels_completed=1, remaining_actions=0)
            r.decide(data)
        page = r.notebook.pages['experiment-1']
        self.assertEqual(page['segment'], 0)
        self.assertEqual(page['data']['review']['verdict'], 'supported')
        self.assertEqual(r.notebook.segment, 1)

    def test_expired_budget_still_allows_host_review_but_interrupts_semantic_review(self):
        for kind in ('region_changed', 'semantic'):
            r = self.runtime(); job = act(); job['experiment']['expected']['kind'] = kind
            with patch.object(LocalVisionLlm, '_complete', return_value=call('submit_decision', job)):
                r.decide(obs()); ack(r); r.deadline = time.monotonic() - 1
                result = r.decide(obs(1))
            data = r.notebook.pages['experiment-1']['data']
            self.assertEqual(data['status'], 'reviewed' if kind == 'region_changed' else 'interrupted')
            self.assertEqual(result['reason'], 'budget_exhausted' if kind == 'region_changed' else 'review_budget_exhausted')

    def test_changed_tested_conditions_allow_retry_without_a_permanent_coordinate_ban(self):
        r = self.runtime()
        def answer(model, payload):
            c = context(payload); step = c['observation']['step']
            return call('submit_decision', revised(act(1 if step == 1 else 0), c))
        with patch.object(LocalVisionLlm, '_complete', answer):
            r.decide(obs()); ack(r); r.decide(obs(1)); ack(r)
            # The other intervention changed the tested cell: retrying 0 is now informative.
            result = r.decide(obs(2, [[1,0,0,0,0,0]]))
        self.assertEqual(result['x'], 0)

    def test_review_journal_and_replay_never_show_future_verdict(self):
        with tempfile.TemporaryDirectory() as directory:
            r = self.runtime(log_dir=directory)
            with patch.object(LocalVisionLlm, '_complete', return_value=call('submit_decision', act())):
                r.decide(obs()); ack(r)
                final = obs(1); final['remaining_actions'] = 0
                r.decide(final)
            t = Timeline(Run(Path(directory), r.session_id)).load()
            start = next(i for i,e in enumerate(t.events) if e['event']=='experiment_started')
            reviewed = next(i for i,e in enumerate(t.events) if e['event']=='experiment_reviewed')
            observed = next(i for i,e in enumerate(t.events) if e['event']=='experiment_observed')
            self.assertEqual(t.snapshot(observed)['experiment']['data']['status'], 'awaiting_review')
            self.assertIsNone(t.snapshot(observed)['review'])
            self.assertIsNone(t.snapshot(start)['review'])
            self.assertIsNone(t.snapshot(start)['experiment']['data']['review'])
            self.assertIn('不支持', dashboard_html(t.snapshot(reviewed), Path(directory)))
            self.assertIn('review', [s['machine']['node'] for s in t.snapshots])
            report = diagnostics(Path(directory))
            self.assertEqual(report['experiment_verdicts'], {'unsupported': 1})
            self.assertEqual(report['host_loaded_skills']['design-experiment'], 1)
            self.assertEqual(report['experiment_reviewers'], {'host': 1})

    def test_duplicate_proposal_escalates_measurement_to_review_once_before_redesign(self):
        r = self.runtime(); requests = []
        def answer(model, payload):
            c = context(payload); requests.append(c)
            if len(requests) <= 2:
                return call('submit_decision', act())
            self.assertFalse(any(m['role'] == 'tool' for m in payload['messages']))
            if c['work'] == 'experiment_review':
                page = c['notebook']['active_experiment']
                self.assertIn('review_request', page['data'])
                schema = next(t['function']['parameters'] for t in payload['tools']
                              if t['function']['name'] == 'submit_review')
                self.assertEqual(schema['properties']['verdict']['enum'], ['unsupported'])
                review = dict(page['data']['review'], update='Test the adjacent target instead of repeating this cell')
                with self.assertRaisesRegex(ValueError, 'verdict is fixed'):
                    r.validate_job(ExperimentReview.model_validate(dict(review, verdict='supported')))
                return call('submit_review', review)
            self.assertEqual(c['notebook']['latest_review']['data']['reviewer'], 'agent')
            return call('submit_decision', revised(act(1), c))
        with patch.object(LocalVisionLlm, '_complete', answer):
            r.decide(obs()); ack(r); result = r.decide(obs(1))
        self.assertEqual(result['x'], 1)
        self.assertEqual(r.calls, 3)
        self.assertEqual(len([p for p in r.notebook.pages.values() if p['kind']=='experiment']), 2)

    def test_relevant_prerequisite_change_allows_same_target_test(self):
        r = self.runtime(); first = act()
        first['experiment']['context_region'] = {'x':5, 'y':0, 'width':1, 'height':1}
        def answer(model, payload):
            c = context(payload); step = c['observation']['step']
            return call('submit_decision', revised(act(5) if step == 1 else first, c))
        with patch.object(LocalVisionLlm, '_complete', answer):
            r.decide(obs()); ack(r); r.decide(obs(1)); ack(r)
            result = r.decide(obs(2, [[0,0,0,0,0,1]]))
        self.assertEqual(result['x'], 0)

    def test_reconsidering_an_older_test_uses_its_frames_and_does_not_relabel_latest_outcome(self):
        r = self.runtime()
        def answer(model, payload):
            c = context(payload); step = c['observation']['step']
            if c['work'] == 'experiment_review':
                texts = [part['text'] for message in payload['messages']
                         if message['role']=='user' and isinstance(message['content'], list)
                         for part in message['content'] if part.get('type')=='text']
                self.assertIn(json.dumps([[0]*6]), texts)
                self.assertIn(json.dumps([[0,0,0,0,0,1]]), texts)
                self.assertNotIn(json.dumps([[0,0,0,0,0,2]]), texts)
                page = c['notebook']['active_experiment']
                return call('submit_review', dict(page['data']['review'], update='Inspect another cell'))
            if step == 0: return call('submit_decision', act(0))
            if step == 1: return call('submit_decision', revised(act(1), c))
            if c['notebook']['latest_review']['id'] == 'experiment-1':
                return call('submit_decision', revised(act(2), c))
            return call('submit_decision', act(0))
        with patch.object(LocalVisionLlm, '_complete', answer):
            r.decide(obs()); ack(r)
            r.decide(obs(1, [[0,0,0,0,0,1]])); ack(r)
            result = r.decide(obs(2, [[0,0,0,0,0,2]]))
        self.assertEqual(result['x'], 2)
        self.assertEqual(r.memory.history[-1]['outcome']['experiment_id'], 'experiment-2')
        self.assertEqual(r.memory.history[-1]['outcome']['review']['experiment_id'], 'experiment-2')

    def test_repetition_after_model_review_stops_without_more_game_actions(self):
        r = self.runtime()
        def answer(model, payload):
            c = context(payload)
            if c['work'] == 'experiment_review':
                return call('submit_review', c['notebook']['active_experiment']['data']['review'])
            return call('submit_decision', act())
        with patch.object(LocalVisionLlm, '_complete', answer):
            r.decide(obs()); ack(r); result = r.decide(obs(1))
        self.assertEqual(result['reason'], 'experiment_redesign_stalled')
        self.assertIsNone(r.memory.pending)
        self.assertEqual(r.experiments.sequence, 1)
        self.assertEqual(r.calls, 3)

    def test_closing_without_observation_marks_interrupted_not_reviewed(self):
        r = self.runtime()
        with patch.object(LocalVisionLlm, '_complete', return_value=call('submit_decision', act())):
            r.decide(obs())
        r.close()
        self.assertEqual(r.notebook.pages['experiment-1']['data']['status'], 'interrupted')
        self.assertIsNone(r.notebook.pages['experiment-1']['data']['review'])
