from copy import deepcopy
import unittest
from unittest.mock import patch

from agent.cognition.hybrid_perception import HybridPerception
from agent.cognition.perception import context_record, questions, without_masks
from agent.cognition.workflow import CognitiveRuntime
from agent.local_vlm import LocalVisionLlm
from runtime_helpers import obs, context, answer, token, ack, understanding


def cross(x=12, color=9):
    grid = [[5]*64 for _ in range(64)]
    for y in range(12, 19):
        grid[y][x+3] = color
    for xx in range(x, x+7):
        grid[15][xx] = 8
    return grid


class Proposer:
    def __init__(self, boxes=None, error=None):
        self.calls = 0
        self.boxes = [[12, 12, 19, 19]] if boxes is None else boxes
        self.error = error

    def propose(self, grid):
        self.calls += 1
        if self.error:
            raise self.error
        return dict(boxes=self.boxes, seconds=.01, model='test_provider')


class HybridTests(unittest.TestCase):
    def test_whole_multicolor_region_reaches_motion_and_question_with_grounding(self):
        proposer = Proposer()
        sensor = HybridPerception(proposer)
        initial = sensor.measure(None, cross(), after_id='o0')
        whole = next(c for c in initial['candidates'] if 'sam' in c['sources'])
        sensor.bind_targets(dict(observation_id='o0', targets=[dict(candidate_refs=[whole['id']])]))
        record = sensor.measure(cross(), cross(15), before_id='o0', after_id='o1', action={'action':'RIGHT'})
        change = next(c for c in record['changes'] if 'sam' in c['after']['sources'])
        self.assertEqual(change['delta_xy'], [3, 0])
        self.assertEqual(change['before']['bbox'], [12,12,18,18])
        self.assertEqual(change['after']['bbox'], [15,12,21,18])
        self.assertEqual(change['before']['track_id'], change['after']['track_id'])
        self.assertNotEqual(change['before']['id'], change['after']['id'])
        self.assertEqual(record['target_correspondence'][0]['current_candidate_refs'], [change['after']['id']])
        outcome = dict(acknowledged=True, prediction='The cross moves right.', decision_id='d0')
        queue, _ = questions(record, outcome, None)
        self.assertEqual(queue[0]['change'], without_masks(change))
        self.assertEqual(queue[0]['target_correspondence'][0]['status'], 'tracked')
        sensor.measure(cross(15), cross(15), before_id='o1', after_id='o2')
        self.assertEqual(proposer.calls, 1)

    def test_reset_gap_size_change_and_missing_grid_break_identity(self):
        p = Proposer(); s = HybridPerception(p)
        first = s.measure(None, cross(), after_id='a')['candidates'][0]['track_id']
        for before_id, after_id, boundary in [('a','b',True), ('wrong','c',False)]:
            r = s.measure(cross(), cross(), before_id=before_id, after_id=after_id, boundary=boundary)
            self.assertEqual(r['status'], 'boundary'); self.assertFalse(r['changes'])
            self.assertNotEqual(first, r['candidates'][0]['track_id'])
        self.assertEqual(p.calls, 3)
        p.boxes = [[0,0,2,2]]
        r = s.measure(cross(), [[8,8],[8,8]], before_id='c', after_id='d')
        self.assertEqual(r['status'], 'boundary'); self.assertFalse(r['changes'])
        s.measure([[8,8],[8,8]], None, before_id='d', after_id='e')
        self.assertIsNone(s.tracker)

    def test_ambiguity_and_recolor_do_not_fabricate_motion_or_rebind_targets(self):
        for changed in ('duplicate','recolor'):
            p = Proposer(); s = HybridPerception(p)
            r = s.measure(None, cross(), after_id='a')
            c = next(c for c in r['candidates'] if 'sam' in c['sources'])
            s.bind_targets(dict(observation_id='a', targets=[dict(candidate_refs=[c['id']])]))
            new = cross(color=12) if changed=='recolor' else cross(8)
            if changed=='duplicate':
                other = cross(16)
                for y in range(64):
                    for x in range(64):
                        if other[y][x] != 5: new[y][x] = other[y][x]
            r = s.measure(cross(), new, before_id='a', after_id='b')
            self.assertTrue(r['requires_review'])
            self.assertFalse(any(c['before']['id']==r0 for c in r['changes'] for r0 in [r['target_correspondence'][0]['candidate_refs'][0]]))
            self.assertEqual(r['target_correspondence'][0]['status'], 'unresolved')
            self.assertEqual(p.calls, 1)

    def test_failed_or_skipped_sam_is_explicit_and_not_retried_each_frame(self):
        for p, budget, status in [(Proposer(error=RuntimeError('no CUDA')), None, 'unavailable'),
                                  (Proposer(), 1, 'skipped_budget'),
                                  (Proposer(boxes=[[0,0,65,65]]), None, 'unavailable')]:
            s = HybridPerception(p)
            r = s.measure(None, cross(), after_id='a', seconds_left=budget)
            self.assertEqual(r['sam']['status'], status)
            self.assertTrue(r['proposal_coverage_incomplete']); self.assertTrue(r['candidates'])
            r = s.measure(cross(), cross(13), before_id='a', after_id='b')
            self.assertFalse(r['sam']['called_this_frame'])
            self.assertLessEqual(p.calls, 1)

    def test_inventory_exposes_all_candidates_without_pixel_pattern_flood(self):
        p = Proposer(boxes=[[x,y,x+2,y+2] for y in range(0,32,4) for x in range(0,32,4)])
        s = HybridPerception(p); r = s.measure(None, cross(), after_id='a')
        c = context_record(r, inventory=True)
        self.assertGreater(len(c['candidate_index']), 24)
        self.assertEqual({row[0] for row in c['candidate_index']}, {row['id'] for row in r['candidates']})
        self.assertEqual(c['candidates_omitted'], 0)
        self.assertNotIn('pattern', str(c))

    def test_unchanged_grid_reuses_geometry(self):
        s = HybridPerception(Proposer()); s.measure(None, cross(), after_id='a')
        with patch('agent.cognition.proposal_tracking.extract', side_effect=AssertionError('unneeded extraction')):
            r = s.measure(cross(), deepcopy(cross()), before_id='a', after_id='b')
        self.assertFalse(r['changes']); self.assertFalse(r['requires_review'])

    def test_actual_workflow_passes_whole_region_to_one_token_semantic_step(self):
        p = Proposer()
        r = CognitiveRuntime('test', 'local/qwen3-vl-4b-instruct', proposal_mode='sam_initial', sam_proposer=p)
        self.addCleanup(r.close)
        seen = []
        def respond(model, payload):
            c = context(payload)
            if c['work']=='understand':
                from runtime_helpers import call
                data = understanding(c)
                sam = next(row for row in c['measured_objects']['candidate_index'] if 'sam' in row[3])
                data['targets'][0]['candidate_refs'] = [sam[0]]
                return call('submit_understanding', data)
            if c['work']=='answer_question':
                seen.append(c)
                self.assertEqual(payload['max_tokens'], 1)
                return token('8')
            return answer(model, payload)
        with patch.object(LocalVisionLlm, '_complete', respond):
            r.decide(obs(grid=cross())); ack(r); r.decide(obs(1, cross(15)))
        self.assertTrue(seen)
        self.assertIn('sam', seen[0]['change']['after']['sources'])
        self.assertEqual(seen[0]['change']['delta_xy'], [3,0])
        self.assertEqual(seen[0]['target_correspondence'][0]['status'], 'tracked')
        self.assertEqual(p.calls, 1)
        self.assertEqual(r.semantic_answers[0]['interpretation'], 'unknown')

    def test_late_inventory_reference_is_valid_but_stale_reference_is_rejected(self):
        from agent.cognition.state import Understanding
        p = Proposer(boxes=[[x,y,x+2,y+2] for y in range(0,32,4) for x in range(0,32,4)])
        r = CognitiveRuntime('test', 'local/qwen3-vl-4b-instruct', proposal_mode='sam_initial', sam_proposer=p)
        self.addCleanup(r.close)
        with patch.object(LocalVisionLlm, '_complete', answer):
            r.decide(obs(grid=cross()))
        data = understanding(dict(observation_id=r.obs['observation_id']))
        late = r.perception['candidates'][-1]['id']
        data['targets'][0]['candidate_refs'] = [late]
        r.validate_stage('understand', Understanding.model_validate(data))
        ack(r)
        with patch.object(LocalVisionLlm, '_complete', answer): r.decide(obs(1, cross()))
        data['observation_id'] = r.obs['observation_id']
        with self.assertRaisesRegex(ValueError, 'unknown current'):
            r.validate_stage('understand', Understanding.model_validate(data))

    def test_actual_workflow_reconciles_recolor_instead_of_claiming_motion(self):
        r = CognitiveRuntime('test', 'local/qwen3-vl-4b-instruct',
                             proposal_mode='sam_initial', sam_proposer=Proposer())
        self.addCleanup(r.close)
        reviews=[]; semantic=[]
        def respond(model, payload):
            c=context(payload)
            if c['work']=='reconcile': reviews.append(c['review']['trigger'])
            if c['work']=='answer_question': semantic.append(c)
            return answer(model, payload)
        with patch.object(LocalVisionLlm, '_complete', respond):
            r.decide(obs(grid=cross())); ack(r); r.decide(obs(1,cross(color=12)))
        self.assertIn('perception_uncertain', reviews)
        self.assertFalse(semantic)
        self.assertFalse(r.perception['changes'])

    def test_actual_observation_duplicate_and_reset_do_not_reuse_old_identity(self):
        p=Proposer(); r=CognitiveRuntime('test',proposal_mode='sam_initial',sam_proposer=p)
        self.addCleanup(r.close)
        self.assertTrue(r._receive(obs(grid=cross())))
        old={c['track_id'] for c in r.perception['candidates']}
        self.assertFalse(r._receive(obs(grid=cross())))
        self.assertEqual(p.calls,1)
        new=obs(1,cross()); new['full_reset']=True
        self.assertTrue(r._receive(new))
        self.assertEqual(p.calls,2)
        self.assertEqual(r.perception['status'],'boundary')
        self.assertFalse(old & {c['track_id'] for c in r.perception['candidates']})


if __name__=='__main__': unittest.main()
