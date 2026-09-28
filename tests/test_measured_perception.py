from copy import deepcopy
import unittest
from unittest.mock import patch
from agent.cognition.perception import measure,questions
from agent.cognition.workflow import CognitiveRuntime
from agent.local_vlm import LocalVisionLlm
from runtime_helpers import obs,context,answer,token,ack,understanding


def board(x=14):
    g=[[5]*64 for _ in range(64)]
    for y in range(16,19):
        for xx in range(x,x+3):g[y][xx]=9
    return g


class MeasuredPerceptionTests(unittest.TestCase):
    def test_one_pixel_movement_is_measured_without_a_model_and_boundaries_are_not_motion(self):
        r=measure(board(),board(15),after_id='after',before_id='before')
        self.assertEqual(r['changed_pixels'],6)
        self.assertEqual(len(r['changes']),1)
        self.assertEqual(r['changes'][0]['delta_xy'],[1,0])
        self.assertFalse(r['requires_review'])
        initial=measure(board(),board(15),after_id='reset',boundary=True)
        self.assertEqual(initial['status'],'boundary');self.assertFalse(initial['changes'])

    def test_coverage_and_ambiguous_identity_do_not_disappear(self):
        b=[[5]*64 for _ in range(64)];a=deepcopy(b)
        for y in range(20,40):
            for x in range(20,40):a[y][x]=9
        r=measure(b,a,after_id='new')
        self.assertEqual(r['uncovered_changed_pixels'],400);self.assertTrue(r['requires_review'])
        b=board();a=board(15)
        for grid,offset in [(b,40),(a,42)]:
            for y in range(16,19):
                for x in range(offset,offset+3):grid[y][x]=9
        r=measure(b,a,after_id='ambiguous')
        self.assertTrue(r['unresolved']);self.assertFalse(r['changes'])

    def test_question_limits_preserve_pending_and_do_not_ask_about_unchanged(self):
        r=measure(board(),board(),after_id='same')
        outcome=dict(acknowledged=True,prediction='Move the blue square.',decision_id='decision')
        self.assertEqual(questions(r,outcome,None),([],0))
        r=measure(board(),board(15),after_id='changed');r['changes']*=6
        q,pending=questions(r,outcome,None)
        self.assertEqual((len(q),pending),(4,2))
        self.assertEqual(q[0]['observation_id'],'changed')
        self.assertEqual(q[0]['choices']['8']['kind'],'unknown')

    def test_actual_runtime_routes_one_token_question_and_keeps_measured_fact(self):
        r=CognitiveRuntime('test','local/qwen3-vl-4b-instruct')
        self.addCleanup(r.close);requests=[]
        def respond(m,p):
            c=context(p);requests.append(c['work'])
            if c['work']=='answer_question':
                self.assertEqual(p['max_tokens'],1)
                self.assertTrue(all(x['type']=='text' for x in p['messages'][1]['content']))
                self.assertTrue(c['change']['changed']['position'])
                return token('8')
            if c['work']=='reconcile':
                self.assertEqual(c['semantic_answers'][0]['interpretation'],'unknown')
                self.assertTrue(c['measured_objects']['changes'][0]['changed']['position'])
            return answer(m,p)
        with patch.object(LocalVisionLlm,'_complete',respond):
            r.decide(obs(grid=board()));ack(r);r.decide(obs(1,board(15)))
        self.assertEqual(requests.count('answer_question'),1)
        self.assertIn('reconcile',requests)
        self.assertTrue(r.perception['changes'][0]['changed']['position'])

    def test_no_change_still_reconciles_an_acknowledged_probe(self):
        r=CognitiveRuntime('test','local/qwen3-vl-4b-instruct')
        self.addCleanup(r.close)
        with patch.object(LocalVisionLlm,'_complete',answer):r.decide(obs(grid=board()))
        r.memory.plan['intent']='probe';ack(r)
        works=[]
        def respond(m,p):
            works.append(context(p)['work']);return answer(m,p)
        with patch.object(LocalVisionLlm,'_complete',respond):r.decide(obs(1,board()))
        self.assertIn('reconcile',works);self.assertNotIn('answer_question',works)

    def test_target_reference_must_be_supplied_current_candidate(self):
        from agent.cognition.state import Understanding
        r=CognitiveRuntime('test');self.addCleanup(r.close)
        r.decide(obs(grid=board()))
        value=understanding(dict(observation_id=r.obs['observation_id']))
        value['targets'][0]['candidate_refs']=['a999']
        with self.assertRaisesRegex(ValueError,'unknown current'):
            r.validate_stage('understand',Understanding.model_validate(value))


if __name__=='__main__':unittest.main()
