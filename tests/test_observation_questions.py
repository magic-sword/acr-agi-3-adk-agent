from copy import deepcopy
import unittest
from scripts.observation_questions import build_questions,grammar,request_for,support
from scripts.benchmark_rule_inference import objects
from scripts.benchmark_one_token import variant


class ObservationQuestionTests(unittest.TestCase):
    def test_every_candidate_gets_each_atomic_question_without_answer_key(self):
        b=objects([[14,16],[40,38]],[9,12]);a=objects([[15,16],[40,38]],[9,12])
        result=build_questions(b,a,'A')
        self.assertEqual(len(result['questions']),6)
        self.assertEqual({q['property'] for q in result['questions']},{'position','appearance','size'})
        self.assertTrue(all(set(q['options'])=={'A','B','X'} and 'truth' not in q for q in result['questions']))
        self.assertEqual(len({q['id'] for q in result['questions']}),6)
        self.assertEqual(result['coverage']['changed_pixels'],6)
        self.assertFalse(result['coverage']['requires_review'])

    def test_missing_large_object_cannot_look_like_no_change(self):
        b=[[5]*64 for _ in range(64)];a=deepcopy(b)
        for y in range(20,40):
            for x in range(20,40):a[y][x]=9
        result=build_questions(b,a,'WAIT')
        self.assertEqual(result['questions'],[])
        self.assertEqual(result['coverage']['uncovered_changed_pixels'],400)
        self.assertTrue(result['coverage']['requires_review'])

    def test_ambiguous_same_color_copies_are_not_assigned_by_order(self):
        r=build_questions(objects([[16,20],[40,20]],[0,0]),objects([[24,20],[32,20]],[0,0]),'A')
        self.assertTrue(r['coverage']['requires_review'])
        self.assertTrue(all(q['context']['correspondence']=='unresolved' for q in r['questions']))
        self.assertTrue(any(len(q['context']['alternative_after_candidates'])==2 for q in r['questions']))

    def test_support_uses_shape_not_box(self):
        obj=dict(bbox=[0,0,2,2],pattern=[[None,0,None],[0,0,0],[None,0,None]])
        self.assertEqual(len(support(obj)),5)
        self.assertNotIn((0,0),support(obj))

    def test_grammar_disallows_injection_or_multichar_labels(self):
        self.assertEqual(grammar(['A','B','X']),'root ::= "A" | "B" | "X"')
        for choices in [[],['A','A'],['AB'],['"'],['Y']]:
            with self.assertRaises(ValueError):grammar(choices)

    def test_output_variants_do_not_change_messages_or_sampling(self):
        p=dict(messages=[dict(role='user',content='Same evidence')],max_tokens=16,temperature=0,cache_prompt=False)
        c=dict(options=[dict(letter='A'),dict(letter='B')])
        for mode in ['baseline16','limit1','grammar1']:
            q=variant(p,c,mode)
            self.assertEqual(q['messages'],p['messages']);self.assertEqual(q['temperature'],0)
            self.assertEqual(q['max_tokens'],16 if mode=='baseline16' else 1)
            self.assertEqual('grammar' in q,mode=='grammar1')
        self.assertEqual(p['max_tokens'],16)

    def test_generated_request_includes_explicit_abstention_and_one_token_cap(self):
        b=objects([[14,16]],[9]);q=build_questions(b,b,'WAIT')['questions'][0];p=request_for(q)
        self.assertEqual(p['max_tokens'],1)
        self.assertIn('"X"',p['grammar'])
        self.assertIn('X. cannot determine',p['messages'][1]['content'])


if __name__=='__main__':unittest.main()
