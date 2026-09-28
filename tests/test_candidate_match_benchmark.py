import unittest

from scripts.benchmark_candidate_match import annotate, aggregate, probabilities, gated
from scripts.benchmark_target_binding import digest


class CandidateMatchBenchmarkTests(unittest.TestCase):
    def test_gold_distinguishes_multiple_matches_from_missing_role_evidence(self):
        candidates = [dict(id=f'a{x}', bbox=[x, 0, x, 0], pattern=[[1]]) for x in (0, 2)]
        source = [dict(name=f'board-{i}', origin='synthetic', grid=[[i]],
                       candidates=candidates, annotated_supports=[[(0, 0)], [(2, 0)]],
                       gold_reason='ambiguous', gold_candidate=None) for i in range(15)]
        source += [dict(source[0], name='role', gold_reason='role_unknown', annotated_supports=[]),
                   dict(source[0], name='fragment', gold_reason='fragmented',
                        annotated_supports=[[(0, 0), (2, 0)]]),
                   dict(source[0], name='unique', gold_reason='unique', gold_candidate='a0',
                        annotated_supports=[[(0, 0)]]),
                   dict(source[0], name='absent', gold_reason='absent', annotated_supports=[])]
        cases = annotate(source)
        byname = {c['name']: c for c in cases}
        multiple = byname['board-0']['candidate_gold']
        self.assertEqual(sum(v == 'M' for v in multiple.values()), 2)
        self.assertEqual(aggregate(multiple)['status'], 'multiple')
        role = byname['role']['candidate_gold']
        self.assertEqual(set(role.values()), {'U'})
        self.assertEqual(set(byname['fragment']['candidate_gold'].values()), {'N'})
        self.assertEqual(byname['unique']['candidate_gold'], {'a0': 'M', 'a2': 'N'})
        self.assertEqual(set(byname['absent']['candidate_gold'].values()), {'N'})
        self.assertEqual({c['split'] for c in cases}, {'calibration', 'test'})
        self.assertFalse({digest(c['grid']) for c in cases if c['split'] == 'calibration'} &
                         {digest(c['grid']) for c in cases if c['split'] == 'test'})

    def test_missing_probability_is_not_filled_with_zero(self):
        choice = {'logprobs': {'content': [{'token': '1', 'logprob': -0.1,
                    'top_logprobs': [{'token': '1', 'logprob': -0.1}, {'token': '2', 'logprob': -5}]}]}}
        p = probabilities(choice, 0)
        self.assertFalse(p['complete']); self.assertEqual(p['missing'], ['8'])
        self.assertEqual(gated({'valid': True, 'decision': 'M', 'probabilities': p}, 0.9), 'U')

    def test_probability_normalization_is_stable_and_encoding_aware(self):
        choice = {'logprobs': {'content': [{'token': '8', 'logprob': -1000,
                    'top_logprobs': [{'token': k, 'logprob': -1000} for k in ('1', '2', '8')]}]}}
        for e in (0, 1):
            p = probabilities(choice, e)
            self.assertTrue(p['complete'])
            self.assertAlmostEqual(p['normalized']['M'], 1/3)
        choice['logprobs']['content'][0]['top_logprobs'][2]['logprob'] = -999
        self.assertGreater(probabilities(choice, 1)['normalized']['M'], probabilities(choice, 0)['normalized']['M'])

    def test_unknown_distractor_prevents_unqualified_single_binding(self):
        result = aggregate({'a1': 'M', 'a2': 'U'})
        self.assertEqual(result['matches'], ['a1'])
        self.assertEqual(result['status'], 'information_insufficient')
        self.assertEqual(aggregate({'a1': 'N'})['status'], 'no_complete_match')


if __name__ == '__main__':
    unittest.main()
