import json
import unittest
from copy import deepcopy
from scripts.benchmark_rule_inference import build_cases, measure_case, request_for, ARMS
from scripts.benchmark_parallel import digest


class RuleInferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = build_cases()
        cls.measured = {c['name']: measure_case(c, 5000+i*1000) for i, c in enumerate(cls.cases)}

    def test_measurement_preserves_expected_instances_and_panel_cells(self):
        for c in self.cases:
            m = self.measured[c['name']]
            expected = 4 if c['family'] == 'panels' else 1 if c['name'] == 'insufficient_action' else 2
            for s in [m['current'], m['target']] + [t[k] for t in m['trials'] for k in ['before', 'after']]:
                self.assertEqual(len(s['instances']), expected, c['name'])
                if c['family'] == 'panels':
                    self.assertTrue(all(o['class_name'] == 'regular_grid_3x3' and len(o['children']) == 9 for o in s['instances']))

    def test_differences_measure_counter_motion_and_wait_drift(self):
        for v in range(3):
            m = self.measured[f'counter_{v}']
            for t in m['trials']:
                ev = t['changes']['changes']
                if t['action'] == 'WAIT': self.assertEqual(ev, [])
                else:
                    self.assertEqual(len(ev), 2)
                    d = [e['displacement_pixels'] for e in ev]
                    self.assertEqual(d[0]['dx'], -d[1]['dx'])
                    self.assertEqual(d[0]['dy'], -d[1]['dy'])
                    self.assertEqual(abs(d[0]['dx']) + abs(d[0]['dy']), 1)
            for t in self.measured[f'clock_{v}']['trials']:
                if t['action'] == 'WAIT':
                    self.assertEqual(len(t['changes']['changes']), 1)
                    self.assertEqual(t['changes']['changes'][0]['displacement_pixels'], {'dx': 1, 'dy': 0})

    def test_no_truth_or_family_or_images_in_requests_and_paired_base(self):
        for c in self.cases:
            contexts = []
            for arm in ARMS:
                p = request_for(c, self.measured[c['name']], arm)
                text = p['messages'][1]['content']; context = json.loads(text)
                self.assertNotIn('truth', context)
                self.assertNotIn('family', context)
                self.assertNotIn('note', context)
                self.assertNotIn('image_url', json.dumps(p))
                context.pop('current_relations', None); context.pop('target_relations', None); context.pop('option_relations', None)
                for t in context['trials']:
                    t.pop('computed_changes', None); t.pop('computed_relations', None)
                contexts.append(context)
            self.assertEqual(contexts[0], contexts[1]); self.assertEqual(contexts[1], contexts[2])

    def test_options_distinct_and_new_panel_patterns(self):
        for c in self.cases:
            self.assertEqual(len({digest(o['pixels']) for o in c['options']}), len(c['options']))
            if c['family'] == 'panels':
                m = self.measured[c['name']]
                old = {digest(o['pattern']) for t in m['trials'] for k in ['before', 'after'] for o in t[k]['instances']}
                self.assertTrue(all(digest(o['pattern']) not in old for o in m['current']['instances']))

    def test_ambiguous_identity_is_retained(self):
        changes = self.measured['ambiguous_identity']['trials'][0]['changes']['changes']
        self.assertEqual(len(changes), 2)
        self.assertTrue(all(e['kind'] == 'uncertain' and len(e['candidates']) == 2 for e in changes))

    def test_brief_preserves_measured_effects_and_equality(self):
        from scripts.benchmark_rule_brief import brief_trial
        for m in self.measured.values():
            for t in m['trials']:
                b = brief_trial(t)
                self.assertEqual([r['change'] for r in b['changed']], [r['kind'] for r in t['changes']['changes']])
                self.assertEqual([r.get('displacement_pixels') for r in b['changed']], [r.get('displacement_pixels') for r in t['changes']['changes']])
                expected = sum(a['appearance_signature'] == c['appearance_signature'] for i, a in enumerate(t['after']['instances']) for c in t['after']['instances'][i+1:])
                self.assertEqual(len(b['identical_patterns_after']), expected)

    def test_split_removes_other_question_and_retains_same_trials(self):
        from scripts.benchmark_rule_split import request_for as split_request
        from scripts.benchmark_rule_brief import request_for as brief_request
        for c in self.cases:
            m = self.measured[c['name']]
            original = json.loads(brief_request(c, m)['messages'][1]['content'])
            for task in ['prediction', 'action']:
                ctx = json.loads(split_request(c, m, task)['messages'][1]['content'])
                self.assertEqual(ctx['trials'], original['trials'])
                if task == 'action':
                    self.assertNotIn('query_action', ctx); self.assertNotIn('prediction_options', ctx)
                    self.assertEqual(ctx['target'], original['target'])
                else:
                    self.assertNotIn('target', ctx)
                    self.assertEqual(ctx['prediction_options'], original['prediction_options'])


if __name__ == '__main__':
    unittest.main()
