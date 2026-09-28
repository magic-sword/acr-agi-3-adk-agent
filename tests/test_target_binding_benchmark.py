"""Check gold regions and fair presentation before spending model calls."""
import json
import unittest

from scripts.benchmark_target_binding import ARMS, build_cases, digest, request_for, support


class TargetBindingBenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = build_cases()

    def test_gold_support_and_abstention_categories(self):
        self.assertEqual(len(self.cases), 66)
        self.assertEqual(sum(c['gold_candidate'] is not None for c in self.cases), 41)
        for c in self.cases:
            if c['gold_candidate']:
                obj = next(o for o in c['candidates'] if o['id'] == c['gold_candidate'])
                self.assertEqual(support(obj), {tuple(p) for p in c['annotated_supports'][0]})
            if c['gold_reason'] == 'fragmented':
                target = {tuple(p) for p in c['annotated_supports'][0]}
                self.assertFalse(any(support(o) == target for o in c['candidates']))

    def test_every_arm_offers_all_candidates_without_gold(self):
        for c in self.cases:
            for permutation in (0, 1):
                mappings = []
                for arm in ARMS:
                    payload, mapping = request_for(c, arm, permutation, {'type': 'image_url', 'image_url': {'url': 'fixture'}})
                    self.assertEqual(set(mapping.values()), {o['id'] for o in c['candidates']})
                    self.assertNotIn('X', mapping)
                    self.assertNotIn('gold_candidate', json.dumps(payload))
                    self.assertNotIn('gold_reason', json.dumps(payload))
                    mappings.append(mapping)
                self.assertEqual(mappings[0], mappings[1])
                self.assertEqual(mappings[0], mappings[2])

    def test_roundtrip_digests_handle_integer_color_keys(self):
        self.assertEqual(digest(self.cases), digest(json.loads(json.dumps(self.cases))))


if __name__ == '__main__':
    unittest.main()
