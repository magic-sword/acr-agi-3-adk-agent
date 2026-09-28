import json
import unittest
from unittest.mock import patch

from scripts.benchmark_attention_selection import candidates, parse, rule_select, run_one, score


PALETTE = {'0': '#000000', '1': '#ffffff', '2': '#ff0000', '3': '#00ff00'}


def frame():
    return [[0]*16 for _ in range(16)]


def put(grid, x, y, color, width=2, height=2):
    for yy in range(y, y+height):
        for xx in range(x, x+width):
            grid[yy][xx] = color


def response(value, fast=False):
    message = {'content': json.dumps(value)} if fast else {'tool_calls': [
        {'function': {'name': 'submit_assessments', 'arguments': json.dumps(value)}}]}
    return {'choices': [{'finish_reason': 'stop' if fast else 'tool_calls', 'message': message}],
            'usage': {'completion_tokens': 40}}


def assessment(ids, change='unchanged', more=False):
    return {'assessments': [{'candidate_id': i, 'visible_change': change, 'role_hypothesis': 'unknown',
                            'role_status': 'unknown', 'evidence': 'measured pixels'} for i in ids],
            'relations': 'unknown', 'needs_more_candidates': more}


class AttentionSelectionTests(unittest.TestCase):
    def test_grouping_retains_multicolor_motion_and_separate_stationary_reference(self):
        a, b = frame(), frame()
        for g, y in ((a, 8), (b, 5)):
            put(g, 5, y, 2); put(g, 5, y+2, 3); put(g, 11, 11, 1)
        cs = candidates(a, b, PALETTE)
        moved = [c for c in cs if c['measurement']['kind'] == 'translation']
        self.assertEqual(len(moved), 1)
        self.assertEqual(moved[0]['measurement']['dy'], -3)
        self.assertEqual(len(moved[0]['colors']), 2)

    def test_growing_support_is_not_reported_unchanged(self):
        a, b = frame(), frame(); put(a, 3, 3, 1); put(b, 3, 3, 1, 3, 2)
        c = next(c for c in candidates(a, b, PALETTE) if c['colors'] == [{'id': 1, 'rgb': '#ffffff'}])
        self.assertEqual(c['measurement']['kind'], 'grew')
        self.assertEqual(c['measurement']['pixel_count_delta'], 2)

    def test_repeated_shapes_keep_ambiguity(self):
        a, b = frame(), frame(); put(a, 1, 3, 2); put(b, 5, 3, 2); put(b, 10, 3, 2)
        c = next(c for c in candidates(a, b, PALETTE) if c['colors'] == [{'id': 2, 'rgb': '#ff0000'}])
        self.assertEqual(c['measurement']['kind'], 'uncertain')
        self.assertEqual(len(c['measurement']['alternative_shape_matches']), 2)

    def test_fast_selection_rejects_duplicates_and_unoffered_ids(self):
        for ids in ([1, 1, 2], [1, 2, 9], [True, 2, 3]):
            with self.assertRaises(ValueError): parse(response(ids, True), [1, 2, 3], True)

    def test_deliberation_cannot_reference_unselected_candidate(self):
        with self.assertRaises(ValueError): parse(response(assessment([4])), [1, 2, 3])

    def test_correct_id_with_wrong_direction_is_not_success(self):
        c = {'name': 'up', 'reference_ids': {'motion': [1], 'cross': [2], 'bar': [3]},
             'catalog': [{'id': 1, 'measurement': {'kind': 'translation', 'dx': 0, 'dy': -3}}]}
        s = score(c, assessment([1], 'down'), [1, 2, 3])
        self.assertTrue(s['selected_motion']); self.assertFalse(s['motion'])
        self.assertEqual(s['measurement_conflicts'], 1)

    def test_requested_expansion_is_bounded_and_uses_remaining_tokens(self):
        a = frame(); put(a, 2, 2, 1); put(a, 8, 8, 2); put(a, 12, 12, 3)
        cs = candidates(a, a, PALETTE)
        c = {'name': 'same_frame', 'context': {'palette': PALETTE}, 'images': [{}, {}],
             'catalog': cs, 'reference_ids': {'motion': [1], 'cross': [2], 'bar': [3]}}
        calls = []
        def fake_call(base, p, deadline, trace, stage, timeout_cap=None):
            calls.append((stage, p['max_tokens']))
            return response(assessment(rule_select(cs), more=True)) if stage == 'integrate' else response(assessment([0, 1, 2, 3]))
        with patch('scripts.benchmark_attention_selection.call', side_effect=fake_call):
            r = run_one(c, 'rule', 0, 'unused', {'before': a, 'after': a})
        self.assertEqual(calls, [('integrate', 700), ('expand', 660)])
        self.assertTrue(r['expanded']); self.assertTrue(r['valid'])
        self.assertEqual(set(r['final_ids']), {c['id'] for c in cs})


if __name__ == '__main__':
    unittest.main()
