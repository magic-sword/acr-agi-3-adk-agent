import json
import unittest
from scripts.hierarchical_questions import encode_flags, flat_result, needs_detail, make_request, measured_flags
from scripts.benchmark_hierarchical import fixtures
from scripts.observation_questions import build_questions


class HierarchicalTests(unittest.TestCase):
    def test_only_explicit_no_skips_detail(self):
        self.assertFalse(needs_detail('A'))
        for answer in ['B','X','',None,'invalid']:
            self.assertTrue(needs_detail(answer))

    def test_simultaneous_changes_survive_combination(self):
        self.assertEqual(encode_flags((True,True,True)), 'H')
        self.assertEqual(flat_result(dict(position='B',appearance='B',size='A')), 'E')
        self.assertEqual(flat_result(dict(position='B',appearance='X',size='A')), 'X')

    def test_fixture_measurement_and_unresolved_routing(self):
        expected={'grow':['G'],'move_grow':['H'],'shrink':['G'],'color':['C'],
                  'move_color_unresolved':['X','X'],'new_unresolved':['X'],
                  'ambiguous_copies':['X']*4,'large_uncovered':[]}
        for f in fixtures():
            r=build_questions(f['before'],f['after'],f['action'])
            contexts=[q['context'] for q in r['questions'] if q['property']=='position']
            self.assertEqual([encode_flags(measured_flags(c)) for c in contexts],expected[f['name']],f['name'])
            if f['name']=='large_uncovered':
                self.assertEqual(r['coverage']['uncovered_changed_pixels'],400)
                self.assertTrue(r['coverage']['requires_review'])
            for c in contexts:
                for stage in ['gate','detail','position','appearance','size']:
                    req=make_request(c,stage)
                    self.assertEqual(req['max_tokens'],1)
                    self.assertIn('"X"',req['grammar'])
                    self.assertNotIn('truth',json.dumps(req))


if __name__=='__main__':unittest.main()
