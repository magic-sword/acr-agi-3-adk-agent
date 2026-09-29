"""Validate click experiment geometry and gold-independent input construction."""
from copy import deepcopy
import json
import unittest

from scripts.benchmark_click_choices import (
    decode_union, build_cases, point_for, region_score, shortlist, request_for, rect, encode, decode,
)


class ClickChoicesTests(unittest.TestCase):
    def test_union_numeric_protocol(self):
        mapping={'1':'a','2':'b'}
        self.assertEqual(decode_union('[1,2]',mapping),['1','2'])
        self.assertEqual(decode_union('[]',mapping),[])
        for answer in ('[true]', '[1,1]', '[3]', '["1"]', '{}'):
            with self.assertRaises(AssertionError):
                decode_union(answer,mapping)

    def test_interior_and_centroid_stay_inside_nonconvex_mask(self):
        ring=rect((5,5,25,25))-rect((9,9,21,21))
        for method in ('centroid','interior'):
            self.assertIn(point_for(ring,method),ring)
        # Explicit padding must make image edges boundaries, too.
        self.assertEqual(point_for(rect((0,0,4,4)),'interior'),(2,2))

    def test_accidental_hit_does_not_make_broad_region_correct(self):
        target=rect((4,4,6,6)); broad=rect((0,0,10,10))
        self.assertIn(point_for(broad,'centroid'),target)
        self.assertFalse(region_score(broad,target)['correct'])
        self.assertTrue(region_score(target,target)['correct'])

    def test_shortlist_does_not_insert_missing_target(self):
        candidates=[dict(id='upper',bbox=[2,2,5,5],mask_runs=encode(rect((2,2,5,5))),colors=[9])]
        self.assertEqual(shortlist(candidates,'9'),[])
        self.assertEqual(shortlist(candidates,'X'),[])
        self.assertEqual(shortlist(candidates,'1'),candidates)

    def test_case_counts_annotation_bounds_and_choice_parity(self):
        cases=build_cases()
        self.assertEqual(len(cases),26)
        self.assertEqual(sum(c['reason']=='unique' for c in cases),18)
        self.assertEqual(sum(c['origin']=='real' for c in cases),18)
        for case in cases:
            gold=decode(case['gold_runs'])
            self.assertEqual(bool(gold),case['reason']=='unique')
            self.assertTrue(all(0<=x<64 and 0<=y<64 for x,y in gold))
        case=cases[0]; options=shortlist(case['candidates'],'2')
        a,ma=request_for(case['grid'],case['query'],options,'choice_text',0)
        b,mb=request_for(case['grid'],case['query'],options,'choice_visual',0)
        self.assertEqual(ma,mb)
        self.assertEqual(a['messages'][0],b['messages'][0])
        self.assertEqual(a['messages'][1]['content'][-1],b['messages'][1]['content'][-1])
        # Scoring-only fields cannot enter any request builder.
        changed=deepcopy(case); changed['gold_runs']=[]; changed['reason']='absent'
        other,_=request_for(changed['grid'],changed['query'],options,'choice_text',0)
        self.assertEqual(a,other)
        self.assertNotIn('gold_runs',json.dumps(a))
        _,reordered=request_for(case['grid'],case['query'],options,'choice_text',1)
        self.assertEqual({o['id'] for o in ma.values()},{o['id'] for o in reordered.values()})


if __name__=='__main__': unittest.main()
