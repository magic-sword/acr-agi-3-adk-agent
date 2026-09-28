import unittest
from scripts.benchmark_object_proposals import synthetic_cases, box_matches, iou, parse_answer
from agent.cognition.geometry import extract
from scripts.benchmark_target_binding import support


class ObjectProposalTests(unittest.TestCase):
    def test_complete_gold_is_independent_of_geometric_candidates(self):
        cases=synthetic_cases()
        self.assertEqual(len(cases),20)
        for c in cases:
            supports=[{tuple(p) for p in g['support']} for g in c['gold']]
            foreground={(x,y) for y,row in enumerate(c['grid']) for x,v in enumerate(row) if v!=5}
            self.assertEqual(set.union(*supports),foreground)
            self.assertEqual(sum(map(len,supports)),len(foreground))
            measured=extract(c['grid'],'a')
            matched=sum(any(support(o)==s for o in measured['instances']) for s in supports)
            self.assertEqual(matched,4 if c['group'] in ('single_color','two_color_rectangle') else 0)

    def test_matching_is_one_to_one_and_can_reassign(self):
        pred=[[0,0,3,0],[0,0,1,0]];gold=[[0,0,1,0],[2,0,3,0]]
        self.assertEqual(len(box_matches(pred,gold,.5)),2)
        self.assertEqual(len(box_matches([[0,0,1,1]]*2,[[0,0,1,1]],'exact')),1)
        self.assertEqual(iou([0,0,0,0],[0,0,0,0]),1)
        self.assertEqual(iou([0,0,0,0],[1,0,1,0]),0)

    def test_no_silent_coordinate_repair(self):
        with self.assertRaises(ValueError):
            parse_answer('{"objects":[{"description":"cross","bbox":[0,0,64,64]}]}','boxes')
        with self.assertRaises(ValueError):
            parse_answer('{"objects":[{"description":"cross","bbox":[3,0,1,2]}]}','boxes')
        self.assertEqual(parse_answer('{"objects":[]}','concepts'),[])


if __name__=='__main__':unittest.main()
