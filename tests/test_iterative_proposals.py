import unittest
from scripts.benchmark_iterative_proposals import parse_boxes, score, coverage, assign, marked
from PIL import Image


def gold(name, x, y, size=3):
    return dict(id=name,bbox=[x,y,x+size-1,y+size-1],
                support=[[xx,yy] for xx in range(x,x+size) for yy in range(y,y+size)])


def norm(box): return [v/64*1000 for v in box]


class IterativeProposalTests(unittest.TestCase):
    def test_local_search_margin_recovers_coarse_box_but_whole_board_does_not_count(self):
        g=gold('a',20,20)
        b=norm([21,21,22,22])
        self.assertEqual(score([b],[g])['tp'],1)
        self.assertEqual(score([b],[g],padding=0)['tp'],0)
        s=score([[0,0,1000,1000]],[g])
        self.assertEqual(s['any_coverage'],1)
        self.assertEqual(s['tp'],0)

    def test_duplicate_and_merged_boxes_cannot_get_multiple_credits(self):
        gg=[gold('a',10,10),gold('b',14,10)]
        b=norm([10,10,17,13])
        s=score([b],gg)
        self.assertEqual((s['tp'],s['any_coverage'],s['mixed_boxes']),(1,2,1))
        s=score([norm([10,10,13,13])]*2,gg)
        self.assertEqual(s['tp'],1)
        self.assertEqual(s['duplicate_eligible'],1)
        self.assertEqual(len(assign([[0,1],[0]])),2)

    def test_fractional_pixel_coverage_and_edge_clipping(self):
        g=gold('a',0,0,1)
        self.assertAlmostEqual(coverage([0,0,.5,1],g),.5)
        self.assertEqual(score([norm([0,0,1,1])],[g])['tp'],1)

    def test_compact_formats_and_overlay_preserve_original(self):
        self.assertEqual(parse_boxes('[1,2,3,4]',single=True),[[1,2,3,4]])
        self.assertEqual(parse_boxes('[]',single=True),[])
        self.assertEqual(parse_boxes('```json\n[[1,2,3,4]]\n```'),[[1,2,3,4]])
        for s in ('[1,2,3,1001]','[3,2,1,4]','[true,2,3,4]'):
            with self.assertRaises(ValueError):parse_boxes(s,single=True)
        im=Image.new('RGB',(384,384),'black');before=im.tobytes()
        annotated=marked(im,[[100,100,200,200]])
        self.assertEqual(im.tobytes(),before)
        self.assertNotEqual(annotated.tobytes(),before)


if __name__=='__main__':unittest.main()
