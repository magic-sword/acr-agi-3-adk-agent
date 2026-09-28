import unittest
from scripts.benchmark_edgeboxes import normalize_xywh
from scripts.benchmark_iterative_proposals import score


class EdgeBoxesTests(unittest.TestCase):
    def test_xywh_conversion_at_border_and_non_square_shape(self):
        self.assertEqual(normalize_xywh([39,19,1,1],40,20),[975.,950.,1000.,1000.])
        self.assertEqual(normalize_xywh([0,0,40,20],40,20),[0.,0.,1000.,1000.])
        for rect in [[0,0,0,1],[-1,0,2,2],[39,19,2,2]]:
            with self.assertRaises(ValueError):normalize_xywh(rect,40,20)

    def test_display_scale_and_unit_pixel_margin(self):
        gold=dict(id='a',bbox=[10,10,12,12],support=[[x,y] for x in range(10,13) for y in range(10,13)])
        whole=normalize_xywh([60,60,18,18],384,384)
        inner=normalize_xywh([66,66,6,6],384,384)
        self.assertEqual(score([whole],[gold],padding=0)['tp'],1)
        self.assertEqual(score([inner],[gold])['tp'],1)
        self.assertEqual(score([inner],[gold],padding=0)['tp'],0)
        self.assertEqual(score([[0,0,1000,1000]],[gold])['tp'],0)


if __name__=='__main__':unittest.main()
