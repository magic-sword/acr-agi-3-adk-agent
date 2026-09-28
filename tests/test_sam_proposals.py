import unittest
import numpy as np
from scripts.benchmark_sam_proposals import normalized_mask_box, ranked
from scripts.benchmark_iterative_proposals import score


class SamProposalTests(unittest.TestCase):
    def test_border_pixel_and_non_square_image(self):
        mask = np.zeros((20, 40), dtype=bool)
        mask[19, 39] = True
        self.assertEqual(normalized_mask_box(mask), [975., 950., 1000., 1000.])
        with self.assertRaises(ValueError):
            normalized_mask_box(np.zeros((20,40),dtype=bool))

    def test_whole_multicolor_support_can_use_one_box(self):
        mask = np.zeros((64,64), dtype=bool)
        mask[20:27,23] = True
        mask[23,20:27] = True
        yy,xx = mask.nonzero()
        gold = dict(id='cross', bbox=[20,20,26,26],support=list(zip(xx.tolist(),yy.tolist())))
        box = normalized_mask_box(mask)
        self.assertEqual(score([box],[gold],padding=0)['tp'],1)
        self.assertEqual(score([[0,0,1000,1000]],[gold])['tp'],0)

    def test_budget_ranking_is_deterministic_and_does_not_mutate_input(self):
        records = [dict(index=i,predicted_iou=q,stability_score=s)
                   for i,q,s in [(0,.9,.99),(1,.95,.96),(2,.95,.98),(3,.95,.98)]]
        order = ranked(records)
        self.assertEqual([r['index'] for r in order],[2,3,1,0])
        self.assertEqual([r['index'] for r in records],[0,1,2,3])
        self.assertEqual(len(order[:16]),4)


if __name__ == '__main__': unittest.main()
