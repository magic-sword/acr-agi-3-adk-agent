import unittest
import numpy as np
from scripts.benchmark_sam3_click import masks_to_points,score

class Sam3ClickTests(unittest.TestCase):
    def test_mask_coordinates_return_to_native_grid(self):
        m=np.zeros((1,1,384,384),dtype=bool);m[0,0,12:18,30:36]=True
        self.assertEqual(masks_to_points(m),[{(5,2)}])
        self.assertEqual(masks_to_points(np.zeros((0,1,384,384),dtype=bool)),[])

    def test_best_score_is_not_oracle_target(self):
        r=score([{(3,3)},{(9,9)}],[.9,.8],{(9,9)})
        self.assertTrue(r['available']);self.assertFalse(r['hit']);self.assertFalse(r['region'])
        self.assertEqual(r['selected_index'],0)

    def test_empty_and_ambiguous_choices(self):
        self.assertTrue(score([],[],set())['abstention'])
        r=score([{(3,3)},{(9,9)}],[.9,.8],set())
        self.assertTrue(r['false_click']);self.assertTrue(r['single_only_abstention'])

if __name__=='__main__':unittest.main()
