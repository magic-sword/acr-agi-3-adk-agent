import unittest
from scripts.benchmark_native_grounding import edges,to_board,iou,match,parse


class NativeGroundingTests(unittest.TestCase):
    def test_round_trip_framed_coordinates_and_bare_coordinates(self):
        box=[20,31,22,33];outer=edges(box)
        normalized=[(outer[0]*6+32)/428*1000,(outer[1]*6+44)/452*1000,
                    (outer[2]*6+32)/428*1000,(outer[3]*6+44)/452*1000]
        for actual,expected in zip(to_board(normalized,'normalized_framed'),outer):self.assertAlmostEqual(actual,expected)
        self.assertEqual(to_board([0,0,1000,1000],'normalized_board'),[0,0,64,64])
        self.assertEqual(to_board([0,0,63,63],'legacy'),[0,0,64,64])
        self.assertAlmostEqual(iou(to_board(normalized,'normalized_framed'),outer),1)

    def test_no_clipping_or_coordinate_snapping(self):
        b=to_board([0,0,10,10],'normalized_framed')
        self.assertLess(b[0],0);self.assertLess(b[1],0)
        self.assertEqual(to_board([1,2,3,4],'normalized_board'),[.064,.128,.192,.256])
        with self.assertRaises(ValueError):parse('[{"label":"cross","bbox_2d":[0,0,1001,1000]}]','normalized_board')
        with self.assertRaises(ValueError):parse('[{"label":"cross","bbox_2d":[3,0,1,2]}]','normalized_board')

    def test_one_to_one_matching_and_small_box_convention(self):
        self.assertEqual(len(match([[0,0,4,1],[0,0,2,1]],[[0,0,2,1],[2,0,4,1]],.5)),2)
        self.assertEqual(len(match([[0,0,1,1]]*2,[[0,0,1,1]],.5)),1)
        self.assertEqual(iou(edges([0,0,0,0]),edges([1,0,1,0])),0)
        self.assertAlmostEqual(iou([0,0,.5,.5],[0,0,1,1]),.25)

    def test_native_array_empty_and_whole_markdown_fence(self):
        self.assertEqual(parse('[]','normalized_cross'),[])
        self.assertEqual(len(parse('```json\n[{"label":"cross","bbox_2d":[1,2,3,4]}]\n```','normalized_cross')),1)
        with self.assertRaises(ValueError):parse('prefix [{"label":"cross","bbox_2d":[1,2,3,4]}]','normalized_board')
        with self.assertRaises(ValueError):parse('{"objects":[]}','normalized_board')


if __name__=='__main__':unittest.main()
