import unittest
from scripts.motion_evidence import analyze


def board():return [[0]*16 for _ in range(16)]


def put(g,x,y,color,width=2,height=2):
    for yy in range(y,y+height):
        for xx in range(x,x+width):g[yy][xx]=color


class MotionEvidenceTests(unittest.TestCase):
    def test_multicolor_common_motion(self):
        a=board();b=board()
        for g,y in [(a,8),(b,5)]:put(g,5,y,2);put(g,5,y+2,3)
        r=analyze(a,b);self.assertEqual(len(r['common_motion_candidates']),1)
        self.assertEqual((r['common_motion_candidates'][0]['dx'],r['common_motion_candidates'][0]['dy']),(0,-3))

    def test_repeated_shapes_remain_ambiguous(self):
        a=board();b=board();put(a,2,3,4);put(b,6,3,4);put(b,10,3,4)
        r=next(x for x in analyze(a,b)['regions'] if x['color_id']==4)
        self.assertEqual(r['correspondence'],'ambiguous_shape_candidates')
        self.assertEqual(len(r['exact_shape_matches']),2)

    def test_color_change_is_not_translation(self):
        a=board();b=board();put(a,3,3,4);put(b,3,3,5)
        r=next(x for x in analyze(a,b)['regions'] if x['color_id']==4)
        self.assertNotIn('exact_shape_matches',r)

    def test_shrink_and_growth_do_not_become_motion(self):
        a=board();b=board();put(a,2,12,6,8,2);put(b,3,12,6,7,2)
        r=next(x for x in analyze(a,b)['regions'] if x['color_id']==6)
        self.assertEqual(r['same_color_overlaps'][0]['pixel_count_delta'],-2)
        r=next(x for x in analyze(b,a)['regions'] if x['color_id']==6)
        self.assertTrue(r['old_support_unchanged'])
        self.assertEqual(r['same_color_overlaps'][0]['pixel_count_delta'],2)

    def test_separate_objects_with_same_motion_not_merged(self):
        a=board();b=board()
        for g,d in [(a,0),(b,2)]:put(g,2+d,3,2);put(g,9+d,9,3)
        self.assertEqual(analyze(a,b)['common_motion_candidates'],[])

    def test_split_or_occlusion_has_no_exact_match(self):
        a=board();b=board();put(a,3,3,4,5,2);put(b,3,3,4,2,2);put(b,6,3,4,2,2)
        r=next(x for x in analyze(a,b)['regions'] if x['color_id']==4)
        self.assertNotIn('exact_shape_matches',r)

    def test_unchanged_image_and_global_shift(self):
        a=board();put(a,3,3,2);put(a,8,8,3)
        self.assertEqual(analyze(a,a)['changed_pixels'],0)
        b=board();put(b,4,3,2);put(b,9,8,3)
        for r in analyze(a,b)['regions']:
            if r['color_id'] in (2,3):self.assertEqual(r['exact_shape_matches'][0]['dx'],1)


if __name__=='__main__':unittest.main()
