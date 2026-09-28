import unittest
from pathlib import Path
from scripts.benchmark_motion_inputs import crop_boxes, make_cases, parse, request
from scripts.probe_motion_points import to_grid, parse as parse_point


class MotionInputTests(unittest.TestCase):
    def test_crop_covers_both_positions_and_is_order_invariant(self):
        b=[[3]*64 for _ in range(64)];a=[r[:] for r in b]
        b[30][20]=12;a[25][20]=12
        boxes=crop_boxes(b,a)
        self.assertEqual(boxes,crop_boxes(a,b))
        for x,y in [(20,30),(20,25)]:
            self.assertTrue(any(l<=x<r and t<=y<bt for l,t,r,bt in boxes))

    def test_empty_change_falls_back_to_whole_board(self):
        b=[[3]*64 for _ in range(64)]
        self.assertEqual(crop_boxes(b,b),[[0,0,64,64]])

    def test_synthetic_truth_matches_pixel_centroids(self):
        source=Path('outputs/visual-emphasis-comparison-20260927/cases.json')
        if not source.exists():self.skipTest('saved input not available')
        for c in make_cases(source)[3:]:
            def center(g):
                p=[(x,y) for y,row in enumerate(g) for x,v in enumerate(row) if v==12]
                return tuple(sum(q[i] for q in p)/len(p) for i in (0,1))
            b,a=center(c['before']),center(c['after']);dx,dy=a[0]-b[0],a[1]-b[1]
            truth='right' if dx>0 else 'left' if dx<0 else 'down' if dy>0 else 'up' if dy<0 else 'unchanged'
            self.assertEqual(truth,c['truth']['motion'])

    def test_parser_tracks_option_order(self):
        r={'choices':[{'finish_reason':'stop','message':{'content':'B'}}]}
        self.assertEqual(parse(r,['up','down']),'down')
        self.assertEqual(parse(r,['down','up']),'up')
        r['choices'][0]['message']['content']='B or A'
        with self.assertRaises(ValueError):parse(r,['up','down'])

    def test_request_contains_no_truth_or_action_hint(self):
        r=request([],'motion',['up','down','unchanged','uncertain'])
        self.assertNotIn('tools',r)
        self.assertFalse(r['cache_prompt'])
        self.assertNotIn('expected',str(r))

    def test_normalized_cell_center_conversion(self):
        self.assertEqual(to_grid([570.3125,742.1875],'normalized'),[36,47])
        self.assertEqual(to_grid([36,47],'grid'),[36,47])

    def test_point_parser_rejects_nonfinite_and_out_of_range(self):
        for content in ('[NaN, 5]','[1001, 5]','[true, 5]'):
            r={'choices':[{'finish_reason':'stop','message':{'content':content}}]}
            with self.assertRaises(ValueError):parse_point(r,'normalized')


if __name__=='__main__':unittest.main()
