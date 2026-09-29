import unittest
from copy import deepcopy
from scripts.benchmark_click_cursor import legacy

class HistoricalCursorTests(unittest.TestCase):
    def test_locate_adjust_confirm_geometry_and_pixel_immutability(self):
        api,instructions=legacy();obs=dict(width=64,height=64,observation_id='o',grid=[[0]*64 for _ in range(64)])
        original=deepcopy(obs);cur=api.start_cursor(obs,'target')
        self.assertEqual(cur['stride'],8)
        self.assertNotIn('7',api.cursor_options(cur,obs))
        api.adjust_cursor(cur,api.cursor_options(cur,obs)['2'])
        self.assertEqual((cur['x'],cur['y']),(47,15))
        api.adjust_cursor(cur,api.cursor_options(cur,obs)['5'])
        api.adjust_cursor(cur,api.cursor_options(cur,obs)['3'])
        self.assertEqual((cur['x'],cur['y'],cur['stride']),(43,15,4))
        self.assertEqual(api.cursor_options(cur,obs)['7']['kind'],'click_cursor')
        api.render_cursor(obs,cur);self.assertEqual(obs,original)
        self.assertIn('ambiguous',instructions['aim'])
        cur['observation_id']='stale'
        with self.assertRaises(ValueError):api.render_cursor(obs,cur)

if __name__=='__main__':unittest.main()
