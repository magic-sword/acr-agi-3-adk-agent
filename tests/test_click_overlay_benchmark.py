import unittest
from scripts.benchmark_click_overlay import request,contextual_image
from scripts.benchmark_click_choices import rect,encode,frame_image

class OverlayTests(unittest.TestCase):
    def test_identical_choices_and_clean_board(self):
        grid=[[0]*64 for _ in range(64)]
        options=[dict(id='a',bbox=[2,3,5,6],colors=[0],mask_runs=encode(rect((2,3,5,6))))]
        a,ma=request(grid,'black square',options,0,'text');b,mb=request(grid,'black square',options,0,'overlay')
        self.assertEqual(ma,mb)
        self.assertEqual(a['messages'][0],b['messages'][0])
        self.assertEqual(a['messages'][1]['content'][0],b['messages'][1]['content'][0])
        self.assertEqual(a['messages'][1]['content'][-1],b['messages'][1]['content'][-1])
        im=contextual_image(grid,ma)
        self.assertEqual(im.getpixel((12+2*4,20+3*4)),(0,255,255))
        self.assertEqual(im.getpixel((12+3*4+1,20+4*4+1)),frame_image(grid).getpixel((3,4)))

if __name__=='__main__':unittest.main()
