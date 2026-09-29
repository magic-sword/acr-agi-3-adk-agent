from copy import deepcopy
import unittest
from unittest.mock import patch
from agent.cognition.perception import measure
from agent.cognition.simple_workflow import SimpleRuntime
from agent.local_vlm import LocalVisionLlm
from runtime_helpers import obs,context,answer,token,ack


def board(x=14):
    g=[[5]*64 for _ in range(64)]
    for y in range(16,19):
        for xx in range(x,x+3):g[y][xx]=9
    return g


class MeasuredPerceptionTests(unittest.TestCase):
    def test_one_pixel_movement_is_measured_without_a_model_and_boundaries_are_not_motion(self):
        r=measure(board(),board(15),after_id='after',before_id='before')
        self.assertEqual(r['changed_pixels'],6)
        self.assertEqual(len(r['changes']),1)
        self.assertEqual(r['changes'][0]['delta_xy'],[1,0])
        self.assertFalse(r['requires_review'])
        initial=measure(board(),board(15),after_id='reset',boundary=True)
        self.assertEqual(initial['status'],'boundary');self.assertFalse(initial['changes'])

    def test_coverage_and_ambiguous_identity_do_not_disappear(self):
        b=[[5]*64 for _ in range(64)];a=deepcopy(b)
        for y in range(20,40):
            for x in range(20,40):a[y][x]=9
        r=measure(b,a,after_id='new')
        self.assertEqual(r['uncovered_changed_pixels'],400);self.assertTrue(r['requires_review'])
        b=board();a=board(15)
        for grid,offset in [(b,40),(a,42)]:
            for y in range(16,19):
                for x in range(offset,offset+3):grid[y][x]=9
        r=measure(b,a,after_id='ambiguous')
        self.assertTrue(r['unresolved']);self.assertFalse(r['changes'])






if __name__=='__main__':unittest.main()
