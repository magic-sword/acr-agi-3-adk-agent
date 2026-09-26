"""Real pixels, archive boundaries, image transport and bounded macro execution."""
from copy import deepcopy
import base64
import io
import unittest
from unittest.mock import patch
from PIL import Image
from agent.rendering import render_current, frame_image, ORIGIN, SCALE
from agent.observation import attach_visuals
from agent.cognition.evidence import EvidenceStore
from agent.local_vlm import LocalVisionLlm
from runtime_helpers import obs


class ObservationTests(unittest.TestCase):
    def test_game_only_pixels_have_no_cursor_overlay(self):
        grid=[[0]*64 for _ in range(64)];grid[32][32]=9
        image=render_current(grid)
        expected=frame_image(grid)
        for x,y in ((0,0),(32,32),(33,32),(63,63)):
            self.assertEqual(image.getpixel((ORIGIN[0]+x*SCALE,ORIGIN[1]+y*SCALE)), expected.getpixel((x,y)))
        self.assertLess(image.width,500)

    def test_archive_raw_grid_and_animation_do_not_advance_time(self):
        store=EvidenceStore(capacity=2);self.addCleanup(store.close)
        a=obs();attach_visuals(a,[[[0]*6],[[0,1,0,0,0,0]]]);store.add(a)
        view=store.view(a['observation_id']);self.assertEqual(view['grid'],a['grid'])
        self.assertFalse(view['time_advanced'])
        animation=store.animation(a['animation']['event_id'])
        self.assertFalse(animation['time_advanced']);self.assertIsNone(animation['next_start_frame'])
        b=obs(1,[[1]*6]);store.add(b)
        self.assertEqual(store.compare(a['observation_id'],b['observation_id'])['changed_cell_count'],6)
        c=obs(2);store.add(c,boundary=True)
        with self.assertRaisesRegex(ValueError,'boundary'):store.compare(b['observation_id'],c['observation_id'])
        with self.assertRaisesRegex(ValueError,'evicted'):store.get(a['observation_id'])
