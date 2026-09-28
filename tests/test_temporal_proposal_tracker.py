import unittest
from copy import deepcopy
from scripts.temporal_proposal_tracker import PixelTracker,native_boxes


def board(objects):
    grid=[[5]*64 for _ in range(64)]
    for x,y,color in objects:
        for yy in range(y,y+3):
            for xx in range(x,x+3):grid[yy][xx]=color
    return grid


class TemporalTrackerTests(unittest.TestCase):
    def test_unique_translation_keeps_id_and_displacement(self):
        t=PixelTracker();t.advance(board([(10,10,9)]));ident=t.tracks[0]['id']
        r=t.advance(board([(14,9,9)]))
        self.assertEqual(t.tracks[0]['id'],ident)
        link=next(x for x in r['links'] if x['id']==ident)
        self.assertEqual(link['before'],[10,10,13,13]);self.assertEqual(link['after'],[14,9,17,12])

    def test_ambiguous_copies_are_not_forced_to_one_identity(self):
        t=PixelTracker();t.advance(board([(10,10,9)]));old=t.tracks[0]['id']
        r=t.advance(board([(9,10,9),(14,10,9)]))
        self.assertGreater(r['ambiguous'],0);self.assertTrue(r['refresh_requested'])
        self.assertFalse(any(x['id']==old for x in r['links']))

    def test_changed_appearance_releases_old_track_and_birth_is_observed(self):
        t=PixelTracker();t.advance(board([(10,10,9)]));old=t.tracks[0]['id']
        r=t.advance(board([(10,10,12),(40,40,8)]))
        self.assertFalse(any(x['id']==old for x in t.tracks))
        self.assertTrue(r['refresh_requested']);self.assertEqual(len(t.tracks),2)

    def test_unchanged_frame_dedup_and_full_image_cannot_hide_local_failure(self):
        g=board([(10,10,9)]);t=PixelTracker();t.advance(g);t.add([[0,0,64,64]],'sam');n=len(t.tracks)
        t.add([[0,0,64,64]],'sam');self.assertEqual(len(t.tracks),n)
        r=t.advance(deepcopy(g));self.assertTrue(r['unchanged']);self.assertFalse(r['refresh_requested'])
        r=t.advance(board([(10,10,12)]));self.assertTrue(r['refresh_requested'])

    def test_normalized_box_rounds_outwards_without_gold(self):
        self.assertEqual(native_boxes([[1,1,999,999]]),[[0,0,64,64]])


if __name__=='__main__':unittest.main()
