import unittest
from scripts.preview_change_emphasis import displayed_color, masks


class ChangeEmphasisTests(unittest.TestCase):
    def test_original_colors_remain_exact_on_highlighted_pixels(self):
        self.assertEqual(displayed_color('#ff851b', True, .2), (255, 133, 27))
        self.assertEqual(displayed_color('#333333', False, .2), (214, 214, 214))

    def test_translation_mask_includes_departure_and_arrival(self):
        a = [[0]*8 for _ in range(8)]; b = [[0]*8 for _ in range(8)]
        a[5][3] = 1; b[2][3] = 1
        changed, context = masks(a, b, max_component_pixels=4)
        self.assertEqual(changed, {(3, 5), (3, 2)})
        self.assertEqual(context, changed)

    def test_partial_change_retains_whole_small_part_without_background_flood(self):
        a = [[0]*8 for _ in range(8)]; b = [[0]*8 for _ in range(8)]
        a[5][2:6] = [1]*4; b[5][3:6] = [1]*3
        changed, context = masks(a, b, max_component_pixels=4)
        self.assertEqual(changed, {(2, 5)})
        self.assertEqual(context, {(x, 5) for x in range(2, 6)})

    def test_identical_frames_have_no_highlights(self):
        self.assertEqual(masks([[1, 2]], [[1, 2]]), (set(), set()))

    def test_invalid_geometry_and_opacity_are_rejected(self):
        with self.assertRaises(ValueError): masks([[1, 2]], [[1]])
        with self.assertRaises(ValueError): displayed_color('#ffffff', False, 2)


if __name__ == '__main__':
    unittest.main()
