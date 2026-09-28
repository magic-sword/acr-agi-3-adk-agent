import unittest
from copy import deepcopy

from scripts.benchmark_delta_recognition import pixel_delta, new_cases


class DeltaRecognitionTests(unittest.TestCase):
    def test_lossless_runs_preserve_old_and_new_colors(self):
        before = [[1, 1, 1, 2], [3, 3, 3, 3]]
        after = [[1, 4, 4, 4], [3, 5, 3, 5]]
        result = pixel_delta(before, after)
        reconstructed = deepcopy(before)
        for y, left, right, old, new in result['runs']:
            for x in range(left, right + 1):
                self.assertEqual(before[y][x], old)
                reconstructed[y][x] = new
        self.assertEqual(reconstructed, after)
        self.assertEqual(result['changed_pixels'], 5)

    def test_unchanged_and_malformed(self):
        self.assertEqual(pixel_delta([[1]], [[1]])['runs'], [])
        with self.assertRaises(ValueError):
            pixel_delta([[1]], [[1, 2]])

    def test_frozen_synthetic_co_response_answers(self):
        # Check the generator's answer keys against actual pixel outcomes,
        # independently of the VLM and its correspondence evidence.
        for c in new_cases():
            if c['group'] != 'coupled_color':
                continue
            signatures = []
            for color in [9, 12, 14]:
                signature = []
                for t in c['trials']:
                    signature.append(any(a != t['after'][y][x] for y, row in enumerate(t['before'])
                                         for x, a in enumerate(row) if a == color))
                signatures.append(signature)
            pairs = [(0, 1), (0, 2), (1, 2)]
            found = [i for i, (a, b) in enumerate(pairs) if signatures[a] == signatures[b] == [True, True, False]]
            expected = ['blue and orange', 'blue and green', 'orange and green'][found[0]] if found else 'none of these pairs'
            self.assertEqual(next(o['text'] for o in c['options'] if o['letter'] == c['truth']), expected)


if __name__ == '__main__':
    unittest.main()
