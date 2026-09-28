import unittest
from scripts.probe_visual_emphasis import parse, score


class CoarseProbeTests(unittest.TestCase):
    def test_direction_and_length_reverse_together(self):
        value = dict(orange_blue_change='up', white_cross_relative_to_orange_blue='above_left',
                     white_cross_change='unchanged', yellow_strip_change='shorter', summary='Observed shapes.')
        self.assertTrue(score(value, 'forward')['all_correct'])
        self.assertFalse(score(value, 'reverse')['all_correct'])
        value.update(orange_blue_change='down', yellow_strip_change='longer')
        self.assertTrue(score(value, 'reverse')['all_correct'])

    def test_unchanged_answer_does_not_pass_changed_pair(self):
        value = dict(orange_blue_change='unchanged', white_cross_relative_to_orange_blue='above_left',
                     white_cross_change='unchanged', yellow_strip_change='unchanged', summary='Same shapes.')
        self.assertTrue(score(value, 'same_frame')['all_correct'])
        self.assertFalse(score(value, 'forward')['orange_blue_change'])
        self.assertFalse(score(value, 'forward')['yellow_strip_change'])

    def test_parser_rejects_unknown_direction(self):
        value = dict(orange_blue_change='diagonal', white_cross_relative_to_orange_blue='above_left',
                     white_cross_change='unchanged', yellow_strip_change='shorter', summary='Observed shapes.')
        response = {'choices': [{'finish_reason': 'tool_calls', 'message': {'tool_calls': [
            {'function': {'name': 'submit_coarse_comparison', 'arguments': value}}]}}]}
        with self.assertRaises(ValueError):
            parse(response)
