import unittest
from scripts.benchmark_motion import score, extract


def region(box,after,kind,dx,dy,count):
    return {'appearance':'colored part','before_box':box,'after_box':after,
            'change':kind,'dx':dx,'dy':dy,'pixel_count_delta':count}


class MotionScoringTests(unittest.TestCase):
    def test_correct_component_pair_covers_moving_object(self):
        v={'regions':[region([34,45,38,46],[34,40,38,41],'translation',0,-5,0),
                      region([34,47,38,49],[34,42,38,44],'translation',0,-5,0)]}
        self.assertTrue(score(v,'up')['motion'])

    def test_correct_delta_with_wrong_location_or_whole_board_fails(self):
        for box in [[0,5,63,63],[20,31,22,33]]:
            v={'regions':[region(box,[box[0],box[1]-5,box[2],box[3]-5],'translation',0,-5,0)]}
            self.assertFalse(score(v,'up')['motion'])

    def test_static_control_requires_no_movement(self):
        v={'regions':[region([20,31,22,33],[20,31,22,33],'unchanged',0,0,0)],'prior_role_status':'unverified'}
        self.assertTrue(score(v,'same_frame')['control'])
        v['regions'][0]['dy']=-5
        self.assertFalse(score(v,'same_frame')['control'])

    def test_extract_rejects_out_of_bounds_even_if_schema_helper_is_permissive(self):
        v={'regions':[region([0,0,80,1],[0,0,80,1],'unchanged',0,0,0)],
           'prior_role_status':'no_prior','role_reason':'','uncertainty':''}
        response={'choices':[{'finish_reason':'tool_calls','message':{'tool_calls':[
            {'function':{'name':'submit_motion','arguments':v}}]}}]}
        with self.assertRaisesRegex(ValueError,'invalid box'):extract(response)


if __name__=='__main__':unittest.main()
