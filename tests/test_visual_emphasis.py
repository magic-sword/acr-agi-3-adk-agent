import json
import unittest

from scripts.benchmark_visual_emphasis import board_image, render, mask_boxes, parse, references, score


def correct(name='forward'):
    b,a=references(name);parts=[]
    for target in b:
        static=name=='same_frame' or target=='white_cross'
        move=target=='orange_blue' and not static
        parts.append({'target':target,'before_box':b[target],'after_box':a[target],
                      'change':'unchanged' if static else 'translation' if move else 'color_or_shape_change',
                      'dx':0 if static or move else None,'dy':0 if static else (5 if name=='reverse' else -5) if move else None,
                      'pixel_delta':0 if static or move else 2 if name=='reverse' else -2,'evidence':'observed geometry'})
    return {'parts':parts,'roles_established':False,'summary':'Roles unverified.'}


class VisualEmphasisTests(unittest.TestCase):
    def test_protected_pixels_are_exact_in_all_renderings(self):
        grid=[[0]*64 for _ in range(64)];grid[20][20]=1;grid[20][21]=2
        palette={'0':'#666666','1':'#ffffff','2':'#ff851b'};mask={(20,20),(21,20)}
        original=render(grid,palette,'original',mask,'test')
        for arm in ('white','outline','blur','numbered'):
            image=render(grid,palette,arm,mask,'test')
            for x,y in mask:
                for yy in range(44+y*6,44+(y+1)*6):
                    for xx in range(32+x*6,32+(x+1)*6):
                        self.assertEqual(image.getpixel((xx,yy)),original.getpixel((xx,yy)),arm)

    def test_outline_keeps_unmarked_static_reference(self):
        grid=[[0]*64 for _ in range(64)];grid[10][10]=1
        palette={'0':'#666666','1':'#ffffff'}
        for arm in ('outline','numbered'):
            im=render(grid,palette,arm,{(30,30)},'test')
            self.assertEqual(im.getpixel((32+10*6+2,44+10*6+2)),(255,255,255))
            self.assertEqual(im.getpixel((32+9*6+2,44+10*6+2)),(102,102,102))

    def test_no_changes_means_no_outline_or_number(self):
        grid=[[0]*64 for _ in range(64)];palette={'0':'#ffffff'}
        base=render(grid,palette,'original',set(),'test').tobytes()
        for arm in ('outline','numbered'):
            self.assertEqual(render(grid,palette,arm,set(),'test').tobytes(),base)

    def test_mask_regions_do_not_merge_disconnected_parts(self):
        self.assertEqual(mask_boxes({(1,1),(1,2),(5,5)}),[[1,1,1,2],[5,5,5,5]])

    def test_correct_forward_reverse_and_duplicate_scores(self):
        for name in ('forward','reverse','same_frame'):
            result=score(correct(name),name)
            self.assertTrue(result['all_three']);self.assertEqual(result['field_conflicts'],0)

    def test_wrong_direction_or_inconsistent_box_fails(self):
        v=correct();v['parts'][0]['dy']=5
        result=score(v,'forward')
        self.assertFalse(result['motion']);self.assertEqual(result['field_conflicts'],1)
        v=correct();v['parts'][0]['after_box']=v['parts'][0]['before_box']
        self.assertFalse(score(v,'forward')['motion'])

    def test_correct_pixel_count_at_wrong_location_does_not_pass(self):
        v=correct();v['parts'][2].update(before_box=[0,0,4,4],after_box=[1,0,4,4])
        self.assertFalse(score(v,'forward')['bar_pixels'])

    def test_parser_rejects_duplicate_targets_and_invalid_boxes(self):
        def response(v):
            return {'choices':[{'finish_reason':'tool_calls','message':{'tool_calls':[
                {'function':{'name':'submit_visual_comparison','arguments':json.dumps(v)}}]}}]}
        v=correct();self.assertEqual(parse(response(v)),v)
        v['parts'][0]['before_box']=[0,0,64,3]
        with self.assertRaises(ValueError):parse(response(v))
        v=correct();v['parts'][0]['target']='white_cross'
        with self.assertRaises(ValueError):parse(response(v))


if __name__=='__main__':unittest.main()
