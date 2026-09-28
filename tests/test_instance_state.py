import unittest
from copy import deepcopy
from scripts.instance_state import extract,compare
from scripts.benchmark_instance_state import extra_cases

def board(block_y=30,cross_x=12):
    g=[[3]*64 for _ in range(64)]
    for y in range(6):
        for x in range(5):g[block_y+y][30+x]=12 if y<3 else 9
    for dx,dy in [(0,0),(-1,0),(1,0),(0,-1),(0,1)]:g[20+dy][cross_x+dx]=0
    return g

class InstanceStateTests(unittest.TestCase):
    def test_independent_measurement_preserves_pixels_and_input(self):
        g=board();old=deepcopy(g);s=extract(g,'b');self.assertEqual(g,old)
        block=next(r for r in s['instances'] if r['class_name']=='multicolor_rectangle')
        self.assertEqual(block['bbox'],[30,30,34,35]);self.assertEqual(block['area'],30)
        self.assertEqual(block['pattern'],[[12]*5]*3+[[9]*5]*3)
        self.assertTrue(all(r['role']=='unknown' for r in s['instances']))

    def test_motion_does_not_need_appearance_change(self):
        r=compare(extract(board(),'b'),extract(board(29),'a'))
        self.assertEqual([x['kind'] for x in r['changes']],['moved'])

    def test_independent_directions_remain_separate(self):
        r=compare(extract(board(),'b'),extract(board(29,13),'a'))
        self.assertEqual([x['kind'] for x in r['changes']],['moved','moved'])
        self.assertEqual(len({x['before_id'] for x in r['changes']}),2)

    def test_unchanged_is_empty(self):
        g=board();r=compare(extract(g,'b'),extract(g,'a'));self.assertEqual(r['changes'],[])

    def test_local_ids_are_not_identity(self):
        b=extract(board(),'b');a=extract(board(29),'a')
        for i,r in enumerate(a['instances']):r['id']='arbitrary'+str(100-i)
        a['instances'].reverse();r=compare(b,a)
        self.assertEqual([x['kind'] for x in r['changes']],['moved'])

    def test_repeated_concept_keeps_instances_and_cell_patterns(self):
        c=extra_cases({})[0];b=extract(c['before'],'b');a=extract(c['after'],'a')
        self.assertEqual(b['classes'],['regular_grid_3x3']);self.assertEqual(len(b['instances']),4)
        self.assertTrue(all(len(r['children'])==9 for r in b['instances']))
        r=compare(b,a);self.assertEqual([x['kind'] for x in r['changes']],['appearance_changed'])
        self.assertEqual(r['same_pattern_after'],[['a1','a4']])

    def test_ambiguous_duplicates_are_not_forced(self):
        c=extra_cases({})[2];r=compare(extract(c['before'],'b'),extract(c['after'],'a'))
        self.assertEqual([x['kind'] for x in r['changes']],['uncertain','uncertain'])
        self.assertTrue(r['needs_image']);self.assertTrue(all(x['after_id'] is None and len(x['candidates'])==2 for x in r['changes']))

if __name__=='__main__':unittest.main()
