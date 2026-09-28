import unittest

from scripts.benchmark_region_descriptions import collapse,select


def obj(ident,box):return dict(id=ident,bbox=box)


class SpatialSelectionTests(unittest.TestCase):
    def setUp(self):
        self.case=dict(candidates=[obj('left',[10,10,14,14]),obj('right',[40,10,44,14]),
            obj('lower',[40,40,44,44]),obj('anchor',[0,25,63,28])])

    def test_quadrant_keeps_identity(self):
        result=select(self.case,dict(class_id='square',filter='quadrant',value='lower right'),
            {'square':['left','right','lower']})
        self.assertEqual(result['selected'],'lower')

    def test_relation_requires_grounded_anchor(self):
        case=dict(candidates=[obj('above',[60,20,63,23]),obj('below',[60,35,63,38]),obj('bar',[20,28,63,31])])
        plan=dict(class_id='square',filter='right_below_anchor',anchor='bar')
        self.assertIsNone(select(case,plan,{'square':['above','below']})['selected'])
        self.assertEqual(select(case,plan,{'square':['above','below'],'bar':['bar']})['selected'],'below')

    def test_duplicates_do_not_merge_separate_copies(self):
        items=[obj('a',[10,10,14,14]),obj('a2',[9,10,14,14]),obj('b',[16,10,20,14])]
        self.assertEqual([o['id'] for o in collapse(items)],['a','b'])

    def test_ambiguous_and_middle_abstain(self):
        matches={'s':['left','right']}
        self.assertIsNone(select(self.case,dict(class_id='s',filter='any'),matches)['selected'])
        self.assertIsNone(select(self.case,dict(class_id='s',filter='any',rank='middle'),matches)['selected'])

    def test_topmost_then_rank(self):
        result=select(self.case,dict(class_id='s',filter='topmost',rank='right'),
            {'s':['left','right','lower']})
        self.assertEqual(result['selected'],'right')


if __name__=='__main__':unittest.main()
