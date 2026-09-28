"""Guard the diagnostic's scoring boundaries and class/instance separation."""
import unittest

from scripts.benchmark_concept_binding_split import eligible, gold_classes, target_query


class SplitEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.gold = dict(bbox=[10,10,14,14], support=[[x,y] for y in range(10,15) for x in range(10,15)])

    def test_full_and_equivalent_regions(self):
        self.assertTrue(eligible({'bbox':[10,10,14,14]},self.gold))
        self.assertTrue(eligible({'bbox':[9,10,14,14]},self.gold))

    def test_fragment_and_broad_background_rejected(self):
        self.assertFalse(eligible({'bbox':[10,10,11,14]},self.gold))
        self.assertFalse(eligible({'bbox':[0,0,63,63]},self.gold))
        self.assertFalse(eligible({'bbox':[8,8,16,16]},self.gold))

    def test_same_appearance_copies_share_class(self):
        case=dict(origin='synthetic',group='large_cross',gold=[dict(id=f'g{i}') for i in range(4)])
        classes=gold_classes(case)
        self.assertEqual(len(classes),1)
        self.assertEqual(len(classes[0]['members']),4)

    def test_oracle_query_disambiguates_real_copies(self):
        case=dict(origin='ls20',name='ls20')
        queries=[target_query(case,dict(id=f'g{i}',kind='red square')) for i in (4,5,6)]
        self.assertEqual(len(set(queries)),3)
        self.assertTrue(all('red squares' in q for q in queries))


if __name__=='__main__':
    unittest.main()
