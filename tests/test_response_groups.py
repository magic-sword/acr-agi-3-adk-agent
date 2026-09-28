from copy import deepcopy
import unittest

from scripts.analyze_response_groups import response_groups


class ResponseGroupsTests(unittest.TestCase):
    def test_co_response_never_claims_identity_or_requires_translation(self):
        trials=[dict(before=[[1,2]],action=a) for a in ['A','WAIT']]
        base=dict(before=dict(regions=[['a',[0,0,0,0],{'1':1}],['b',[1,0,1,0],{'2':1}]]),
                  delta=dict(runs=[[0,0,0,1,3],[0,1,1,2,4]]),tracking=dict(links=[]))
        quiet=deepcopy(base);quiet['delta']['runs']=[]
        result=response_groups(trials,[base,quiet])
        self.assertEqual(len(result['groups']),1)
        self.assertEqual(result['groups'][0]['kind'],'co_response')
        self.assertEqual(result['groups'][0]['members'],['a','b'])
        self.assertTrue(all(s['translations']==[None,None] for s in result['signatures']))

    def test_no_group_across_unaligned_trials_or_static_responses(self):
        self.assertEqual(response_groups([dict(before=[[1]]),dict(before=[[2]])],[])['status'],
                         'incomparable_or_single_trial')
        trials=[dict(before=[[1,2]],action=a) for a in ['A','WAIT']]
        record=dict(before=dict(regions=[['a',[0,0,0,0],{'1':1}],['b',[1,0,1,0],{'2':1}]]),
                    delta=dict(runs=[]),tracking=dict(links=[]))
        self.assertEqual(response_groups(trials,[record,record])['groups'],[])

    def test_matching_translation_requires_known_correspondence(self):
        trials=[dict(before=[[1,2,3]],action=a) for a in ['A','WAIT']]
        record=dict(before=dict(regions=[[name,[i,0,i,0],{str(i+1):1}] for i,name in enumerate(['a','b','c'])]),
                    delta=dict(runs=[]),tracking=dict(links=[dict(before=[i,0,i,0],delta_xy=[2,0]) for i in [0,1]]))
        quiet=deepcopy(record)
        for link in quiet['tracking']['links']:link['delta_xy']=[0,0]
        groups=response_groups(trials,[record,quiet])['groups']
        self.assertEqual(len(groups),1)
        self.assertEqual(groups[0]['kind'],'same_translation')
        self.assertEqual(groups[0]['members'],['a','b'])


if __name__=='__main__':
    unittest.main()
