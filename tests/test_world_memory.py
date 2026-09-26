"""Standalone geometric memory utilities; not used by the image-led production policy."""
import unittest

from agent.cognition.notebook import Notebook
from agent.cognition.world import WorldMemory, components
from agent.cognition.tasks import WorldInterpretation
from test_skill_learning import obs


def interpretation(world, *, grouped=False, previous=None):
    refs=list(world.candidates)
    groups=[refs] if grouped else [[r] for r in refs]
    return WorldInterpretation(task_id='task', objects=[
        {'key':str(i),'component_ids':g,'description':f'candidate {i}',
         'possible_roles':['control','indicator'],
         'previous_ids':previous or [],'identity_reason':'Reconsider grouping' if previous else ''}
        for i,g in enumerate(groups)], relations=[] if len(groups)==1 else [
            {'subject':'0','other':'1','description':'Possible interaction, not established'}],
        questions=[{'subject':'0','action':'CLICK','observe':['1' if len(groups)>1 else '0'],
            'property':'appearance','question':'Does the first region affect the other?',
            'alternatives':['the observed region changes','no visible effect']}])


class WorldTests(unittest.TestCase):
    def memory(self, grid):
        n=Notebook('test','run');w=WorldMemory(n);w.observe(obs(grid=grid))
        return w

    def test_components_keep_small_regions_and_background_without_claiming_roles(self):
        w=self.memory([[3,3,5,11,5,9]])
        self.assertEqual(sum(c['area'] for c in w.candidates.values()),6)
        self.assertEqual(len(w.candidates),5)
        self.assertEqual(w.objects,{})
        self.assertTrue(all('possible_roles' not in c for c in w.candidate_view()))

    def test_click_must_hit_actual_pixels_not_bounding_box_hole(self):
        w=self.memory([[5,5,5],[5,0,5],[5,5,5]])
        a=interpretation(w);w.apply(a);oid=next(iter(w.objects))
        self.assertTrue(w.contains(oid,0,0))
        self.assertFalse(w.contains(oid,1,1))

    def test_original_bar_miss_is_rejected_by_pixel_mask(self):
        grid=[[3]*64 for _ in range(64)]
        for y in range(28,32):
            for x in range(20,64):grid[y][x]=5
        w=self.memory(grid);w.apply(interpretation(w))
        oid=next(k for k,o in w.objects.items() if o['components'][0]['color']==5)
        self.assertTrue(w.contains(oid,24,30))
        self.assertFalse(w.contains(oid,24,32))

    def test_bad_component_reference_cannot_mutate_world(self):
        w=self.memory([[0,1]]);a=interpretation(w)
        a.objects[0].component_ids=['invented']
        with self.assertRaisesRegex(ValueError,'current component'):
            w.apply(a)
        self.assertEqual(w.objects,{})
        self.assertNotIn('world',w.notebook.pages)

    def test_split_merge_preserves_provenance_and_revises_grouping(self):
        w=self.memory([[0,1]]);w.apply(interpretation(w,grouped=True))
        first=list(w.objects)
        w.apply(interpretation(w,previous=first))
        self.assertEqual(len(w.objects),2)
        self.assertTrue(all(o['previous_ids']==first for o in w.objects.values()))
        self.assertTrue(all(o['status']=='hypothesis' for o in w.objects.values()))
        self.assertEqual(w.notebook.pages['world']['revision'],2)

    def test_question_identity_ignores_paraphrase_and_hud_change(self):
        w=self.memory([[7,7,7],[5,5,3]]);a=interpretation(w)
        a.questions[0].subject='1';a.questions[0].observe=['1']
        w.apply(a);qid=next(iter(w.questions));w.questions[qid]['status']='tested'
        # Top row changes; the tested black component is unchanged.
        w.observe(obs(1,[[7,7,4],[5,5,3]]));b=interpretation(w)
        black=next(str(i) for i,c in enumerate(w.candidates.values()) if c['color']==5)
        b.questions[0].subject=black;b.questions[0].observe=[black]
        b.questions[0].question='Reworded question about the same response'
        w.apply(b)
        self.assertEqual(w.questions[qid]['status'],'tested')
        self.assertTrue(w.questions[qid]['current'])

    def test_small_change_to_tested_object_does_not_erase_its_question_history(self):
        w=self.memory([[7]*12]);w.apply(interpretation(w));qid=next(iter(w.questions))
        w.questions[qid]['status']='tested';oid=next(iter(w.objects))
        w.observe(obs(1,[[7]*11+[4]]));a=interpretation(w);a.questions[0].observe=['0'];w.apply(a)
        self.assertIn(oid,w.objects)
        self.assertEqual(w.questions[qid]['status'],'tested')
        self.assertTrue(w.questions[qid]['current'])
        self.assertEqual(w.candidate_view()[0]['correspondence'],'overlap_hypothesis')

    def test_split_components_do_not_claim_the_same_identity(self):
        w=self.memory([[5]*5]);old=set(w.candidates)
        w.observe(obs(1,[[5,5,0,5,5]]))
        self.assertFalse(old & w.candidates.keys())
        self.assertEqual(len(w.candidates),3)

    def test_unseen_pixels_are_counted_when_context_is_bounded(self):
        w=self.memory([[(x+y)%2 for x in range(12)] for y in range(12)])
        self.assertEqual(len(w.candidates),64)
        self.assertEqual(w.omitted,80)

    def test_control_and_context_are_part_of_question_identity(self):
        w=self.memory([[0,1]]);w.legal_actions=['CLICK','ACT']
        a=interpretation(w);w.apply(a);first=next(iter(w.questions))
        a.questions[0].action='ACT';w.apply(a)
        second=next(q['id'] for q in w.questions.values() if q['current'])
        self.assertNotEqual(first,second)
        a.questions[0].context=['1'];w.apply(a)
        third=next(q['id'] for q in w.questions.values() if q['current'])
        self.assertNotEqual(second,third)

    def test_unknown_receipt_boundary_and_inconclusive_do_not_resolve_question(self):
        for acknowledged,boundary,verdict in [(False,'','unsupported'),(True,'level','unsupported'),
                                                (True,'','inconclusive')]:
            with self.subTest(acknowledged=acknowledged,boundary=boundary,verdict=verdict):
                w=self.memory([[0]]);w.apply(interpretation(w))
                qid=next(iter(w.questions));oid=next(iter(w.objects))
                w.bind('experiment-1',qid,oid,obs())
                w.record({'id':'experiment-1','data':{
                    'outcome':{'acknowledged':acknowledged,'boundary':boundary},
                    'review':{'verdict':verdict,'finding':'No supported conclusion','evidence_ids':['experience-1']},
                    'action':{'action':'ACTION6','x':0,'y':0},
                    'plan':{'conditions':'current frame','expected':{'kind':'region_changed'}},
                    'after_id':'o1'}})
                self.assertFalse(w.findings[0]['resolved_test'])
                self.assertEqual(w.questions[qid]['status'],'open')

    def test_boundary_discards_local_claims_but_keeps_versioned_notebook(self):
        w=self.memory([[0,1]]);w.apply(interpretation(w));old=w.notebook.pages['world']
        w.notebook.begin_segment('level');w.observe(obs(1,[[0,1]]))
        self.assertEqual(w.objects,{})
        self.assertEqual(w.findings,[])
        self.assertEqual(old['segment'],0)
        self.assertIsNone(w.notebook.opening()['world'])



if __name__ == '__main__':
    unittest.main()
