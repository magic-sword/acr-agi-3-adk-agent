from copy import deepcopy
import unittest
from unittest.mock import patch

from agent.cognition.action_update_comparison import ActionUpdateRuntime, execution_evidence, named_objects
from agent.cognition.memory_comparison import SeparatedMemoryRuntime
from agent.cognition.planning_comparison import OneActionPlan
import test_memory_comparison
from test_planning_comparison import EmptyProposer, plan
from runtime_helpers import obs


class ActionUpdateTests(unittest.TestCase):
    def runtime(self, arm):
        r = ActionUpdateRuntime('test', 'local/qwen3-vl-4b-instruct', update_arm=arm, sam_proposer=EmptyProposer())
        self.addCleanup(r.close)
        return r

    def test_factor_isolation_and_shared_execution(self):
        arms = {a: self.runtime(a) for a in 'ABCD'}
        for r in arms.values():
            r._receive(obs())
            r.obs['observation_id'] = 'CURRENT'
            r.perception['observation_id'] = 'CURRENT'
            r.work = 'ground'
        # Identical evidence, including nondeterministic SAM timing metadata.
        for r in arms.values():
            r.perception = deepcopy(arms['A'].perception)
        for pair in ('AB', 'CD'):
            self.assertEqual(*(arms[a]._agent('ground').instruction for a in pair))
        for pair in ('AC', 'BD'):
            contexts = [arms[a]._context('ground') for a in pair]
            for c in contexts:
                c.pop('seconds_left')
            self.assertEqual(*contexts)
        ref = arms['A']
        for r in arms.values():
            self.assertEqual(r._agent('reconcile').instruction, ref._agent('reconcile').instruction)
            self.assertEqual(r._agent('ground').tools[0]._get_declaration(), ref._agent('ground').tools[0]._get_declaration())
            self.assertEqual(r._visual_parts(), ref._visual_parts())
            self.assertIs(r._fast.__func__, ref._fast.__func__)
            self.assertIs(r.validate_stage.__func__, SeparatedMemoryRuntime.validate_stage)

    def test_names_follow_track_and_never_rebind_a_lost_track(self):
        candidate = dict(id='f1t1', track_id='e1t1', bbox=[0,0,1,1], colors=['white'])
        a = named_objects({'candidates':[candidate]}, [[0]*8 for _ in range(8)])[0]
        candidate.update(id='f2t1', bbox=[2,0,3,1])
        b = named_objects({'candidates':[candidate]}, [[0]*8 for _ in range(8)])[0]
        self.assertEqual(a['name'], b['name'])
        self.assertNotEqual(a['candidate_ref'], b['candidate_ref'])
        candidate.update(id='f3t2', track_id='e1t2')
        c = named_objects({'candidates':[candidate]}, [[0]*8 for _ in range(8)])[0]
        self.assertNotEqual(b['name'], c['name'])
        self.assertEqual(c['role'], 'unknown')

    def test_cursor_report_never_becomes_verified_contact(self):
        for confirmed in (False, True):
            e = execution_evidence(dict(action={'action':'ACTION6'}, acknowledged=True, binding={'confirmed':confirmed}))
            self.assertEqual(e['target_contact'], 'unknown')
            self.assertEqual(e['cursor_report'], 'self_confirmed' if confirmed else 'unconfirmed')

    def test_real_trial_is_recorded_once_and_boundary_clears_ledger(self):
        r = self.runtime('D')
        helper = test_memory_comparison.MemoryComparisonTests()
        helper.execute(r)
        r._accept_stage('reconcile', helper.review(r))
        c = r._context('ground')
        self.assertEqual(len(c['trial_ledger']), 1)
        self.assertEqual(c['trial_ledger'][0]['measured_result']['changed_cell_count'], 0)
        self.assertTrue(c['trial_ledger'][0]['execution']['acknowledged'])
        self.assertNotIn('last_completed_trial', c)
        self.assertNotIn('measurements', c)
        saved = deepcopy(r.trial_ledger)
        r._accept_stage('ground', OneActionPlan.model_validate(plan({'observation_id':r.obs['observation_id']})))
        self.assertEqual(r.trial_ledger, saved)
        r._receive({**obs(2), 'levels_completed':1})
        self.assertEqual(r.trial_ledger, [])


if __name__ == '__main__':
    unittest.main()
