"""Actual ADK dispatch with scripted HTTP: stage dataflow and acknowledged control."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from agent.cognition.workflow import CognitiveRuntime
from agent.local_vlm import LocalVisionLlm
from runtime_helpers import obs, context, call, token, skill, understanding, backchain, grounding, reconciliation, answer, ack


class FastSlowTests(unittest.TestCase):
    def runtime(self, **kwargs):
        r=CognitiveRuntime('test','local/qwen3-vl-4b-instruct',**kwargs)
        self.addCleanup(r.close)
        return r

    def test_three_stage_dataflow_then_single_token_execution_without_repeat_guard(self):
        r=self.runtime();requests=[]
        def respond(m,p):
            requests.append(p);c=context(p)
            if c['work']=='backchain':
                self.assertEqual(c['understanding']['targets'][0]['role_hypothesis'],'actor')
                schema=p['tools'][0]['function']['parameters']
                self.assertIn('target_query',schema['$defs']['Goal']['properties'])
                self.assertNotIn('target_ids',schema['$defs']['Goal']['properties'])
            if c['work']=='ground':self.assertEqual(c['current_goal']['id'],'approach')
            return answer(m,p)
        with patch.object(LocalVisionLlm,'_complete',respond):
            for step in range(6):
                result=r.decide(obs(step));self.assertEqual(result['status'],'action');ack(r)
        self.assertEqual([context(p)['work'] for p in requests if context(p)['work']!='read_memory'],
                         ['understand','backchain','ground','choose_skill']+['execute_step']*6)
        for p in requests[3:]:
            self.assertEqual(p['max_tokens'],1);self.assertNotIn('tools',p)
            self.assertIn('"8"',p['grammar']);self.assertNotIn('causal_notes',context(p))
        self.assertEqual(r.calls,1)
        self.assertEqual(r.memory.active_skill['action_count'],5)

    def test_eight_reconciles_before_regrounding_and_keeps_goal_and_evidence(self):
        r=self.runtime();executions=0;works=[]
        def respond(m,p):
            nonlocal executions
            c=context(p);works.extend([c['work']] if c['work']!='read_memory' else [])
            if c['work']=='execute_step':
                executions+=1
                return token('8' if executions==2 else '1')
            if c['work']=='reconcile':
                self.assertEqual(c['review']['trigger'],'unexpected')
                self.assertEqual(c['current_invocation_result']['changed_cell_count'],1)
                self.assertEqual(c['current_goal']['id'],'approach')
                self.assertIsNone(r.memory.pending)
            if c['work']=='ground' and c['last_review']:
                self.assertIn('not yet',c['last_review'][-1]['causal_notes'])
                return call('submit_grounding',grounding(c,x=2))
            return answer(m,p)
        with patch.object(LocalVisionLlm,'_complete',respond):
            r.decide(obs());ack(r);result=r.decide(obs(1,[[0,0,0,0,0,1]]))
        self.assertEqual(result['x'],2)
        self.assertEqual(works.count('understand'),1);self.assertEqual(works.count('backchain'),1)
        self.assertEqual(works[-7:],['reconcile','ground','choose_skill','execute_step','aim','aim','aim'])

    def test_completion_candidate_is_not_confirmed_and_same_name_has_fresh_invocation(self):
        r=self.runtime();executions=0;old_id=None
        def respond(m,p):
            nonlocal executions
            c=context(p)
            if c['work']=='execute_step':
                executions+=1
                if executions==2:return token('7')
                if executions==3:
                    self.assertIsNone(c['last_result'])
                    self.assertNotEqual(c['active_skill']['invocation_id'],old_id)
                    self.assertEqual(c['active_skill']['action_count'],0)
            if c['work']=='reconcile':
                self.assertEqual(c['goal_status']['status'],'candidate')
                self.assertEqual(c['review']['trigger'],'completion_candidate')
                self.assertEqual(len(c['attempt_results']),1)
            return answer(m,p)
        with patch.object(LocalVisionLlm,'_complete',respond):
            r.decide(obs());old_id=r.memory.active_skill['invocation_id'];ack(r)
            self.assertEqual(r.decide(obs(1))['status'],'action')
        self.assertEqual(r.memory.goal_status['approach']['status'],'active')
        self.assertEqual(r.memory.reconciliations[-1]['goal_status'],'active')

    def test_probe_bypasses_backchain_and_receipt_returns_to_reconcile_without_fast_call(self):
        r=self.runtime();works=[]
        def respond(m,p):
            c=context(p);works.extend([c['work']] if c['work']!='read_memory' else [])
            if c['work']=='understand':
                v=understanding(c);v['next']='ground';return call('submit_understanding',v)
            if c['work']=='ground':
                v=grounding(c);v['plan'].update(intent='probe',question='Does the square move?')
                return call('submit_grounding',v)
            if c['work']=='execute_step':
                self.assertNotIn('"7"',p['grammar'])
                self.assertIn('"8"',p['grammar'])
            if c['work']=='reconcile':
                self.assertEqual(c['review']['trigger'],'probe_result')
                self.assertFalse(c['current_invocation_result']['frame_changed'])
                properties=p['tools'][0]['function']['parameters']['properties']
                self.assertNotIn('resume',properties['next']['enum'])
                self.assertNotIn('confirmed',properties['goal_status']['enum'])
                v=reconciliation(c);v['assessment']='probe_result';return call('submit_reconciliation',v)
            return answer(m,p)
        with patch.object(LocalVisionLlm,'_complete',respond):
            r.decide(obs());ack(r);r.decide(obs(1))
        self.assertEqual(works,['understand','ground','choose_skill','execute_step',
                               'reconcile','ground','choose_skill','execute_step'])
        self.assertEqual(r.memory.goal_status,{})

    def test_probe_result_cannot_be_promoted_to_confirmed_parent_goal(self):
        r=self.runtime()
        def respond(m,p):
            c=context(p)
            if c['work']=='ground':
                v=grounding(c);v['plan'].update(intent='probe',question='Does a click move it?')
                return call('submit_grounding',v)
            if c['work']=='reconcile':
                v=reconciliation(c);v['goal_status']='confirmed';return call('submit_reconciliation',v)
            return answer(m,p)
        with patch.object(LocalVisionLlm,'_complete',respond):
            r.decide(obs());ack(r);result=r.decide(obs(1))
        self.assertEqual(result['reason'],'stage_output_invalid')
        self.assertEqual(r.memory.goal_status['approach']['status'],'active')
        self.assertEqual(r.memory.reconciliations,[])

    def test_intermediate_step_advances_without_slow_call_and_scopes_previous_result(self):
        r=self.runtime();fast=iter(['1','1','7','1'])
        def respond(m,p):
            c=context(p)
            if c['work']=='ground':
                v=grounding(c);v['plan']['skills'][0]['steps'].append(skill(x=2)['steps'][0])
                return call('submit_grounding',v)
            if c['work'] in ('choose_skill','execute_step'):
                if c.get('active_skill',{}).get('index')==1:
                    self.assertIsNone(c['last_result']);self.assertEqual(c['active_skill']['step_action_count'],0)
                return token(next(fast))
            return answer(m,p)
        with patch.object(LocalVisionLlm,'_complete',respond):
            r.decide(obs());ack(r);self.assertEqual(r.decide(obs(1))['x'],2)
        self.assertEqual(r.memory.model_calls,10);self.assertEqual(r.memory.reconciliations,[])

    def test_reconcile_can_resume_same_invocation(self):
        r=self.runtime();executions=0
        def respond(m,p):
            nonlocal executions
            c=context(p)
            if c['work']=='execute_step':
                executions+=1;return token('8' if executions==2 else '1')
            if c['work']=='reconcile':
                v=reconciliation(c);v['next']='resume';return call('submit_reconciliation',v)
            return answer(m,p)
        with patch.object(LocalVisionLlm,'_complete',respond):
            r.decide(obs());old=r.memory.active_skill['invocation_id'];ack(r);r.decide(obs(1))
        self.assertEqual(r.memory.active_skill['invocation_id'],old)
        self.assertEqual(r.calls,4)  # read_memory, reconciliation and two execution choices

    def test_confirmed_leaf_is_preserved_when_backchaining_to_parent(self):
        r=self.runtime();executions=0;backchains=0
        def respond(m,p):
            nonlocal executions,backchains
            c=context(p)
            if c['work']=='execute_step':
                executions+=1;return token('7' if executions==2 else '1')
            if c['work']=='reconcile':
                v=reconciliation(c);v.update(goal_status='confirmed',next='backchain',assessment='matched')
                return call('submit_reconciliation',v)
            if c['work']=='backchain':
                backchains+=1;v=backchain(c)
                if backchains==2:
                    self.assertEqual(c['goals']['approach']['assessment']['status'],'confirmed')
                    v['selected_goal_id']='exit'
                return call('submit_backchain',v)
            return answer(m,p)
        with patch.object(LocalVisionLlm,'_complete',respond):
            r.decide(obs());ack(r);r.decide(obs(1))
        self.assertEqual(r.memory.selected_goal_id,'exit')
        self.assertEqual(r.memory.goal_status['approach']['status'],'confirmed')

    def test_invalid_stage_results_are_atomic_and_repaired_once(self):
        for bad in ('stale','target','parent_cycle','dependency_cycle','unknown_target','selected_goal','coordinate','candidate'):
            with self.subTest(bad=bad):
                r=self.runtime()
                def respond(m,p):
                    c=context(p);work=c['work']
                    if work=='understand':
                        v=understanding(c)
                        if bad=='stale':v['observation_id']='old'
                        if bad=='target':v['targets'][0]['x']=63
                        return call('submit_understanding',v)
                    if work=='backchain':
                        v=backchain(c)
                        if bad=='parent_cycle':v['goals'][0]['parent_id']='approach'
                        if bad=='dependency_cycle':v['goals'][1]['requires']=['exit']
                        if bad=='unknown_target':v['goals'][0]['target_query']=''
                        if bad=='selected_goal':v['selected_goal_id']='missing'
                        return call('submit_backchain',v)
                    if work=='ground':
                        v=grounding(c)
                        if bad=='coordinate':v['plan']['skills'][0]['steps'][0]['options'][0]['action']['x']=63
                        if bad=='candidate':v['plan']['reuse']=['missing']
                        return call('submit_grounding',v)
                    return answer(m,p)
                with patch.object(LocalVisionLlm,'_complete',respond):result=r.decide(obs())
                self.assertEqual(result['reason'],'stage_output_invalid')
                self.assertEqual(r.memory.skills,{})
                if bad in ('stale','target'):self.assertIsNone(r.memory.understanding)
                elif bad not in ('coordinate','candidate'):self.assertEqual(r.memory.goals,{})

    def test_valid_stage_repair_retains_prior_stages(self):
        r=self.runtime();grounds=0
        def respond(m,p):
            nonlocal grounds
            c=context(p)
            if c['work']=='ground':
                grounds+=1;v=grounding(c)
                if grounds==1:v['observation_id']='old'
                else:self.assertIn('stale',c['correction'])
                return call('submit_grounding',v)
            return answer(m,p)
        with patch.object(LocalVisionLlm,'_complete',respond):result=r.decide(obs())
        self.assertEqual(result['status'],'action');self.assertEqual(r.memory.model_calls,6)

    def test_invalid_fast_output_reconciles_and_reuses_grounded_skill(self):
        r=self.runtime();selections=0;reused=[]
        def respond(m,p):
            nonlocal selections
            c=context(p)
            if c['work']=='choose_skill':
                selections+=1
                if selections==1:return token('not a digit')
            if c['work']=='read_memory' and c['for_work']=='ground':
                if c['selected']:return token('8')
                if c['opened_record']:return token('7')
                for label,option in c['choices'].items():
                    if option.get('record_kind')=='procedure' or option.get('key')=='procedures':return token(label)
            if c['work']=='ground' and c['retained_skills']:
                reused.append(c['retained_skills']['move'])
                v=grounding(c);v['plan']['skills']=[];v['plan']['reuse']=['move'];return call('submit_grounding',v)
            return answer(m,p)
        with patch.object(LocalVisionLlm,'_complete',respond):result=r.decide(obs())
        self.assertEqual(result['status'],'action');self.assertEqual(selections,2)
        self.assertEqual(len(reused),1)
        self.assertEqual(r.memory.reconciliations[-1]['trigger']['trigger'],'invalid_fast_output')

    def test_grounding_can_return_to_understanding_without_sending_action(self):
        r=self.runtime();grounds=0;understandings=0
        def respond(m,p):
            nonlocal grounds,understandings
            c=context(p)
            if c['work']=='understand':understandings+=1
            if c['work']=='ground':
                grounds+=1
                if grounds==1:
                    return call('submit_grounding',{'observation_id':c['observation_id'],
                        'next':'understand','reason':'Locate the moving actor again.','next_question':'Where did the actor move?','plan':None})
            return answer(m,p)
        with patch.object(LocalVisionLlm,'_complete',respond):result=r.decide(obs())
        self.assertEqual(result['status'],'action');self.assertEqual(understandings,2)

    def test_boundary_reinterprets_and_retains_only_knowledge_and_skills(self):
        for boundary in ('level','reset'):
            with self.subTest(boundary=boundary):
                r=self.runtime();inputs=[]
                def respond(m,p):inputs.append(context(p));return answer(m,p)
                with patch.object(LocalVisionLlm,'_complete',respond):
                    r.decide(obs());ack(r)
                    if boundary=='reset':
                        r.decide({**obs(1),'state':'GAME_OVER'});ack(r);r.decide(obs(2))
                    else:r.decide({**obs(1),'levels_completed':1})
                u=[c for c in inputs if c['work']=='understand'][-1]
                self.assertNotIn('previous_understanding',u);self.assertIsNone(u['current_goal'])
                self.assertNotIn('recent_trials',u);self.assertEqual(u['last_review'],[])
                self.assertTrue(any(n['kind']=='hypothesis' for n in r.memory.notes.values()))
                self.assertIn('move',r.memory.skills)

    def test_changed_legal_controls_require_reconciliation_and_regrounding(self):
        r=self.runtime()
        def respond(m,p):
            c=context(p)
            if c['work']=='ground' and c['available_actions']==['DOWN']:
                v=grounding(c);v['plan']['skills'][0]['steps'][0]['options'][0]['action']={'action':'DOWN'}
                return call('submit_grounding',v)
            return answer(m,p)
        with patch.object(LocalVisionLlm,'_complete',respond):
            r.decide(obs());ack(r);result=r.decide({**obs(1),'available_actions':['ACTION2']})
        self.assertEqual(result['action'],'ACTION2')
        self.assertEqual(r.memory.reconciliations[-1]['trigger']['trigger'],'controls_changed')

    def test_deadline_during_slow_or_fast_never_sends_fallback_action(self):
        for work in ('understand','backchain','ground','execute_step'):
            with self.subTest(work=work):
                r=self.runtime()
                def respond(m,p):
                    if context(p)['work']==work:r.turn_deadline=0
                    return answer(m,p)
                with patch.object(LocalVisionLlm,'_complete',respond):result=r.decide(obs())
                self.assertEqual(result['reason'],'decision_time_exhausted')
                self.assertIsNone(r.memory.pending)

    def test_receipts_idempotence_terminal_and_missing_pixels(self):
        r=self.runtime()
        with patch.object(LocalVisionLlm,'_complete',answer):
            first=r.decide(obs());self.assertEqual(r.decide(obs()),first)
            with self.assertRaisesRegex(ValueError,'conflicting'):r.decide(obs(grid=[[1]*6]))
            self.assertEqual(r.decide(obs(1))['reason'],'execution_outcome_unknown')
        self.assertEqual(r.memory.model_calls,5)
        other=self.runtime()
        self.assertEqual(other.decide({**obs(),'state':'WIN'})['reason'],'win')
        o=obs();o.pop('grid');third=self.runtime()
        self.assertEqual(third.decide(o)['reason'],'observation_unavailable')

    def test_image_prefix_journals_and_directional_schema(self):
        from agent.observation import attach_visuals
        with tempfile.TemporaryDirectory() as d:
            r=self.runtime(log_dir=d);requests=[]
            def respond(m,p):
                requests.append(p);c=context(p)
                if c['work']=='ground':
                    a=p['tools'][0]['function']['parameters']['$defs']['ActionIntent']
                    self.assertNotIn('x',a['properties'])
                    self.assertEqual(a['properties']['action']['enum'],['UP'])
                    v=grounding(c);v['plan']['skills'][0]['steps'][0]['options'][0]['action']={'action':'UP'}
                    return call('submit_grounding',v)
                return answer(m,p)
            a=obs();b=obs(1,[[0,0,0,0,0,1]])
            for o in (a,b):o['available_actions']=['ACTION1'];attach_visuals(o,[o['grid']],None)
            with patch.object(LocalVisionLlm,'_complete',respond):r.decide(a);ack(r);r.decide(b)
            parts=requests[-1]['messages'][-1]['content']
            self.assertEqual(sum(p['type']=='image_url' for p in parts),2)
            self.assertEqual(context(requests[-1])['last_result']['changed_cell_count'],1)
            self.assertNotIn('available_actions',context(requests[-1]))
            calls=[json.loads(s) for s in (Path(d)/(r.session_id+'.model.jsonl')).read_text().splitlines()]
            self.assertEqual([c['work'] for c in calls[:3]],['understand','backchain','ground'])
            self.assertTrue(all(c['schema_valid'] for c in calls))
            memory=json.loads((Path(d)/(r.session_id+'.json')).read_text())
            self.assertIn('approach',memory['goals']);self.assertEqual(memory['schema_version'],12)
