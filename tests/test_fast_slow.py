"""Real ADK planning with scripted HTTP: fast control, replanning and receipt boundaries."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from agent.cognition.workflow import CognitiveRuntime
from agent.local_vlm import LocalVisionLlm
from runtime_helpers import obs, context, call, token, skill, plan, answer, ack


class FastSlowTests(unittest.TestCase):
    def runtime(self, **kwargs):
        runtime=CognitiveRuntime('test','local/qwen3-vl-4b-instruct',**kwargs)
        self.addCleanup(runtime.close)
        return runtime

    def test_plan_once_then_single_token_execution_without_repetition_rejection(self):
        r=self.runtime(); requests=[]
        def respond(m,p):
            requests.append(p);return answer(m,p)
        with patch.object(LocalVisionLlm,'_complete',respond):
            for step in range(6):
                result=r.decide(obs(step))
                self.assertEqual(result['status'],'action');self.assertEqual(result['x'],0);ack(r)
        works=[context(p)['work'] for p in requests]
        self.assertEqual(works,['deliberate','choose_skill']+['execute_step']*6)
        for p in requests[1:]:
            self.assertEqual(p['max_tokens'],1);self.assertNotIn('tools',p)
            self.assertIn('"8"',p['grammar'])
            self.assertNotIn('causal_notes',context(p))
        self.assertEqual(r.calls,1)
        self.assertEqual(len(r.recent_trials),5)

    def test_reconsider_sends_no_action_and_slow_receives_expectation_and_evidence(self):
        r=self.runtime(); requests=[]; deliberations=0; executions=0
        def respond(m,p):
            nonlocal deliberations, executions
            c=context(p);requests.append(c)
            if c['work']=='deliberate':
                deliberations+=1
                if deliberations==2:
                    self.assertIn('chose 8',c['return_to_deliberation'])
                    self.assertEqual(c['last_result']['prediction'],'The target may change color.')
                    self.assertEqual(c['last_result']['changed_cell_count'],1)
                    self.assertIsNone(r.memory.pending)
                return call('submit_plan',plan(c,x=0 if deliberations==1 else 2))
            if c['work']=='execute_step':
                executions+=1
                return token('8' if executions==2 else '1')
            return token('1')
        with patch.object(LocalVisionLlm,'_complete',respond):
            self.assertEqual(r.decide(obs())['x'],0);ack(r)
            result=r.decide(obs(1,[[0,0,0,0,0,1]]))
        self.assertEqual(result['x'],2)
        self.assertEqual(deliberations,2)
        self.assertEqual(r.memory.pending['action']['x'],2)
        self.assertEqual(r.routing_history[-1]['source'],'execute_step')

    def test_complete_step_then_select_another_skill_without_slow_call(self):
        r=self.runtime(); fast=iter(['1','1','7','1','7','1','1'])
        def respond(m,p):
            c=context(p)
            if c['work']=='deliberate':
                value=plan(c);s=value['skills'][0]
                s['steps'].append(skill(x=1)['steps'][0])
                value['skills'].append(skill('other',2));value['candidates'].append('other')
                return call('submit_plan',value)
            return token(next(fast))
        with patch.object(LocalVisionLlm,'_complete',respond):
            self.assertEqual(r.decide(obs())['x'],0);ack(r)
            self.assertEqual(r.decide(obs(1))['x'],1);ack(r)
            self.assertEqual(r.decide(obs(2))['x'],2)
        self.assertEqual(r.completed,{'probe'})
        self.assertEqual(r.memory.active_skill['name'],'other')
        self.assertEqual(r.memory.model_calls,8)

    def test_completed_plan_replans_and_can_reuse_retained_skill(self):
        r=self.runtime(); executions=0; slow=0
        def respond(m,p):
            nonlocal executions,slow
            c=context(p)
            if c['work']=='deliberate':
                slow+=1;value=plan(c)
                if slow==2:
                    self.assertIn('probe',c['retained_skills']);value['skills']=[]
                    self.assertEqual(c['completed_candidates'],['probe'])
                    self.assertEqual(c['completion_judgments'][-1]['skill'],{'name':'probe','index':0})
                    self.assertEqual(c['completion_judgments'][-1]['source'],'fast_model_judgment')
                return call('submit_plan',value)
            if c['work']=='execute_step':
                executions+=1;return token('7' if executions==2 else '1')
            return token('1')
        with patch.object(LocalVisionLlm,'_complete',respond):
            r.decide(obs());ack(r);result=r.decide(obs(1))
        self.assertEqual(result['status'],'action');self.assertEqual(slow,2)

    def test_fast_invalid_output_returns_to_deliberation_without_dispatch(self):
        r=self.runtime(); count=0
        def respond(m,p):
            nonlocal count
            c=context(p)
            if c['work']=='choose_skill':
                count+=1
                if count==1:return token('not a token')
            return answer(m,p)
        with patch.object(LocalVisionLlm,'_complete',respond):
            result=r.decide(obs())
        self.assertEqual(result['status'],'action')
        self.assertEqual(count,2)
        self.assertIn('could not be read',r.routing_history[-1]['reason'])

    def test_invalid_plan_is_atomic_and_only_repaired_once(self):
        for bad in ['coordinate','stale','unknown_skill','illegal','duplicate']:
            with self.subTest(bad=bad):
                r=self.runtime()
                def respond(m,p):
                    value=plan(context(p))
                    if bad=='coordinate':value['skills'][0]['steps'][0]['options'][0]['action']['x']=63
                    elif bad=='stale':value['observation_id']='stale'
                    elif bad=='unknown_skill':value['candidates']=['missing']
                    elif bad=='illegal':value['skills'][0]['steps'][0]['options'][0]['action']={'action':'DOWN'}
                    else:value['candidates']=['probe','probe']
                    return call('submit_plan',value)
                with patch.object(LocalVisionLlm,'_complete',respond):result=r.decide(obs())
                self.assertEqual(result['reason'],'plan_output_invalid')
                self.assertEqual(r.memory.model_calls,2)
                self.assertEqual(r.memory.skills,{})
                self.assertIsNone(r.memory.pending)

    def test_valid_repair_reaches_fast_executor(self):
        r=self.runtime(); count=0
        def respond(m,p):
            nonlocal count
            c=context(p)
            if c['work']=='deliberate':
                count+=1;value=plan(c)
                if count==1:value['observation_id']='old'
                else:self.assertIn('stale',c['correction'])
                return call('submit_plan',value)
            return token('1')
        with patch.object(LocalVisionLlm,'_complete',respond):result=r.decide(obs())
        self.assertEqual(result['status'],'action');self.assertEqual(count,2)

    def test_level_boundary_replans_keeps_knowledge_and_clears_bound_execution(self):
        r=self.runtime(); inputs=[]
        def respond(m,p):inputs.append(context(p));return answer(m,p)
        with patch.object(LocalVisionLlm,'_complete',respond):
            r.decide(obs());ack(r)
            r.decide({**obs(1),'levels_completed':1})
        slow=[c for c in inputs if c['work']=='deliberate']
        self.assertEqual(len(slow),2);self.assertIsNone(slow[-1]['previous_plan'])
        self.assertEqual(slow[-1]['recent_trials'],[]);self.assertIn('probe',slow[-1]['retained_skills'])
        self.assertEqual(slow[-1]['causal_notes_hypotheses'],'No causal relation confirmed yet.')

    def test_legal_controls_changed_returns_for_regrounding(self):
        r=self.runtime(); slow=0
        def respond(m,p):
            nonlocal slow
            c=context(p)
            if c['work']=='deliberate':
                slow+=1;value=plan(c)
                if slow==2:value['skills'][0]['steps'][0]['options'][0]['action']={'action':'UP'}
                return call('submit_plan',value)
            return token('1')
        with patch.object(LocalVisionLlm,'_complete',respond):
            r.decide(obs());ack(r)
            result=r.decide({**obs(1),'available_actions':['ACTION1']})
        self.assertEqual(result['action'],'ACTION1');self.assertEqual(slow,2)

    def test_time_budget_bounds_replanning_without_sending_fallback_action(self):
        r=self.runtime(); slow=0
        def respond(m,p):
            nonlocal slow
            c=context(p)
            if c['work']=='deliberate':
                slow+=1
                if slow==2:r.turn_deadline=0
                return call('submit_plan',plan(c))
            return token('8')
        with patch.object(LocalVisionLlm,'_complete',respond):result=r.decide(obs())
        self.assertEqual(result['reason'],'decision_time_exhausted')
        self.assertEqual(slow,2);self.assertIsNone(r.memory.pending)

    def test_deadline_after_fast_reply_prevents_dispatch(self):
        r=self.runtime()
        def respond(m,p):
            if context(p)['work']=='execute_step':r.deadline=0
            return answer(m,p)
        with patch.object(LocalVisionLlm,'_complete',respond):result=r.decide(obs())
        self.assertEqual(result['reason'],'budget_exhausted');self.assertIsNone(r.memory.pending)

    def test_unknown_receipt_duplicate_and_conflicting_observation(self):
        r=self.runtime()
        with patch.object(LocalVisionLlm,'_complete',answer):
            result=r.decide(obs());self.assertEqual(r.decide(obs()),result)
            with self.assertRaisesRegex(ValueError,'conflicting'):r.decide(obs(grid=[[1]*6]))
            self.assertEqual(r.decide(obs(1))['reason'],'execution_outcome_unknown')
        self.assertEqual(r.memory.model_calls,3)

    def test_terminal_and_missing_visuals_skip_inference(self):
        r=self.runtime()
        with patch.object(LocalVisionLlm,'_complete',answer):
            r.decide(obs());ack(r)
            self.assertEqual(r.decide({**obs(1),'state':'WIN'})['reason'],'win')
        self.assertEqual(r.memory.model_calls,3)
        other=self.runtime();o=obs();o.pop('grid')
        self.assertEqual(other.decide(o)['reason'],'observation_unavailable')
        self.assertEqual(other.memory.model_calls,0)

    def test_images_precede_question_for_shared_prefix_and_actual_result_is_preserved(self):
        from agent.observation import attach_visuals
        r=self.runtime(); requests=[]
        def respond(m,p):requests.append(p);return answer(m,p)
        a=obs();b=obs(1,[[0,0,0,0,0,1]])
        for o in [a,b]:attach_visuals(o,[o['grid']],None)
        with patch.object(LocalVisionLlm,'_complete',respond):
            r.decide(a);ack(r);r.decide(b)
        p=requests[-1];parts=p['messages'][-1]['content']
        self.assertEqual(sum(x['type']=='image_url' for x in parts),2)
        self.assertEqual(context(p)['last_result']['changed_cell_count'],1)
        self.assertEqual(json.loads(parts[-1]['text'])['work'],'execute_step')
        self.assertNotIn('available_actions',context(p))

    def test_journals_record_both_speeds_and_cognitive_snapshots(self):
        with tempfile.TemporaryDirectory() as d:
            r=self.runtime(log_dir=d)
            with patch.object(LocalVisionLlm,'_complete',answer):r.decide(obs())
            calls=[json.loads(s) for s in (Path(d)/(r.session_id+'.model.jsonl')).read_text().splitlines()]
            self.assertEqual([c['work'] for c in calls],['deliberate','choose_skill','execute_step'])
            self.assertTrue(all(c['schema_valid'] for c in calls))
            events=[json.loads(s) for s in (Path(d)/(r.session_id+'.artifacts.jsonl')).read_text().splitlines()]
            self.assertTrue(any(e['event']=='cognition_updated' and e['cognition']['active_skill'] for e in events))

    def test_directional_schema_has_no_click_coordinates_and_fast_choice_executes(self):
        r=self.runtime()
        def respond(m,p):
            c=context(p)
            if c['work']=='deliberate':
                action=p['tools'][0]['function']['parameters']['$defs']['Action']
                self.assertNotIn('x',action['properties'])
                self.assertEqual(action['properties']['action']['enum'],['UP','DOWN'])
                value=plan(c);value['skills'][0]['steps'][0]['options'][0]['action']={'action':'UP'}
                return call('submit_plan',value)
            return token('1')
        with patch.object(LocalVisionLlm,'_complete',respond):
            result=r.decide({**obs(),'available_actions':['ACTION1','ACTION2']})
        self.assertEqual(result['action'],'ACTION1')

    def test_reset_requires_fresh_plan_before_reusing_grounded_procedure(self):
        r=self.runtime();inputs=[]
        def respond(m,p):inputs.append(context(p));return answer(m,p)
        with patch.object(LocalVisionLlm,'_complete',respond):
            r.decide(obs());ack(r)
            self.assertEqual(r.decide({**obs(1),'state':'GAME_OVER'})['action'],'RESET');ack(r)
            self.assertEqual(r.decide(obs(2))['status'],'action')
        slow=[c for c in inputs if c['work']=='deliberate']
        self.assertEqual(len(slow),2)
        self.assertIn('reset boundary',slow[-1]['return_to_deliberation'])
        self.assertIsNone(slow[-1]['previous_plan'])
