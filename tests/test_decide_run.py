"""Real ADK dispatch through a scripted model, including autonomous skill lifecycle."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from agent.cognition.workflow import CognitiveRuntime
from agent.local_vlm import LocalVisionLlm
from agent.cognition.state import Decision
from test_skill_learning import obs, spec, candidate, promote


def call(name, arguments):
    return {'choices':[{'message':{'tool_calls':[{'id':'test-call','type':'function',
        'function':{'name':name,'arguments':json.dumps(arguments)}}]},'finish_reason':'tool_calls'}]}


def context(payload):
    return next(json.loads(p['text']) for m in payload['messages'] if m['role']=='user' and isinstance(m['content'],list)
                for p in m['content'] if p.get('type')=='text')


def act(x=0, patch_data=None):
    return {'kind':'act','action':{'action':'CLICK','x':x,'y':0,'reason':'Test the visible cell'},
            'prediction':'This cell may change.', 'patch':patch_data or {}}


def ack(runtime):
    runtime.record_execution('action_dispatched');runtime.record_execution('action_acknowledged')


class RuntimeTests(unittest.TestCase):
    def runtime(self, **kw):
        r=CognitiveRuntime('test','local/qwen3-vl-4b-instruct',**kw)
        self.addCleanup(r.close)
        return r

    def test_one_controller_one_judgment_and_persistent_task(self):
        r=self.runtime();requests=[]
        def answer(model,p):
            requests.append(p);return call('submit_decision',act(0,{'task':'Test zero cells','summary':'Observed zero row'}))
        with patch.object(LocalVisionLlm,'_complete',answer):
            result=r.decide(obs());self.assertEqual(result['action'],'ACTION6');ack(r)
            r.decide(obs(1,[[1,0,0,0,0,0]]))
        self.assertEqual(len(requests),2)
        self.assertEqual(context(requests[1])['task'],'Test zero cells')
        self.assertTrue(context(requests[1])['last_result']['acknowledged'])
        self.assertEqual(r.memory.history[0]['trace'],['DECIDE','RUN'])
        self.assertEqual(r.controller.name,'decision_controller')

    def test_invalid_action_does_not_publish_patch(self):
        r=self.runtime();requests=[]
        def answer(model,p):
            requests.append(p)
            data=act(63,{'task':'invalid overwrite'}) if len(requests)==1 else act(0)
            return call('submit_decision',data)
        with patch.object(LocalVisionLlm,'_complete',answer):
            result=r.decide(obs())
        self.assertEqual(result['x'],0)
        self.assertNotEqual(r.memory.task,'invalid overwrite')
        self.assertIn('error',json.loads(next(m['content'] for m in requests[1]['messages'] if m['role']=='tool')))

    def test_no_implicit_cursor_and_no_unavailable_buttons(self):
        r=self.runtime();r.obs=obs()
        for action in ({'action':'CLICK'},{'action':'UP'},{'action':'RESET'}):
            with self.assertRaises(ValueError):
                r.validate_job(Decision(kind='act',action=action,prediction='test'))

    def test_unknown_execution_cannot_become_learning_evidence(self):
        r=CognitiveRuntime('test');self.addCleanup(r.close)
        first=r.decide(obs())
        self.assertEqual(r.decide(obs()),first)
        stopped=r.decide(obs(1,[[1]*6]))
        self.assertEqual(stopped['reason'],'execution_outcome_unknown')
        self.assertFalse(list(r.library.experiences.values())[0]['acknowledged'])

    def test_terminal_and_reset_boundaries(self):
        for state,reason in [('WIN','win'),('GAME_OVER','reset_budget')]:
            r=CognitiveRuntime('test',max_resets=0);self.addCleanup(r.close)
            data=obs();data['state']=state
            self.assertEqual(r.decide(data)['reason'],reason)
        r=CognitiveRuntime('test');self.addCleanup(r.close)
        data=obs();data['state']='NOT_PLAYED';data['available_actions']=[]
        self.assertEqual(r.decide(data)['action'],'RESET')

    def test_model_call_budget_and_text_without_submission(self):
        r=self.runtime(max_calls=2,max_http_requests=2)
        with patch.object(LocalVisionLlm,'_complete',return_value={'choices':[{'message':{'content':'not a job'}}]}) as http:
            self.assertEqual(r.decide(obs())['status'],'stop')
        self.assertLessEqual(http.call_count,2)

    def test_native_tools_reach_adk_and_audit_contains_unfinished_events(self):
        with tempfile.TemporaryDirectory() as d:
            r=self.runtime(log_dir=d);requests=[]
            def answer(model,p):
                requests.append(p)
                name,args=('list_observations',{}) if len(requests)==1 else ('submit_decision',act())
                return {'choices':[{'message':{'content':'<tool_call>'+json.dumps({'name':name,'arguments':args})+'</tool_call>'}}]}
            with patch.object(LocalVisionLlm,'_complete',answer):r.decide(obs())
            tools=[json.loads(l) for l in next(Path(d).glob('*.tools.jsonl')).read_text().splitlines()]
            self.assertEqual(tools[0]['event'],'tool_started')
            self.assertEqual(tools[1]['event'],'tool_finished')
            self.assertEqual(len(requests),2)
            self.assertTrue(next(Path(d).glob('*.requests.jsonl')).is_file())

    def test_learning_off_still_invokes_but_cannot_draft_or_trial(self):
        r=self.runtime(learning=False);r.obs=obs();key=candidate(r.library)
        with self.assertRaisesRegex(ValueError,'disabled'):
            r.validate_job(Decision(kind='trial',skill_id=key,arguments={'x':1},prediction='test'))
        tool_names=[t.name for t in r._agent().tools if hasattr(t,'name')]
        self.assertNotIn('propose_skill',tool_names)
        promote(r.library,key)
        r.validate_job(Decision(kind='invoke',skill_id=key,arguments={'x':3},prediction='reuse'))

    def test_complete_autonomous_creation_trial_promotion_and_reuse(self):
        with tempfile.TemporaryDirectory() as d:
            r=self.runtime(log_dir=d);requests=[]
            def answer(model,p):
                requests.append(p);c=context(p);step=c['observation']['step'];skills=c['skills']
                if step==0:return call('submit_decision',act(0))
                if not skills:
                    return call('propose_skill',{'spec':spec(),'evidence_ids':['experience-1'],'examples':[{'x':0}]})
                skill=skills[0];key=skill['id']
                if skill['fresh_trials']<2:
                    return call('submit_decision',{'kind':'trial','skill_id':key,'arguments':{'x':step},'prediction':'Cell changes to 1.'})
                if skill['status']=='candidate':
                    return call('submit_decision',{'kind':'evaluate','skill_id':key,'prediction':'Check real trial evidence.'})
                if step==3:
                    return call('submit_decision',{'kind':'invoke','skill_id':key,'arguments':{'x':3},'prediction':'Reuse the verified local effect.'})
                return call('submit_decision',{'kind':'stop','prediction':'Test completed.'})
            grid=[[0]*6]
            with patch.object(LocalVisionLlm,'_complete',answer):
                for step in range(5):
                    result=r.decide(obs(step,grid))
                    if step==4:
                        self.assertEqual(result['status'],'stop');break
                    self.assertEqual(result.get('action'),'ACTION6',result)
                    ack(r);grid[0][result['x']]=1
            records=list(r.library.records.values())
            self.assertEqual(len(records),1)
            self.assertEqual(records[0]['status'],'active')
            self.assertEqual(len(records[0]['trials']),2)
            self.assertTrue(all(records[0]['evaluation']['checks'].values()))
            events=[json.loads(l)['event'] for l in next(Path(d).glob('*.learning.jsonl')).read_text().splitlines()]
            for event in ('skill_drafted','skill_trial_finished','skill_evaluated','skill_promoted','skill_execution_finished'):
                self.assertIn(event,events)
            self.assertEqual(r.memory.model_calls,7)

    def test_active_effect_violation_suspends_and_returns_to_decision(self):
        r=self.runtime();key=candidate(r.library);promote(r.library,key)
        def answer(model,p):
            return call('submit_decision',{'kind':'invoke','skill_id':key,'arguments':{'x':3},'prediction':'Cell should change.'}) if context(p)['observation']['step']==0 else call('submit_decision',act(4))
        with patch.object(LocalVisionLlm,'_complete',answer):
            r.decide(obs());ack(r)
            result=r.decide(obs(1))
        self.assertEqual(r.library.get(key)['status'],'suspended')
        self.assertEqual(result['x'],4)

    def test_conflicting_and_wrong_game_observations_rejected(self):
        r=CognitiveRuntime('test');self.addCleanup(r.close);r.decide(obs())
        with self.assertRaisesRegex(ValueError,'conflicting'):r.decide(obs(0,[[1]*6]))
        data=obs();data['game_id']='other'
        with self.assertRaisesRegex(ValueError,'wrong game'):r.decide(data)

    def test_direction_only_game_does_not_offer_click_arguments_or_candidate_trials(self):
        r=self.runtime();seen=[]
        def answer(model,payload):
            seen.append(payload)
            tool=next(t['function'] for t in payload['tools'] if t['function']['name']=='submit_decision')
            schema=tool['parameters']
            self.assertNotIn('trial',schema['properties']['kind']['enum'])
            self.assertNotIn('x',schema['$defs']['Action']['properties'])
            self.assertIn('action',schema['required'])
            self.assertIn('prediction',schema['required'])
            self.assertNotIn('oneOf',schema)
            self.assertNotIn('skill_id',schema['properties'])
            self.assertNotIn('propose_skill',[t['function']['name'] for t in payload['tools']])
            return call('submit_decision',{'kind':'act','action':{'action':'UP'},'prediction':'Test movement.'})
        data=obs();data['available_actions']=['ACTION1','ACTION2']
        with patch.object(LocalVisionLlm,'_complete',answer):
            self.assertEqual(r.decide(data)['action'],'ACTION1')
        self.assertEqual(len(seen),1)

    def test_missing_action_gets_specific_correction_and_can_recover(self):
        r=self.runtime();seen=[]
        def answer(model,payload):
            seen.append(payload)
            if len(seen)==1:
                return call('submit_decision',{'kind':'act','prediction':'Test movement.'})
            feedback=json.loads(next(m['content'] for m in payload['messages'] if m['role']=='tool'))
            self.assertIn('requires action=',feedback['correction'])
            return call('submit_decision',{'kind':'act','action':{'action':'UP'},'prediction':'Test movement.'})
        data=obs();data['available_actions']=['ACTION1','ACTION2']
        with patch.object(LocalVisionLlm,'_complete',answer):
            self.assertEqual(r.decide(data)['action'],'ACTION1')
        self.assertEqual(len(seen),2)
