"""Production focused workflow, using real ADK dispatch and scripted model answers."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from agent.cognition.workflow import CognitiveRuntime
from agent.cognition.tasks import GoalSelection
from agent.local_vlm import LocalVisionLlm
from test_decide_run import call, context, ack
from test_skill_learning import obs, candidate, promote


def answer_for(c, x=0):
    work = c['work']
    result = {'task_id':c['task_id']}
    if work == 'interpret_world':
        name='submit_world'
        objects=[{'key':str(i),'component_ids':[v['id']], 'description':f"Region {i}",
                  'possible_roles':['unknown']} for i,v in enumerate(c['components'])]
        result.update(objects=objects,relations=[],questions=[{'subject':o['key'],'action':'CLICK','observe':[o['key']],
            'property':prop,'question':f"Does {o['key']} change {prop}?",
            'alternatives':['observed change','no observed change']}
            for o in objects[:2] for prop in ['appearance','position']])
    elif work == 'select_goal':
        name='submit_goal'; result.update(question_id=next(q['id'] for q in c['world']['questions'] if q['status']=='open' and q['id']!=(c.get('previous_goal') or {}).get('question_id')),text='Find a reactive cell',goal_type='knowledge',
            done_when='A candidate cell has been tested',reason='Its response is unknown')
    elif work == 'assess_goal':
        name='submit_goal_assessment'; result.update(decision='continue',reason='Another cell remains untested',
            remaining_question='Does a different cell react?',evidence_ids=c['fact']['evidence_ids'] if c['fact'] else [c['observation_id']])
        if not c['world']['questions'][0]['current']:
            result.update(decision='replace',reason='Object grounding changed',remaining_question='')
    elif work == 'design_experiment':
        oid=next((o['id'] for o in c['grounded_objects'] if any(v['region']['x'] <= x < v['region']['x']+v['region']['width'] for v in o['components'])),c['grounded_objects'][0]['id'])
        name='submit_experiment'; result.update(target_object_id=oid,action={'action':'CLICK','x':x,'y':0},target=f'Candidate cell {x}',
            question='Does this cell react?',hypothesis='This cell changes',conditions='Current state',
            expected={'kind':'region_changed','description':'Cell changes','region':{'x':x,'y':0,'width':1,'height':1}})
    elif work == 'inspect_target':
        name='submit_target'; result.update(region={'x':int(c['target'].split()[-1]),'y':0,'width':1,'height':1},finding='The click is inside the target')
    elif work == 'judge_effect':
        name='submit_effect'; result.update(verdict='unsupported',finding='The frozen effect did not occur',evidence_ids=[c['evidence_ids'][-1]])
    elif work == 'choose_method':
        name='submit_method'; result.update(method='explore',reason='Continue exploration')
    else:
        raise AssertionError(work)
    return name,result


class FocusedTests(unittest.TestCase):
    def runtime(self, **kwargs):
        kwargs.setdefault('max_calls',8); kwargs.setdefault('max_http_requests',16)
        r=CognitiveRuntime('test','local/qwen3-vl-4b-instruct',**kwargs)
        self.addCleanup(r.close)
        return r

    def test_three_separate_tasks_and_scoped_inputs(self):
        r=self.runtime(); requests=[]
        def answer(model,p):
            requests.append(p);return call(*answer_for(context(p)))
        with patch.object(LocalVisionLlm,'_complete',answer):
            result=r.decide(obs())
        self.assertEqual(result['x'],0)
        self.assertEqual([context(p)['work'] for p in requests],['interpret_world','select_goal','design_experiment','inspect_target'])
        for p in requests:
            c=context(p)
            self.assertNotIn('notebook',c)
            self.assertNotIn('skills',c)
            self.assertLessEqual(len(p['tools']),3)
            self.assertNotIn('write_note',str(p['tools']))
        target=context(requests[-1]);self.assertNotIn('goal',target)
        self.assertEqual(len({context(p)['task_id'] for p in requests}),4)

    def test_same_failed_test_routes_to_goal_then_actual_new_action(self):
        r=self.runtime(max_calls=8,max_http_requests=12); requests=[]; repeated=False
        def answer(model,p):
            nonlocal repeated
            c=context(p);requests.append(c)
            x=1 if repeated else 0
            if c['work']=='assess_goal' and c.get('rejection'):
                repeated=True;x=1
            return call(*answer_for(c,x))
        with patch.object(LocalVisionLlm,'_complete',answer):
            first=r.decide(obs());ack(r);second=r.decide(obs(1))
        self.assertEqual(first['x'],0);self.assertEqual(second['x'],1)
        self.assertEqual(len(r.library.experiences),1)
        assessments=[c for c in requests if c['work']=='assess_goal']
        self.assertEqual(assessments[-1]['rejection']['reason'],'same_test_unchanged_conditions')
        self.assertNotIn('grid',assessments[-1])
        self.assertEqual(r.experiments.active['data']['plan']['revision']['change'],'action')

    def test_missed_target_returns_region_to_designer_without_executing_it(self):
        r=self.runtime(max_calls=6,max_http_requests=8);requests=[]
        def answer(model,p):
            c=context(p);requests.append(c)
            x=1 if c.get('target_assessment') else 0
            name,data=answer_for(c,x)
            if c['work']=='inspect_target' and c['target']=='Candidate cell 0':
                data.update(region={'x':1,'y':0,'width':1,'height':1},finding='Target is one cell to the right')
            return call(name,data)
        with patch.object(LocalVisionLlm,'_complete',answer):
            result=r.decide(obs())
        self.assertEqual(result['x'],1)
        self.assertEqual(r.experiments.sequence,1)
        self.assertEqual([c['work'] for c in requests].count('design_experiment'),2)

    def test_stale_answer_cannot_mutate_goal(self):
        r=self.runtime();r.obs=obs();r.work='select_goal';r._open_task()
        job=GoalSelection(task_id=r.task_id,text='Test',goal_type='knowledge',done_when='Observed',reason='Unknown')
        r.obs=obs(1)
        with self.assertRaisesRegex(ValueError,'stale'):
            r.validate_job(job)
        self.assertIsNone(r._goal())

    def test_semantic_verdict_does_not_close_goal(self):
        r=self.runtime(max_calls=6,max_http_requests=10)
        def answer(model,p):
            c=context(p);name,data=answer_for(c,1 if ':1:' in c['observation_id'] else 0)
            if c['work']=='design_experiment':data['expected']['kind']='semantic'
            return call(name,data)
        with patch.object(LocalVisionLlm,'_complete',answer):
            r.decide(obs());ack(r);result=r.decide(obs(1))
        self.assertEqual(result['x'],1)
        page=r.notebook.pages['experiment-1']
        self.assertEqual(page['data']['reviewer'],'agent')
        self.assertEqual(r._goal()['data']['status'],'active')

    def test_task_inputs_are_journaled_and_recovery_is_bounded(self):
        with tempfile.TemporaryDirectory() as directory:
            r=self.runtime(log_dir=directory,max_calls=12,max_http_requests=24)
            with patch.object(LocalVisionLlm,'_complete',lambda m,p:call(*answer_for(context(p)))):
                r.decide(obs());ack(r);result=r.decide(obs(1))
            self.assertEqual(result['reason'],'recovery_budget_exhausted')
            self.assertEqual(r.experiments.sequence,1)
            rows=[json.loads(line) for line in (Path(directory)/(r.session_id+'.notebook.jsonl')).read_text().splitlines()]
            self.assertTrue(any(e['event']=='task_opened' for e in rows))

    def test_skill_choice_and_arguments_are_separate(self):
        r=self.runtime(learning=False);key=candidate(r.library);promote(r.library,key)
        def answer(model,p):
            c=context(p)
            if c['work']=='choose_method':
                return call('submit_method',{'task_id':c['task_id'],'method':'invoke','skill_id':key,'reason':'Applicable learned skill'})
            if c['work']=='resolve_arguments':
                return call('submit_arguments',{'task_id':c['task_id'],'arguments':{'x':3}})
            return call(*answer_for(c))
        with patch.object(LocalVisionLlm,'_complete',answer):
            result=r.decide(obs())
        self.assertEqual(result['x'],3)
        self.assertFalse(r.memory.active_skill['trial'])

    def test_goal_replacement_can_reach_a_different_real_action(self):
        r=self.runtime(max_calls=6,max_http_requests=10)
        def answer(model,p):
            c=context(p);name,data=answer_for(c,1 if ':1:' in c['observation_id'] else 0)
            if c['work']=='assess_goal':
                data.update(decision='replace',reason='Test another visible candidate',remaining_question='')
            if c['work']=='select_goal' and c.get('previous_goal'):
                data['text']='Identify the second candidate response'
            return call(name,data)
        with patch.object(LocalVisionLlm,'_complete',answer):
            r.decide(obs());old=r._goal()['id'];ack(r);result=r.decide(obs(1))
        self.assertEqual(result['x'],1)
        self.assertEqual(r.notebook.pages[old]['data']['status'],'abandoned')
        self.assertNotEqual(old,r._goal()['id'])
        self.assertEqual(r.experiments.active['data']['plan']['subgoal']['id'],r._goal()['id'])

    def test_unrelated_pixel_change_does_not_offer_skill_learning(self):
        r=self.runtime();requests=[]
        def answer(model,p):
            c=context(p);requests.append(c)
            return call(*answer_for(c,1 if ':1:' in c['observation_id'] else 0))
        with patch.object(LocalVisionLlm,'_complete',answer):
            r.decide(obs());ack(r);result=r.decide(obs(1,[[0,0,0,0,0,1]]))
        self.assertEqual(result['x'],1)
        self.assertNotIn('choose_method',[c['work'] for c in requests])
        fact=r.notebook.pages['experiment-1']['data']['review']
        self.assertNotIn('understanding',fact)
        self.assertNotIn('subgoal_status',fact)

    def test_receipt_unknown_stops_production_without_another_model_call(self):
        r=self.runtime()
        with patch.object(LocalVisionLlm,'_complete',side_effect=lambda p:call(*answer_for(context(p)))) as http:
            r.decide(obs());calls=http.call_count;result=r.decide(obs(1))
        self.assertEqual(result['reason'],'execution_outcome_unknown')
        self.assertEqual(http.call_count,calls)
        self.assertFalse(r.library.experiences['experience-1']['acknowledged'])

    def test_evidence_access_cannot_escape_task_snapshot(self):
        r=self.runtime();r.obs=obs();r.work='select_goal';r._open_task()
        r.evidence.add(obs());r.evidence.add(obs(1))
        self.assertIn('error',r.read_task_evidence('o1'))
        self.assertNotIn('error',r.read_task_evidence('o0'))

    def test_finite_predeclared_retry_is_not_a_coordinate_ban(self):
        r=self.runtime()
        def answer(model,p):
            c=context(p);name,data=answer_for(c)
            if c['work']=='design_experiment':
                data.update(max_attempts=2,repeat_reason='Check a possible delayed response')
                if ':1:' in c['observation_id']:
                    data['retry_of']='experiment-1'
            return call(name,data)
        with patch.object(LocalVisionLlm,'_complete',answer):
            r.decide(obs());ack(r);result=r.decide(obs(1))
        self.assertEqual(result['x'],0)
        self.assertEqual(r.experiments.active['data']['attempt'],2)

    def test_skill_creation_trials_and_host_promotion(self):
        from test_skill_learning import spec
        r=self.runtime(max_calls=8,max_http_requests=16)
        def answer(model,p):
            c=context(p)
            if c['work']=='choose_method':
                skills=c['skills'];method='learn' if not skills else 'invoke' if skills[0]['status']=='active' else 'trial'
                data={'task_id':c['task_id'],'method':method,'reason':'Verified zero to one effect'}
                if skills:data['skill_id']=skills[0]['id']
                else:data['evidence_ids']=list(c['experiences'])
                return call('submit_method',data)
            if c['work']=='skill_creation':
                return call('propose_skill',{'task_id':c['task_id'],'spec':spec(),'evidence_ids':c['request']['evidence_ids'],'examples':[{'x':0}]})
            if c['work']=='resolve_arguments':
                x=next(i for i,v in enumerate(r.obs['grid'][0]) if v==0)
                return call('submit_arguments',{'task_id':c['task_id'],'arguments':{'x':x}})
            x=next((i for i,v in enumerate(r.obs['grid'][0]) if v==0),0)
            name,data=answer_for(c,x)
            return call(name,data)
        with patch.object(LocalVisionLlm,'_complete',answer):
            self.assertEqual(r.decide(obs())['x'],0);ack(r)
            # Build is followed by a design call; choose a distinct probe before trials.
            result=r.decide(obs(1,[[1,0,0,0,0,0]]))
            self.assertEqual(len(r.library.records),1)
            self.assertEqual(next(iter(r.library.records.values()))['status'],'candidate')
            ack(r)
            # The repeated probe is deliberately unchanged; its result is no support.
            r.decide(obs(2,[[1,0,0,0,0,0]]));ack(r)
            r.decide(obs(3,[[1,1,0,0,0,0]]));ack(r)
            result=r.decide(obs(4,[[1,1,1,0,0,0]]))
        record=next(iter(r.library.records.values()))
        self.assertEqual(record['status'],'active')
        self.assertEqual(result['x'],3)
        self.assertFalse(r.memory.active_skill['trial'])

    def test_active_experiment_endpoints_survive_archive_eviction(self):
        from agent.cognition.evidence import EvidenceStore
        e=EvidenceStore(capacity=1);self.addCleanup(e.close)
        e.add(obs());e.pin(['o0']);e.add(obs(1))
        self.assertEqual(e.get('o0')['step'],0)
        self.assertEqual(e.get('o1')['step'],1)
        e.pin([])
        self.assertNotIn('o0',e.index)
        self.assertIn('o1',e.index)

    def test_focused_replay_shows_only_task_input_available_then(self):
        from scripts.agent_monitor import Timeline, Run, dashboard_html
        from scripts.state_diagram import state_diagram_html
        with tempfile.TemporaryDirectory() as directory:
            r=self.runtime(log_dir=directory)
            with patch.object(LocalVisionLlm,'_complete',lambda m,p:call(*answer_for(context(p)))):
                r.decide(obs())
            timeline=Timeline(Run(Path(directory),r.session_id)).load()
            snapshots=[s for s in timeline.snapshots if s['event']['event']=='task_opened']
            self.assertEqual([s['context']['work'] for s in snapshots],['interpret_world','select_goal','design_experiment','inspect_target'])
            self.assertNotIn('goal',snapshots[-1]['context'])
            self.assertIsNone(snapshots[0]['action'])
            self.assertIn('この判断に渡した入力',dashboard_html(snapshots[-1],Path(directory)))
            self.assertEqual(state_diagram_html(snapshots[-1]['machine']).count('data-active="true"'),1)

    def test_invalid_geometry_stops_after_two_corrections_without_action(self):
        r=self.runtime();inspections=[]
        def answer(model,p):
            c=context(p);name,data=answer_for(c)
            if c['work']=='inspect_target':
                inspections.append(c)
                data['region']={'x':5,'y':0,'width':6,'height':1}
            return call(name,data)
        with patch.object(LocalVisionLlm,'_complete',answer):
            result=r.decide(obs())
        self.assertEqual(result['reason'],'invalid_task_output')
        self.assertEqual(len(inspections),3)
        self.assertIsNone(r.memory.pending)
