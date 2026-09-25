"""Memory integrity, bounded retrieval and historical replay of shared notes."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from agent.cognition.notebook import Notebook
from agent.cognition.workflow import CognitiveRuntime
from agent.local_vlm import LocalVisionLlm
from scripts.agent_monitor import Run, Timeline
from test_decide_run import call, act, context, ack
from test_skill_learning import obs


class NotebookTests(unittest.TestCase):
    def notebook(self, **kw):
        n = Notebook('game', 'run', **kw)
        n.record_observation({'observation_id':'o1', 'step':1, 'state':'NOT_FINISHED'},
                             {'experience_id':'e1','acknowledged':True,'action':{'action':'UP'}})
        return n

    def test_edits_preserve_versions_and_reject_stale_write(self):
        with tempfile.TemporaryDirectory() as d:
            n=self.notebook(directory=d)
            page=n.write('hypothesis','UP effect','UP may move an object',['e1'])
            n.bookmark('experiment',page['id'])
            updated=n.write('hypothesis','UP effect','UP may move only the selection',['e1'],page['id'],1)
            with self.assertRaisesRegex(ValueError,'stale'):
                n.write('hypothesis','UP effect','overwrite',['e1'],page['id'],1)
            self.assertEqual(n.read('experiment')['page'],updated)
            n.erase(page['id'],2,'Contradicted by the next observation')
            self.assertNotIn('experiment',n.bookmarks)
            self.assertNotIn(page['id'],[p['id'] for p in n.read()['notes']])
            self.assertEqual(n.read(page['id'])['page']['status'],'withdrawn')
            saved=json.loads((Path(d)/f'{page["id"]}-v1.json').read_text())
            self.assertEqual(saved,page)

    def test_agent_cannot_rewrite_host_results_or_fabricate_evidence(self):
        n=self.notebook();original=n.read('latest_result')
        with self.assertRaisesRegex(ValueError,'host-owned'):
            n.write('plan','Changed result','Succeeded',[], 'result-1',1)
        with self.assertRaisesRegex(ValueError,'host-owned'):
            n.erase('result-1',1,'Hide failure')
        with self.assertRaisesRegex(ValueError,'experience'):
            n.write('hypothesis','Rule','Always works',['invented'])
        with self.assertRaisesRegex(ValueError,'references'):
            n.write('interpretation','Meaning','Opened a door',[])
        self.assertEqual(n.read('latest_result'),original)
        page=n.write('hypothesis','Scene hypothesis','This might be a door',['o1'])
        self.assertEqual(page['evidence_ids'],['o1'])

    def test_goal_is_single_source_and_results_cannot_be_bookmark_spoofed(self):
        n=self.notebook()
        with self.assertRaisesRegex(ValueError,'existing goal'):
            n.write('goal','Other goal','Replace the goal',[])
        with self.assertRaisesRegex(ValueError,'automatically'):
            n.bookmark('latest_result','goal')
        with self.assertRaisesRegex(ValueError,'instead of erasing'):
            n.erase('goal',1,'Remove goal')
        n.write('goal','Current goal','Test one visible cell',[], 'goal',1)
        self.assertEqual(n.opening()['goal']['revision'],2)

    def test_boundaries_hide_old_local_notes_but_keep_explicit_access(self):
        n=self.notebook()
        page=n.write('plan','Local plan','Test a cell',[])
        n.bookmark('experiment',page['id'])
        n.record_observation({'observation_id':'o2','step':2,'state':'NOT_FINISHED'},None,'level')
        self.assertEqual(n.opening()['segment'],1)
        self.assertNotIn('experiment',n.bookmarks)
        self.assertEqual(n.read(query='Local')['notes'],[])
        self.assertEqual(len(n.read(query='Local',include_previous=True)['notes']),1)
        self.assertEqual(n.read(page['id'])['page']['segment'],0)
        with self.assertRaisesRegex(ValueError,'previous level'):
            n.write('plan','Local plan','Edit old plan',[],page['id'],1)

    def test_retrieval_and_bookmarks_are_bounded_and_return_copies(self):
        n=self.notebook()
        for i in range(10):
            page=n.write('plan',f'Plan {i}','x'*1000,[])
            if i<6:n.bookmark(f'plan{i}',page['id'])
        with self.assertRaisesRegex(ValueError,'limit'):
            n.bookmark('overflow',page['id'])
        batch=n.read(query='Plan')
        self.assertEqual(len(batch['notes']),6)
        self.assertEqual(batch['next_offset'],6)
        self.assertEqual(len(n.read(query='Plan',offset=6)['notes']),4)
        self.assertTrue(all(len(p['preview'])<=120 for p in batch['notes']))
        opening=n.opening();opening['goal']['text']='tamper'
        self.assertNotEqual(n.opening()['goal']['text'],'tamper')
        self.assertNotIn('x'*1000,json.dumps(opening))
        self.assertEqual(n.read(query='UP')['notes'][0]['id'],'result-1')


class NotebookRuntimeTests(unittest.TestCase):
    def test_native_notes_bookmarks_and_replay_never_show_future_revision(self):
        with tempfile.TemporaryDirectory() as d:
            r=CognitiveRuntime('test','local/qwen3-vl-4b-instruct',log_dir=d,max_http_requests=12)
            self.addCleanup(r.close)
            replies=[('write_note',{'kind':'plan','title':'Experiment','text':'Click a visible cell','evidence_ids':[]}),
                     ('set_bookmark',{'name':'experiment','note_id':'note-1'}),
                     ('read_notebook',{'reference':'experiment'}),
                     ('submit_decision',act())]
            def answer(model,payload):
                name,args=replies.pop(0)
                return call(name,args)
            with patch.object(LocalVisionLlm,'_complete',answer):r.decide(obs())
            self.assertEqual(r.result['action'],'ACTION6')
            self.assertEqual(r.errors,[])
            self.assertEqual(replies,[])
            self.assertEqual(r.obs['step'],0)
            # A later edit must not modify the page seen at the earlier read event.
            r.submission=None
            r.write_note('plan','Experiment','Future changed plan',[],'note-1',1)
            t=Timeline(Run(Path(d),r.session_id)).load()
            i=next(i for i,e in enumerate(t.events) if e['event']=='notebook_read')
            s=t.snapshot(i)
            self.assertEqual(s['notebook']['last_read']['result']['page']['text'],'Click a visible cell')
            self.assertNotIn('Future changed plan',json.dumps(s))
            self.assertEqual(t.snapshot(len(t.events)-1)['notebook']['last_change']['after']['revision'],2)

    def test_note_tool_errors_are_correctable_and_do_not_mutate_goal(self):
        r=CognitiveRuntime('test','local/qwen3-vl-4b-instruct')
        self.addCleanup(r.close);seen=[]
        def answer(model,p):
            seen.append(p)
            if len(seen)==1:
                return call('write_note',{'kind':'goal','title':'goal','text':'bad stale overwrite',
                            'evidence_ids':[],'note_id':'goal','expected_revision':99})
            feedback=json.loads(next(m['content'] for m in p['messages'] if m['role']=='tool'))
            self.assertIn('stale',feedback['error'])
            return call('submit_decision',act())
        with patch.object(LocalVisionLlm,'_complete',answer):r.decide(obs())
        self.assertNotEqual(r.notebook.opening()['goal']['text'],'bad stale overwrite')
        self.assertEqual(r.result['action'],'ACTION6')

    def test_work_scoping_and_shared_goal_when_builder_needs_more_evidence(self):
        r=CognitiveRuntime('test','local/qwen3-vl-4b-instruct')
        self.addCleanup(r.close);seen=[]
        def answer(model,p):
            c=context(p);seen.append(c)
            names=[t['function']['name'] for t in p['tools']]
            self.assertEqual(c['notebook']['goal']['text'],'Test the cell rule')
            if c['observation']['step']==0:
                self.assertNotIn('propose_skill',names)
                return call('submit_decision',act())
            if c['work']=='experiment_design':
                if c['last_tool_result'] and 'missing_evidence' in c['last_tool_result']:
                    return call('submit_decision',act(1))
                self.assertNotIn('propose_skill',names)
                return call('submit_decision',{'kind':'learn','evidence_ids':['experience-1'],
                                             'purpose':'Extract a cell rule'})
            self.assertIn('propose_skill',names)
            self.assertEqual(c['learning_request']['evidence_ids'],['experience-1'])
            self.assertTrue(c['notebook']['latest_result']['data']['outcome']['acknowledged'])
            self.assertNotIn('submit_decision', names)
            return call('defer_skill',{'reason':'Need another observed cell transition'})
        r.notebook.write('goal','Current goal','Test the cell rule',[],'goal',1)
        with patch.object(LocalVisionLlm,'_complete',answer):
            r.decide(obs());ack(r)
            result=r.decide(obs(1,[[1,0,0,0,0,0]]))
        self.assertEqual(result['x'],1)
        self.assertEqual([c['work'] for c in seen],['experiment_design','experiment_design','skill_creation','experiment_design'])
        self.assertEqual(r.work,'experiment_design')

    def test_action_may_cite_retained_experience_without_requesting_learning(self):
        r=CognitiveRuntime('test','local/qwen3-vl-4b-instruct')
        self.addCleanup(r.close)
        def answer(model,p):
            decision=act(context(p)['observation']['step'])
            if context(p)['observation']['step']:
                decision['evidence_ids']=['experience-1']
            return call('submit_decision',decision)
        with patch.object(LocalVisionLlm,'_complete',answer):
            r.decide(obs());ack(r)
            result=r.decide(obs(1))
        self.assertEqual(result['action'],'ACTION6')
        self.assertEqual(r.errors,[])
        self.assertEqual(r.calls,1)
