"""Replay must show what was knowable then, and must never act on the game."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.agent_monitor import read_journal, Timeline, Run, discover_evaluations, discover_runs, dashboard_html, safe_asset, screen_html
from agent.cognition.workflow import CognitiveRuntime
from agent.local_vlm import LocalVisionLlm
from test_skill_learning import obs
from test_decide_run import call, act


class ReplayTests(unittest.TestCase):
    def test_partial_final_line_is_not_a_completed_event(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'r.states.jsonl'
            line=json.dumps({'event':'state_entered','state':'DECIDE','input':'観測','sequence':1},ensure_ascii=False).encode()+b'\n'
            path.write_bytes(line[:-3])
            rows,invalid=read_journal(path);self.assertEqual(rows,[])
            path.write_bytes(line)
            t=Timeline(Run(Path(d),'r')).load()
            self.assertEqual(len(t.events),1)
            path.write_text('')
            # Saved playback remains fixed until explicitly loaded again.
            self.assertEqual(len(t.events),1)
            t.load();self.assertEqual(t.events,[])

    def test_replay_does_not_leak_future_output_or_ack_or_frame(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            events=[('observations',{'event':'observation_received','step':0,'grid':[[0]]}),
                    ('states',{'event':'state_entered','state':'DECIDE','input':{'task':'test'}}),
                    ('states',{'event':'state_exited','state':'DECIDE','output':{'secret':'future'}}),
                    ('artifacts',{'event':'action_selected','state':'RUN','step':0,'pending':{'action':{'action':'UP'}}}),
                    ('execution',{'event':'action_acknowledged','step':0,'action':{'action':'UP'}}),
                    ('observations',{'event':'observation_received','step':1,'grid':[[1]]})]
            for i,(kind,row) in enumerate(events):
                with (root/f'r.{kind}.jsonl').open('a') as f:f.write(json.dumps(dict(row,sequence=i+1))+'\n')
            t=Timeline(Run(root,'r'));t.load()
            s=t.snapshot(1);self.assertEqual(s['phase'],'処理中');self.assertIsNone(s['output']);self.assertIsNone(s['action'])
            self.assertEqual(s['current']['step'],0)
            self.assertIn('未送信',t.snapshot(3)['action_status'])
            self.assertIn('次の観測待ち',t.snapshot(4)['action_status'])
            self.assertEqual(t.snapshot(5)['current']['step'],1)

    def test_model_text_is_escaped_and_state_comes_from_boundary(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            (root/'r.states.jsonl').write_text(json.dumps({'event':'state_entered','state':'DECIDE',
                'input':{'notebook':{'goal':{'text':'<script>bad()</script>'}}},'sequence':2})+'\n')
            (root/'r.observations.jsonl').write_text(json.dumps({'event':'observation_received',
                'grid':[[8,9]],'step':0,'sequence':1})+'\n')
            # A tool's execution label must not replace the actual graph state.
            (root/'r.tools.jsonl').write_text(json.dumps({'event':'tool_started','state':'RUN',
                'tool':'test','sequence':3})+'\n')
            t=Timeline(Run(root,'r'));t.load();s=t.snapshot(2)
            self.assertEqual(s['state'],'DECIDE');self.assertIsNone(s['output'])
            html=dashboard_html(s,root)
            self.assertNotIn('<script>',html);self.assertIn('&lt;script&gt;',html)
            self.assertIn('#F93C31',html)

    def test_asset_path_and_click_overlay(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):safe_asset(Path(d),'../other.png')
            html=screen_html({'grid':[[0,0]]},Path(d),{'action':'CLICK','x':1,'y':0})
            self.assertIn('cx="1.5"',html)

    def test_state_entry_is_flushed_before_model_returns(self):
        with tempfile.TemporaryDirectory() as d:
            r=CognitiveRuntime('test','local/qwen3-vl-4b-instruct',log_dir=d)
            self.addCleanup(r.close)
            def answer(model,payload):
                timeline=Timeline(Run(Path(d),r.session_id));timeline.load()
                s=timeline.snapshot(len(timeline.events)-1)
                self.assertEqual(s['state'],'DECIDE');self.assertEqual(s['phase'],'処理中')
                self.assertIsNone(s['output'])
                return call('submit_decision',act())
            with patch.object(LocalVisionLlm,'_complete',answer):r.decide(obs())
            timeline=Timeline(Run(Path(d),r.session_id));timeline.load()
            rows=[e for e in timeline.events if e['_journal']=='states']
            self.assertEqual([(e['event'],e['state']) for e in rows],
                             [('state_entered','DECIDE'),('state_exited','DECIDE'),
                              ('state_entered','RUN'),('state_exited','RUN')])
            sequence=[e['sequence'] for e in timeline.events]
            self.assertEqual(sequence,sorted(set(sequence)))

    def fixture(self, root, evaluation='20260925T100000Z',game='ls20'):
        folder=root/evaluation;directory=folder/game/'cognition';directory.mkdir(parents=True)
        (folder/'manifest.json').write_text('{}')
        (directory.parent/'result.json').write_text('{}')
        for kind,rows in {
            'observations':[{'event':'observation_received','sequence':1,'step':0,'grid':[[8,9]],'observation_id':'o0'}],
            'states':[{'event':'state_entered','state':'DECIDE','sequence':2,'step':0,'input':{'task':'test'}},
                      {'event':'state_exited','state':'DECIDE','sequence':4,'step':0,'output':{'done':True}}],
            'requests':[{'event':'model_request','sequence':3,'step':0,'request':{'large':'data'}}]
        }.items():
            (directory/f'r.{kind}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
        return folder

    def test_dropdowns_load_only_the_selected_benchmark_game(self):
        from scripts.notebook_monitor import BenchmarkReplay
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);one=self.fixture(root);two=self.fixture(root,'20260925T110000Z','ft09')
            self.assertEqual(discover_evaluations(root),[two,one])
            with patch.object(Timeline,'load',autospec=True,side_effect=AssertionError('must not eagerly load')):
                w=BenchmarkReplay(root)
                self.addCleanup(w.close)
                self.assertIsNone(w.timeline)
                w.evaluation.value=str(one)
                self.assertIn('ls20',w.game.options[0][0])
            w.load();self.assertEqual(w.timeline.run.directory.parent.name,'ls20')
            w.evaluation.value=str(two);self.assertIsNone(w.timeline)
            w.load();self.assertEqual(w.timeline.run.directory.parent.name,'ft09')
            self.assertFalse(hasattr(w,'start'));self.assertFalse(hasattr(w,'_watch'))

    def test_playback_is_cached_and_detail_is_rendered_only_when_opened(self):
        from scripts.notebook_monitor import BenchmarkReplay
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);self.fixture(root)
            w=BenchmarkReplay(root);self.addCleanup(w.close)
            w.mode.value='操作'
            with patch('scripts.notebook_monitor.json_html',return_value='detail') as detail:
                w.load();self.assertEqual(detail.call_count,0)
                self.assertEqual(w.slider.max,0)  # One game step, four individual events.
                w.mode.value='イベント';self.assertEqual(w.slider.max,3)
                with patch('scripts.notebook_monitor.picture',side_effect=AssertionError('unchanged image rebuilt')):
                    w.slider.value=1;w.slider.value=2
                self.assertEqual(detail.call_count,0)
                w.details.selected_index=3;self.assertEqual(detail.call_count,1)
                self.assertFalse(w.play.playing)
                w.play.playing=True;self.assertIsNone(w.details.selected_index)
            s=w.timeline.snapshot(1)
            self.assertIsNone(s['output'])
            self.assertEqual(w.timeline.snapshot(3)['output'],{'done':True})
            self.assertIsNone(s['output'])  # Later snapshots must not mutate the past.
            self.assertIs(s,w.timeline.snapshot(1))

    def test_diagram_tracks_work_on_entry_and_driver_wait_without_future_leak(self):
        from scripts.state_diagram import state_diagram_html
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            events=[
                ('observations',{'event':'observation_received','step':0,'grid':[[0]]}),
                ('states',{'event':'state_entered','state':'DECIDE','work':'action','input':{}}),
                ('states',{'event':'state_exited','state':'DECIDE','work':'action','output':{}}),
                ('states',{'event':'state_entered','state':'RUN','work':'action','input':{'job':{'kind':'learn'}}}),
                ('states',{'event':'state_exited','state':'RUN','work':'skill_creation','output':{'result':None}}),
                ('states',{'event':'state_entered','state':'DECIDE','work':'skill_creation','input':{}}),
                ('tools',{'event':'tool_started','state':'RUN','work':'action','tool':'read_notebook'}),
                ('states',{'event':'state_exited','state':'DECIDE','work':'skill_creation','output':{}}),
                ('states',{'event':'state_entered','state':'RUN','work':'skill_creation','input':{'job':{'spec':{}}}}),
                ('states',{'event':'state_exited','state':'RUN','work':'action','output':{'result':None}}),
                ('states',{'event':'state_entered','state':'DECIDE','work':'action','input':{}}),
                ('states',{'event':'state_entered','state':'RUN','work':'action','input':{'job':{'kind':'act'}}}),
                ('states',{'event':'state_exited','state':'RUN','work':'action','output':{'result':{'status':'action'}}}),
                ('execution',{'event':'action_dispatched','step':0,'action':{'action':'UP'}}),
                ('execution',{'event':'action_acknowledged','step':0,'action':{'action':'UP'}}),
                ('observations',{'event':'observation_received','step':1,'grid':[[1]]}),
                ('states',{'event':'state_entered','state':'RUN','work':'action','input':{}}),
                ('states',{'event':'state_exited','state':'RUN','work':'action','output':{'result':{'status':'stop','reason':'win'}}}),
                ('states',{'event':'runtime_closed','stop_reason':'win'}),
            ]
            for i,(kind,row) in enumerate(events):
                with (root/f'r.{kind}.jsonl').open('a') as f:
                    f.write(json.dumps(dict(row,sequence=i+1))+'\n')
            t=Timeline(Run(root,'r')).load()
            self.assertEqual([s['machine']['node'] for s in t.snapshots],
                ['observe','action','action','run','run','build','build','build','run',
                 'run','action','run','wait','wait','wait','observe','run','end','end'])
            self.assertEqual(t.snapshot(4)['machine']['work'],'action')
            self.assertEqual(t.snapshot(9)['machine']['work'],'skill_creation')
            self.assertEqual(t.snapshot(8)['machine']['job'],'propose_skill')
            self.assertIn('未送信',t.snapshot(12)['machine']['phase'])
            self.assertIn('受付待ち',t.snapshot(13)['machine']['phase'])
            self.assertIn('次の観測待ち',t.snapshot(14)['machine']['phase'])
            self.assertEqual(t.snapshot(17)['machine']['phase'],'win')
            html=state_diagram_html(t.snapshot(5)['machine'])
            self.assertEqual(html.count('data-active="true"'),1)
            self.assertIn('data-state="build" data-active="true"',html)
            self.assertIn('スキル作成 · 処理中',html)
            self.assertNotIn('<script>',state_diagram_html({'node':'action','phase':'<script>x</script>'}))

    def test_state_replay_skips_same_state_events_and_caches_the_diagram(self):
        from scripts.notebook_monitor import BenchmarkReplay
        from scripts.state_diagram import state_diagram_html
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);self.fixture(root)
            w=BenchmarkReplay(root);self.addCleanup(w.close)
            self.assertEqual(w.mode.value,'状態遷移')
            w.load()
            # Observation, state entry and exit; the intervening HTTP request is skipped.
            self.assertEqual(w._positions,[0,1,3])
            w.mode.value='イベント'
            w.slider.value=1
            with patch('scripts.notebook_monitor.state_diagram_html',wraps=state_diagram_html) as diagram:
                w.slider.value=2
                diagram.assert_not_called()
                w.slider.value=3
                diagram.assert_called_once()
            self.assertIn('完了',w.diagram.value)
