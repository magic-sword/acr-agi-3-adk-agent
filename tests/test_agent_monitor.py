"""Replay must show what was knowable then, and must never act on the game."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.agent_monitor import Journal, Timeline, Run, discover_runs, dashboard_html, safe_asset, screen_html
from agent.cognition.workflow import CognitiveRuntime
from agent.local_vlm import LocalVisionLlm
from test_skill_learning import obs
from test_decide_run import call, act


class ReplayTests(unittest.TestCase):
    def test_partial_utf8_line_is_retried_and_truncation_clears_events(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'r.states.jsonl'
            line=json.dumps({'event':'state_entered','input':'観測','sequence':1},ensure_ascii=False).encode()+b'\n'
            path.write_bytes(line[:-3])
            j=Journal(path);j.refresh();self.assertEqual(j.rows,[])
            with path.open('ab') as f:f.write(line[-3:])
            j.refresh();self.assertEqual(j.rows[0]['input'],'観測')
            j.refresh();self.assertEqual(len(j.rows),1)
            t=Timeline(Run(Path(d),'r'));t.refresh();self.assertEqual(len(t.events),1)
            path.write_text('');self.assertTrue(t.refresh());self.assertEqual(t.events,[])

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
            t=Timeline(Run(root,'r'));t.refresh()
            s=t.snapshot(1);self.assertEqual(s['phase'],'処理中');self.assertIsNone(s['output']);self.assertIsNone(s['action'])
            self.assertEqual(s['current']['step'],0)
            self.assertIn('未送信',t.snapshot(3)['action_status'])
            self.assertIn('次の観測待ち',t.snapshot(4)['action_status'])
            self.assertEqual(t.snapshot(5)['current']['step'],1)

    def test_model_text_is_escaped_and_state_comes_from_boundary(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            (root/'r.states.jsonl').write_text(json.dumps({'event':'state_entered','state':'DECIDE',
                'input':{'task':'<script>bad()</script>'},'sequence':2})+'\n')
            (root/'r.observations.jsonl').write_text(json.dumps({'event':'observation_received',
                'grid':[[8,9]],'step':0,'sequence':1})+'\n')
            # A tool's execution label must not replace the actual graph state.
            (root/'r.tools.jsonl').write_text(json.dumps({'event':'tool_started','state':'RUN',
                'tool':'test','sequence':3})+'\n')
            self.assertEqual(len(discover_runs(root)),1)
            t=Timeline(discover_runs(root)[0]);t.refresh();s=t.snapshot(2)
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
                timeline=Timeline(Run(Path(d),r.session_id));timeline.refresh()
                s=timeline.snapshot(len(timeline.events)-1)
                self.assertEqual(s['state'],'DECIDE');self.assertEqual(s['phase'],'処理中')
                self.assertIsNone(s['output'])
                return call('submit_decision',act())
            with patch.object(LocalVisionLlm,'_complete',answer):r.decide(obs())
            timeline=Timeline(Run(Path(d),r.session_id));timeline.refresh()
            rows=[e for e in timeline.events if e['_journal']=='states']
            self.assertEqual([(e['event'],e['state']) for e in rows],
                             [('state_entered','DECIDE'),('state_exited','DECIDE'),
                              ('state_entered','RUN'),('state_exited','RUN')])
            sequence=[e['sequence'] for e in timeline.events]
            self.assertEqual(sequence,sorted(set(sequence)))

    def test_notebook_widget_seek_and_follow_are_distinct(self):
        from scripts.notebook_monitor import AgentMonitor
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'r.states.jsonl'
            path.write_text(json.dumps({'event':'state_entered','state':'DECIDE','sequence':1})+'\n')
            w=AgentMonitor(d);self.addCleanup(w.close)
            w.follow.value=True
            with path.open('a') as f:f.write(json.dumps({'event':'state_exited','state':'DECIDE','sequence':2,'output':{'x':1}})+'\n')
            w.refresh();self.assertEqual(w.slider.value,1)
            w.slider.value=0;self.assertFalse(w.follow.value)
            w.refresh();self.assertEqual(w.slider.value,0)
            self.assertIn('まだ記録',w.panels[1].value)

    def test_stop_only_owned_process_group(self):
        import asyncio,os,time
        from scripts.notebook_monitor import AgentMonitor
        async def exercise():
            with tempfile.TemporaryDirectory() as d:
                root=Path(d);(root/'scripts').mkdir()
                (root/'scripts/benchmark_local.py').write_text(
                    'import subprocess,sys,time\nfrom pathlib import Path\n'
                    'p=subprocess.Popen([sys.executable,"-c","import time; time.sleep(60)"])\n'
                    'Path("worker-pid").write_text(str(p.pid))\ntime.sleep(60)\n')
                w=AgentMonitor(root,project=root)
                try:
                    self.assertTrue(w.stop_button.disabled)
                    w.stop()  # Read-only monitors cannot stop another evaluation.
                    w.start(game='ls20',model=False,seconds=10)
                    deadline=time.monotonic()+5
                    while not (root/'worker-pid').exists() and time.monotonic()<deadline:
                        await asyncio.sleep(.05)
                    child=int((root/'worker-pid').read_text())
                    w.stop()
                    await asyncio.sleep(.3)
                    self.assertIsNotNone(w.process.poll())
                    stat=Path(f'/proc/{child}/stat')
                    self.assertTrue(not stat.exists() or stat.read_text().split()[2]=='Z')
                finally:
                    w.close(stop=True)
        asyncio.run(exercise())
