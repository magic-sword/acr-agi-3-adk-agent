import json
from pathlib import Path
import tempfile
import unittest
from scripts.agent_monitor import Timeline,Run
from scripts.cognition_view import cognition_html


class CognitionViewTests(unittest.TestCase):
    def test_cognitive_memory_does_not_leak_future_plan_or_skill_changes(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            events=[{'event':'cognition_updated','sequence':1,'cognition':{'plan':{'goal':'first'},'skills':{}}},
                    {'event':'cognition_updated','sequence':2,'cognition':{'plan':{'goal':'<script>future</script>'},
                     'skills':{'probe':{'effect':'new effect'}},'routing_history':[{'reason':'unexpected'}]}}]
            (root/'r.artifacts.jsonl').write_text('\n'.join(map(json.dumps,events))+'\n')
            timeline=Timeline(Run(root,'r')).load()
            self.assertNotIn('future',cognition_html(timeline.snapshot(0)))
            html=cognition_html(timeline.snapshot(1))
            self.assertIn('new effect',html);self.assertIn('unexpected',html)
            self.assertNotIn('<script>',html);self.assertIn('&lt;script&gt;',html)
            self.assertEqual(timeline.snapshot(0)['cognition']['plan']['goal'],'first')
