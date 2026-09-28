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
                     'skills':{'probe':{'effect':'new effect'}},'reconciliations':[{'reason':'unexpected'}]}}]
            (root/'r.artifacts.jsonl').write_text('\n'.join(map(json.dumps,events))+'\n')
            timeline=Timeline(Run(root,'r')).load()
            self.assertNotIn('future',cognition_html(timeline.snapshot(0)))
            html=cognition_html(timeline.snapshot(1))
            self.assertIn('new effect',html);self.assertIn('unexpected',html)
            self.assertNotIn('<script>',html);self.assertIn('&lt;script&gt;',html)
            self.assertEqual(timeline.snapshot(0)['cognition']['plan']['goal'],'first')

    def test_measurements_and_semantic_answers_follow_the_replay_position(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            measured={'measured_objects':{'changed_pixels':2},
                      'pending_semantic_questions':1,'semantic_answers':[]}
            answered={**measured,'pending_semantic_questions':0,
                      'deferred_semantic_questions':0,
                      'semantic_answers':[{'interpretation':'unknown'}]}
            rows=[{'event':'cognition_updated','sequence':i+1,'cognition':memory}
                  for i,memory in enumerate([measured,answered])]
            (root/'r.artifacts.jsonl').write_text('\n'.join(map(json.dumps,rows))+'\n')
            timeline=Timeline(Run(root,'r')).load()
            before=cognition_html(timeline.snapshot(0))
            after=cognition_html(timeline.snapshot(1))
            self.assertIn('プログラムによる物体別の測定',before)
            self.assertIn('changed_pixels',before)
            self.assertNotIn('unknown',before)
            self.assertIn('変更と期待効果についての意味回答',after)
            self.assertIn('unknown',after)
            self.assertIn('予算上保留した意味質問',after)
