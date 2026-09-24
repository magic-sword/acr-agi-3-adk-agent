"""Source drift and render checks; requires no game SDK or model."""
import json
from pathlib import Path
import shutil
import tempfile
import unittest
import xml.etree.ElementTree as ET

from scripts.visualize_agent import ROOT, extract, render_svg


class VisualizeAgentTests(unittest.TestCase):
    def test_current_wiring_and_svg(self):
        data = extract(ROOT)
        self.assertIn({'from': 'OBSERVE', 'to': 'DECIDE', 'label': 'DECIDE'}, data['edges'])
        self.assertNotIn('CONSOLIDATE', data['nodes'])
        self.assertIn('CONSOLIDATE', [x['state'] for x in data['internal']])
        expected = set(data['state_skills']['DECIDE'])
        self.assertEqual(set(data['effective_skills']['DECIDE']), expected)
        self.assertEqual(data['effective_skills']['COMMIT'], [])
        self.assertEqual(data['completion_tools']['DECIDE'], 'submit_decision')
        self.assertIn('agent/cognition/completion.py', data['sha256'])
        ET.fromstring(render_svg(data))
        json.dumps(data)

    def copy_sources(self, root):
        shutil.copytree(ROOT / 'agent/cognition', root / 'agent/cognition')
        shutil.copytree(ROOT / 'agent/skills', root / 'agent/skills')

    def test_source_changes_are_reflected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.copy_sources(root)
            path = root / 'agent/cognition/workflow.py'
            path.write_text(path.read_text().replace('(decide, commit)', '(decide, observe, commit)'))
            path = root / 'agent/cognition/skills.py'
            path.write_text(path.read_text().replace('"COMMIT": (),', '"COMMIT": ("S01",),'))
            data = extract(root)
            self.assertIn({'from': 'DECIDE', 'to': 'OBSERVE', 'label': ''}, data['edges'])
            self.assertNotIn({'from': 'DECIDE', 'to': 'COMMIT', 'label': ''}, data['edges'])
            self.assertEqual(data['effective_skills']['COMMIT'], ['S01'])

    def test_unsupported_edges_fail_explicitly(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.copy_sources(root)
            path = root / 'agent/cognition/workflow.py'
            path.write_text(path.read_text().replace('(decide, commit)', '(decide, create_destination())'))
            with self.assertRaisesRegex(ValueError, 'Unsupported workflow expression'):
                extract(root)


if __name__ == '__main__':
    unittest.main()
