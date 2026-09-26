"""Architecture reads source without executing it; replay preserves temporal boundaries."""
import json
from pathlib import Path
import tempfile
import unittest
from scripts.runtime_structure import read_structure, structure_html, failure_points, trace_html
from scripts.agent_monitor import discover_evaluations

ROOT=Path(__file__).resolve().parents[1]

class StructureTests(unittest.TestCase):
    def test_registry_and_contracts_are_read_from_current_source(self):
        data=read_structure(ROOT)
        tasks={t['id']:t for t in data['tasks']}
        self.assertEqual(set(tasks), {'understand','backchain','ground','reconcile','choose_skill','execute_step'})
        self.assertEqual(tasks['ground']['tool'], 'submit_grounding')
        self.assertIn('goals', tasks['backchain']['fields'])
        self.assertIn('targets', tasks['understand']['fields'])
        self.assertEqual(data['edges'][0],['START','DECIDE'])

    def test_changed_registry_appears_without_execution_or_display_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);p=root/'agent/cognition';p.mkdir(parents=True)
            (p/'workflow.py').write_text("raise RuntimeError('must not execute')\nWorkflow(edges=[('START', decide),(decide,run)])")
            (p/'tasks.py').write_text("raise RuntimeError('must not execute')\nclass NewAnswer:\n    reason: str\nTASKS={'new_task':('submit_new',NewAnswer)}\nINSTRUCTIONS={'new_task':'<script>untrusted</script>'}")
            (p/'machine.py').write_text("STATES={'new_task':('New','llm',0,0)}\nTRANSITIONS={}\nGLOBAL_GATES=[]")
            data=read_structure(root)
            self.assertEqual([t['id'] for t in data['tasks']],['new_task'])
            html=structure_html(data)
            self.assertIn('new_task',html);self.assertNotIn('<script>',html)
            before=data['hash']
            (p/'tasks.py').write_text((p/'tasks.py').read_text().replace('submit_new','submit_other'))
            self.assertNotEqual(read_structure(root)['hash'],before)

    def test_failure_navigation_and_visible_prefix_do_not_infer_causes(self):
        events=[{'event':'state_entered','state':'DECIDE','work':'assess_goal','sequence':1},
                {'event':'task_rejected','reason':'same_test','sequence':2,'step':0},
                {'event':'model_decision','error':'bad output','sequence':3,'step':0},
                {'event':'state_exited','state':'RUN','sequence':4,'output':{'result':{'status':'stop','reason':'model_budget'}}}]
        self.assertEqual([i for _,i in failure_points(events)],[1,2,3])
        self.assertNotIn('model_budget',trace_html(events,1))
        self.assertIn('model_budget',trace_html(events,3))

    def test_outputs_root_discovers_evaluation_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);evaluation=root/'evaluations'/'id-1';game=evaluation/'game'
            (game/'cognition').mkdir(parents=True)
            (evaluation/'manifest.json').write_text('{}')
            (game/'result.json').write_text('{}')
            (game/'cognition/r.states.jsonl').write_text('')
            self.assertEqual(discover_evaluations(root),[evaluation])

    def test_widget_load_and_problem_jump_with_current_record(self):
        from scripts.notebook_monitor import BenchmarkReplay
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);evaluation=root/'evaluations'/'current';game=evaluation/'game'
            log=game/'cognition';log.mkdir(parents=True)
            (evaluation/'manifest.json').write_text('{"observatory_schema":3}')
            (game/'result.json').write_text('{}')
            (log/'r.states.jsonl').write_text(json.dumps({'event':'state_entered','state':'DECIDE','work':'assess_goal','sequence':1,'input':{'work':'assess_goal'}})+'\n'+json.dumps({'event':'state_exited','state':'RUN','sequence':2,'output':{'result':{'status':'stop','reason':'model_budget'}}})+'\n')
            viewer=BenchmarkReplay(root,source_root=ROOT);self.addCleanup(viewer.close)
            self.assertIn('<svg',viewer.diagram.value)
            viewer.load();self.assertIsNotNone(viewer.timeline)
            viewer.problem.value=1
            self.assertEqual(viewer.snapshot()['index'],1)
            self.assertEqual(viewer.mode.value,'イベント')
            self.assertFalse(hasattr(viewer,'structure_choice'))
            self.assertEqual(viewer.structure['root'],str(ROOT))

    def test_widget_rejects_old_log_but_keeps_latest_graph(self):
        from scripts.notebook_monitor import BenchmarkReplay
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);game=root/'game';log=game/'cognition';log.mkdir(parents=True)
            (root/'manifest.json').write_text('{}');(game/'result.json').write_text('{}')
            (log/'r.states.jsonl').write_text('')
            viewer=BenchmarkReplay(root,source_root=ROOT);self.addCleanup(viewer.close)
            viewer.load()
            self.assertIsNone(viewer.timeline)
            self.assertIn('旧形式',viewer.status.value)
            self.assertIn('data-state="observe"',viewer.diagram.value)

    def test_all_task_states_and_feedback_are_visible_without_logs(self):
        data=read_structure(ROOT);html=structure_html(data)
        self.assertIn('<svg',html)
        for task in data['tasks']:
            self.assertIn('data-state="'+task['id']+'"',html)
        for event in ('next_observation','repair_ground','invalid_ground'):
            self.assertIn('data-transition="'+event+'"',html)

    def test_missing_machine_definition_is_an_error(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);p=root/'agent/cognition';p.mkdir(parents=True)
            (p/'workflow.py').write_text('')
            with self.assertRaisesRegex(ValueError,'最新'):
                read_structure(root)

    def test_runtime_routing_and_graph_share_transition_targets(self):
        from agent.cognition.machine import destination, TRANSITIONS
        data=read_structure(ROOT)
        self.assertEqual(data['transitions'],TRANSITIONS)
        self.assertEqual(destination('repair_ground'), 'ground')
        self.assertEqual(destination('invalid_ground'), 'stop')
