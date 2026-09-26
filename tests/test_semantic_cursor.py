"""Visual pointing is local; only an explicit, current-frame click crosses the driver boundary."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from agent.cognition.cursor import start_cursor, cursor_options, adjust_cursor, render_cursor, observation_board
from agent.cognition.state import Understanding, ActionIntent
from agent.cognition.validation import validate_intent
from agent.cognition.workflow import CognitiveRuntime
from agent.local_vlm import LocalVisionLlm
from agent.observation import attach_visuals
from runtime_helpers import obs, context, call, token, grounding, understanding, answer, ack


def board(step=0):
    value=obs(step,[[5]*64 for _ in range(64)])
    value['available_actions']=['ACTION6']
    value['grid'][15][47]=8
    attach_visuals(value,[value['grid']])
    return value


def click_answer(model,payload):
    c=context(payload)
    if c['work']=='ground':
        v=grounding(c,x=0)
        v['plan']['skills'][0]['steps'][0]['options'][0]['action']['target_query']='The red cell in the upper-right region.'
        return call('submit_grounding',v)
    return answer(model,payload)


class SemanticCursorTests(unittest.TestCase):
    def runtime(self,**kwargs):
        r=CognitiveRuntime('test','local/qwen3-vl-4b-instruct',**kwargs)
        self.addCleanup(r.close)
        return r

    def test_shared_concepts_support_groups_and_distinct_roles_without_coordinates(self):
        value=understanding({'observation_id':'o'})
        value['concepts']=[{'name':'panel','description':'A three by three arrangement of colored cells.'}]
        value['targets']=[
            {'concept':'panel','appearance':'Three plain panels','role_hypothesis':'Possible examples',
             'relations':'Surround the highlighted panel'},
            {'concept':'panel','appearance':'The highlighted panel','role_hypothesis':'Possible editable board',
             'relations':'Shares its layout with the three plain panels'}]
        parsed=Understanding.model_validate(value)
        self.assertEqual(parsed.targets[0].concept,parsed.targets[1].concept)
        self.assertNotEqual(parsed.targets[0].role_hypothesis,parsed.targets[1].role_hypothesis)
        for target in parsed.targets:
            self.assertNotIn('x',target.model_dump());self.assertNotIn('id',target.model_dump())
        with self.assertRaises(ValueError):
            ActionIntent.model_validate({'action':'CLICK','x':32,'y':32})
        with self.assertRaisesRegex(ValueError,'target_query'):
            validate_intent({'action':'CLICK'},board())

    def test_cursor_moves_then_clicks_once_with_audited_binding_and_images(self):
        with tempfile.TemporaryDirectory() as directory:
            r=self.runtime(log_dir=directory);requests=[];labels=iter(['2','5','7'])
            original=board();saved=deepcopy(original)
            def respond(m,p):
                requests.append(p);c=context(p)
                if c['work']=='ground':
                    definition=p['tools'][0]['function']['parameters']['$defs']['ActionIntent']
                    self.assertEqual(definition['required'],['action','target_query'])
                    self.assertNotIn('x',definition['properties'])
                if c['work']=='aim':
                    self.assertIsNone(r.memory.pending)
                    self.assertEqual(r.obs['remaining_actions'],30)
                    self.assertEqual(c['target_query'],'The red cell in the upper-right region.')
                    self.assertEqual(c['cursor']['observation_id'],r.obs['observation_id'])
                    self.assertEqual(p['max_tokens'],1)
                    self.assertIn('"8"',p['grammar'])
                    self.assertEqual(sum(part['type']=='image_url' for part in p['messages'][-1]['content']),
                                     1 if c['cursor']['mode']=='locate' else 2)
                    if c['cursor']['mode']=='locate':
                        self.assertIsNone(c['cursor']['x']);self.assertNotIn('7',c['choices'])
                    return token(next(labels))
                return click_answer(m,p)
            with patch.object(LocalVisionLlm,'_complete',respond):result=r.decide(original)
            self.assertEqual((result['action'],result['x'],result['y']),('ACTION6',47,15))
            self.assertEqual(original,saved)
            self.assertEqual(r.memory.pending['binding']['adjustments'],1)
            self.assertTrue(r.memory.pending['binding']['confirmed'])
            self.assertNotIn('x',r.memory.skills['move']['steps'][0]['options'][0]['action'])
            self.assertEqual(r.memory.active_skill['action_count'],0)
            records=[json.loads(line) for line in Path(directory,f'{r.session_id}.requests.jsonl').read_text().splitlines()]
            preview=records[-1]['request']['messages'][-1]['content'][-2]['image_url']
            self.assertTrue(Path(directory,preview['path']).is_file())
            from scripts.notebook_monitor import aim_preview
            snapshot={'request':records[-1],'current':{'observation_id':r.obs['observation_id']}}
            self.assertEqual(aim_preview(snapshot,Path(directory)),Path(directory,preview['path']).read_bytes())
            snapshot['current']['observation_id']='next'
            self.assertEqual(aim_preview(snapshot,Path(directory)),b'')
            ack(r)

    def test_same_query_reacquires_position_after_each_observation(self):
        r=self.runtime();seen=[]
        def respond(m,p):
            c=context(p)
            if c['work']=='aim':
                cur=c['cursor'];seen.append(deepcopy(cur))
                if cur['mode']=='locate':
                    return token('2' if r.obs['step']==0 else '1')
                return token('7')
            return click_answer(m,p)
        with patch.object(LocalVisionLlm,'_complete',respond):
            first=r.decide(board());ack(r)
            second=r.decide(board(1))
        self.assertEqual((first['x'],second['x']),(47,15))
        starts=[c for c in seen if c['mode']=='locate']
        self.assertEqual([c['x'] for c in starts],[None,None])
        self.assertNotEqual(starts[0]['observation_id'],starts[1]['observation_id'])
        self.assertEqual(starts[0]['target_query'],starts[1]['target_query'])
        self.assertEqual(r.outcome['binding']['x'],47)

    def test_aim_eight_reconciles_without_click_or_receipt(self):
        r=self.runtime();aims=0
        def respond(m,p):
            nonlocal aims
            c=context(p)
            if c['work']=='aim':
                aims+=1;return token('8' if aims==1 else '2' if c['cursor']['mode']=='locate' else '7')
            if c['work']=='reconcile':
                self.assertIsNone(r.memory.pending)
                self.assertIsNone(c['current_invocation_result'])
                self.assertEqual(c['review']['cursor']['target_query'],'The red cell in the upper-right region.')
                self.assertFalse(c['review']['cursor']['confirmed'])
            return click_answer(m,p)
        with patch.object(LocalVisionLlm,'_complete',respond):result=r.decide(board())
        self.assertEqual(result['status'],'action');self.assertEqual(aims,3)
        self.assertEqual(r.memory.active_skill['action_count'],0)

    def test_deadline_during_adjustment_or_confirmation_never_clicks(self):
        for label in ('4','7'):
            with self.subTest(label=label):
                r=self.runtime()
                def respond(m,p):
                    if context(p)['work']=='aim':
                        if context(p)['cursor']['mode']=='locate' and label=='7':return token('2')
                        r.turn_deadline=0;return token(label)
                    return click_answer(m,p)
                with patch.object(LocalVisionLlm,'_complete',respond):result=r.decide(board())
                self.assertEqual(result['reason'],'decision_time_exhausted')
                self.assertIsNone(r.memory.pending)

    def test_probe_cursor_confirmation_is_a_click_not_goal_completion(self):
        r=self.runtime();works=[]
        def respond(m,p):
            c=context(p);works.append(c['work'])
            if c['work']=='ground':
                v=grounding(c,x=0);v['plan'].update(intent='probe',question='Does the marked cell change?')
                return call('submit_grounding',v)
            if c['work']=='execute_step':self.assertNotIn('7',c['choices'])
            if c['work']=='aim':
                return token('2' if c['cursor']['mode']=='locate' else '7')
            if c['work']=='reconcile':
                self.assertEqual(c['review']['trigger'],'probe_result')
                self.assertTrue(c['current_invocation_result']['binding']['confirmed'])
                self.assertEqual(c['current_invocation_result']['action']['action'],'CLICK')
            return answer(m,p)
        with patch.object(LocalVisionLlm,'_complete',respond):
            result=r.decide(board());self.assertEqual(result['status'],'action')
            self.assertEqual(r.memory.goal_status['approach']['status'],'active')
            ack(r);r.decide(board(1))
        self.assertEqual(works[works.index('reconcile')-1],'aim')
        self.assertEqual(r.memory.reconciliations[0]['trigger']['trigger'],'probe_result')

    def test_rendering_preserves_raw_pixels_and_checks_observation_binding(self):
        o=board();cursor=start_cursor(o,'red cell');original=deepcopy(o)
        adjust_cursor(cursor,cursor_options(cursor,o)['2'])
        preview=render_cursor(o,cursor)
        self.assertGreater(preview.width,428);self.assertEqual(o,original)
        raw=observation_board(o)
        image_only={k:v for k,v in o.items() if k not in ('grid','_visual_frames')}
        self.assertEqual(observation_board(image_only).tobytes(),raw.tobytes())
        cursor['observation_id']='stale'
        with self.assertRaisesRegex(ValueError,'stale'):render_cursor(o,cursor)
        image_only.pop('viewport')
        with self.assertRaisesRegex(ValueError,'viewport'):observation_board(image_only)

    def test_rgb_and_non_square_boards_keep_exact_pixel_mapping(self):
        o=obs(grid=[[0]*3 for _ in range(5)])
        o.pop('grid');rgb=[[[10,20,30] for _ in range(3)] for _ in range(5)]
        attach_visuals(o,[rgb]);cursor=start_cursor(o,'colored point')
        cursor.update(x=2,y=4,stride=1,mode='adjust')
        options=cursor_options(cursor,o)
        self.assertNotIn('2',options);self.assertNotIn('4',options)
        self.assertNotIn('5',options)
        adjust_cursor(cursor,options['1'])
        self.assertEqual((cursor['x'],cursor['y']),(2,3))
        self.assertEqual(observation_board(o).getpixel((2,3)),(10,20,30))
        self.assertIsInstance(render_cursor(o,cursor),Image.Image)
