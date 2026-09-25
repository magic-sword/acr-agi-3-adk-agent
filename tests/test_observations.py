"""Real pixels, archive boundaries, image transport and bounded macro execution."""
from copy import deepcopy
import base64
import io
import unittest
from unittest.mock import patch
from PIL import Image
from agent.rendering import render_current, frame_image, ORIGIN, SCALE
from agent.observation import attach_visuals
from agent.cognition.evidence import EvidenceStore
from agent.cognition.library import SkillLibrary
from agent.cognition.state import Draft
from agent.cognition.workflow import CognitiveRuntime
from agent.local_vlm import LocalVisionLlm
from test_skill_learning import obs, spec
from test_decide_run import call, context, ack


class ObservationTests(unittest.TestCase):
    def test_game_only_pixels_have_no_cursor_overlay(self):
        grid=[[0]*64 for _ in range(64)];grid[32][32]=9
        image=render_current(grid)
        expected=frame_image(grid)
        for x,y in ((0,0),(32,32),(33,32),(63,63)):
            self.assertEqual(image.getpixel((ORIGIN[0]+x*SCALE,ORIGIN[1]+y*SCALE)), expected.getpixel((x,y)))
        self.assertLess(image.width,500)

    def test_archive_raw_grid_and_animation_do_not_advance_time(self):
        store=EvidenceStore(capacity=2);self.addCleanup(store.close)
        a=obs();attach_visuals(a,[[[0]*6],[[0,1,0,0,0,0]]]);store.add(a)
        view=store.view(a['observation_id']);self.assertEqual(view['grid'],a['grid'])
        self.assertFalse(view['time_advanced'])
        animation=store.animation(a['animation']['event_id'])
        self.assertFalse(animation['time_advanced']);self.assertIsNone(animation['next_start_frame'])
        b=obs(1,[[1]*6]);store.add(b)
        self.assertEqual(store.compare(a['observation_id'],b['observation_id'])['changed_cell_count'],6)
        c=obs(2);store.add(c,boundary=True)
        with self.assertRaisesRegex(ValueError,'boundary'):store.compare(b['observation_id'],c['observation_id'])
        with self.assertRaisesRegex(ValueError,'evicted'):store.get(a['observation_id'])

    def test_parameterized_multi_step_skill_continues_without_another_model_call(self):
        runtime=CognitiveRuntime('test','local/qwen3-vl-4b-instruct');self.addCleanup(runtime.close)
        lib=runtime.library
        data=spec();second=deepcopy(data['steps'][0]);second['x']=5
        for guard in second['before']+second['after']:guard['x']=5
        data['steps'].append(second)
        def run_trace(x):
            start=lib.sequence
            a=obs(start);b=obs(start+1);b['grid'][0][x]=1
            c=obs(start+2,deepcopy(b['grid']));c['grid'][0][5]=1
            first=lib.add_experience(a,{'action':'ACTION6','x':x,'y':0},b,acknowledged=True)
            second=lib.add_experience(b,{'action':'ACTION6','x':5,'y':0},c,acknowledged=True)
            return [first,second]
        seeds=run_trace(0)
        key=lib.draft(Draft(spec=data,evidence_ids=[e['id'] for e in seeds],examples=[{'x':0}]))['skill_id']
        for x in (1,2):lib.finish_trial(key,{'x':x},run_trace(x),'pass')
        self.assertEqual(lib.evaluate(key)['status'],'pass')
        replies=[]
        def answer(model,p):
            replies.append(p)
            return call('submit_decision',{'kind':'invoke','skill_id':key,'arguments':{'x':3},'purpose':'Execute two checked steps.'})
        with patch.object(LocalVisionLlm,'_complete',answer):
            first=runtime.decide(obs());self.assertEqual(first['x'],3);ack(runtime)
            changed=obs(1);changed['grid'][0][3]=1
            second=runtime.decide(changed);self.assertEqual(second['x'],5)
        self.assertEqual(len(replies),1)

    def test_forged_skill_status_is_rejected_by_contract(self):
        from pydantic import ValidationError
        data={'spec':spec(),'evidence_ids':['experience-1'],'examples':[{'x':0}],'status':'active'}
        with self.assertRaises(ValidationError):Draft.model_validate(data)
