"""Behavioral gates use actual acknowledged transitions, not model claims."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from pydantic import ValidationError
from agent.cognition.library import SkillLibrary, condition, matches, step_action
from agent.cognition.state import Draft, SkillSpec, Condition


def obs(step=0, grid=None):
    grid = deepcopy(grid if grid is not None else [[0]*6])
    return {'game_id':'test','step':step,'observation_id':f'o{step}',
            'grid':grid,'frame_hash':str(grid),'width':len(grid[0]),'height':len(grid),
            'state':'NOT_FINISHED','levels_completed':0,'available_actions':['ACTION6'], 'remaining_actions':12-step}


def spec(name='change-cell'):
    return {'name':name,'description':'Change a zero cell by clicking it; stop if the effect differs.',
            'game_id':'test','parameters':['x'], 'steps':[{
                'action':'CLICK','x':'$x','y':0,
                'before':[{'kind':'cell_is','x':'$x','y':0,'value':0}],
                'after':[{'kind':'cell_changed','x':'$x','y':0},
                         {'kind':'cell_is','x':'$x','y':0,'value':1}]}]}


def experience(lib, x, *, acknowledged=True, value=1):
    before=obs(lib.sequence, [[0]*6]);after=obs(lib.sequence+1, [[0]*6]);after['grid'][0][x]=value
    return lib.add_experience(before, {'action':'ACTION6','x':x,'y':0},after,acknowledged=acknowledged)


def candidate(lib):
    seed=experience(lib,0)
    return lib.draft(Draft(spec=spec(),evidence_ids=[seed['id']],examples=[{'x':0}]))['skill_id']


def promote(lib,key):
    for x in (1,2):
        lib.finish_trial(key,{'x':x},[experience(lib,x)],'pass')
    return lib.evaluate(key)


class SkillGateTests(unittest.TestCase):
    def test_seed_alone_cannot_promote_and_fresh_trials_can(self):
        lib=SkillLibrary('test');key=candidate(lib)
        self.assertEqual(lib.evaluate(key)['status'],'unknown')
        self.assertEqual(lib.get(key)['status'],'candidate')
        self.assertEqual(promote(lib,key)['status'],'pass')
        self.assertEqual(lib.get(key)['status'],'active')

    def test_repeated_seed_or_duplicate_trial_is_not_heldout_evidence(self):
        lib=SkillLibrary('test');key=candidate(lib)
        for x in (0,0,1,1):
            lib.finish_trial(key,{'x':x},[experience(lib,x)],'pass')
        self.assertFalse(lib.evaluate(key)['checks']['fresh_behavior'])

    def test_unknown_failed_and_unacknowledged_trials_never_promote(self):
        for outcome,ack,value in [('unknown',True,1),('pass',False,1),('pass',True,0)]:
            with self.subTest(outcome=outcome,ack=ack):
                lib=SkillLibrary('test');key=candidate(lib)
                lib.finish_trial(key,{'x':3},[experience(lib,3,acknowledged=ack,value=value)],outcome)
                self.assertNotEqual(promote(lib,key)['status'],'pass')

    def test_candidate_requires_real_matching_evidence(self):
        lib=SkillLibrary('test')
        with self.assertRaisesRegex(ValueError,'retained'):
            lib.draft(Draft(spec=spec(),evidence_ids=['fake'],examples=[{'x':0}]))
        e=experience(lib,0,acknowledged=False)
        with self.assertRaisesRegex(ValueError,'complete observed'):
            lib.draft(Draft(spec=spec(),evidence_ids=[e['id']],examples=[{'x':0}]))

    def test_historical_success_cannot_masquerade_as_fresh_trial(self):
        lib=SkillLibrary('test');old=experience(lib,1);key=candidate(lib)
        lib.finish_trial(key,{'x':1},[old],'pass')
        self.assertEqual(lib.get(key)['trials'][0]['outcome'],'fail')

    def test_bounds_unknown_guard_and_negative_fixture(self):
        s=SkillSpec.model_validate(spec())
        for args in ({'x':True},{'x':64},{'x':-1},{'x':0,'extra':1},{}):
            with self.assertRaises(ValueError):step_action(s,0,args,obs())
        with self.assertRaisesRegex(ValueError,'precondition'):
            step_action(s,0,{'x':1},obs(grid=[[1]*6]))
        self.assertIsNone(condition(Condition(kind='cell_changed',x=0,y=0),{},obs(),{}))
        lib=SkillLibrary('test');key=candidate(lib)
        self.assertTrue(lib.evaluate(key)['checks']['negative_guard'])

    def test_no_unbounded_or_unmeasured_programs(self):
        for change in ['empty_effect','extra_capability','undeclared_parameter']:
            data=spec()
            if change=='empty_effect':data['steps'][0]['after']=[{'kind':'cell_is','x':0,'y':0,'value':0}]
            if change=='extra_capability':data['script']='import os'
            if change=='undeclared_parameter':data['parameters']=[]
            with self.assertRaises(ValidationError):SkillSpec.model_validate(data)

    def test_version_hash_and_frozen_reload_gate(self):
        with tempfile.TemporaryDirectory() as d:
            lib=SkillLibrary('test',d);key=candidate(lib);promote(lib,key)
            path=Path(d)/'library.json'
            restored=SkillLibrary('test',source=path)
            self.assertEqual(restored.get(key)['status'],'active')
            data=json.loads(path.read_text());data['records'][key]['spec']['description']='tampered'
            path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError,'hash mismatch'):SkillLibrary('test',source=path)
            self.assertTrue((Path(d)/key/'procedure.json').is_file())

    def test_failed_revision_preserves_parent(self):
        lib=SkillLibrary('test');key=candidate(lib);promote(lib,key)
        proposal=spec();proposal['description']='Revised procedure'
        seed=experience(lib,0)
        revision=lib.draft(Draft(spec=proposal,parent_id=key,evidence_ids=[seed['id']],examples=[{'x':0}]))['skill_id']
        self.assertEqual(lib.evaluate(revision)['status'],'unknown')
        self.assertEqual(lib.get(key)['status'],'active')
        self.assertEqual(promote(lib,revision)['status'],'pass')
        self.assertEqual(lib.get(key)['status'],'suspended')

    def test_boundary_and_noncontiguous_trace_cannot_pass(self):
        lib=SkillLibrary('test');e=experience(lib,0);e['boundary']=True;e['boundary_kind']='reset'
        self.assertFalse(matches(SkillSpec.model_validate(spec()),{'x':0},[e]))
        double=spec();double['steps'].append(deepcopy(double['steps'][0]))
        e1=experience(lib,0);e2=experience(lib,0);e2['before']['observation_id']='different'
        self.assertFalse(matches(SkillSpec.model_validate(double),{'x':0},[e1,e2]))

    def test_scope_and_suspension(self):
        lib=SkillLibrary('test');key=candidate(lib);promote(lib,key)
        lib.suspend(key,'unexpected effect');self.assertEqual(lib.get(key)['status'],'suspended')
        lib.game_id='other'
        with self.assertRaisesRegex(ValueError,'game version'):lib.get(key)
