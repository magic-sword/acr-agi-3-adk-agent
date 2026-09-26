"""Memory navigation, provenance and non-destructive learning across ADK stages."""
from copy import deepcopy
import json
import unittest
from unittest.mock import patch

from agent.cognition.notebook import append_note, start_reader, reader_options, reader_context, navigate
from agent.cognition.state import Memory, Understanding, Reconciliation, Grounding
from agent.cognition.deliberation import StageTool
from agent.cognition.workflow import CognitiveRuntime
from agent.local_vlm import LocalVisionLlm
from runtime_helpers import obs, context, token, understanding, grounding, skill, answer, ack, call


class MemoryReaderTests(unittest.TestCase):
    def runtime(self):
        r=CognitiveRuntime('test','local/qwen3-vl-4b-instruct')
        self.addCleanup(r.close)
        r._receive(obs())
        return r

    def test_records_are_copied_and_hypotheses_cannot_overwrite_outcomes(self):
        r=self.runtime();m=r.memory;body={'action':'CLICK','changed_cell_count':0}
        outcome=r._write_note('outcome','No visible change',body,author='host')
        m.last_outcome_id=outcome;body['changed_cell_count']=99
        first=understanding({'observation_id':r.obs['observation_id']})
        r._accept_stage('understand',Understanding.model_validate(first))
        m.plan={'goal_id':None,'intent':'probe'};m.review={'trigger':'probe_result'}
        review={'observation_id':r.obs['observation_id'],'goal_id':None,'assessment':'probe_result',
                'evidence':'No change after the acknowledged click.','goal_status':'unknown',
                'causal_notes':'This instance may be inert.','next':'understand','reason':'Compare instances.',
                'next_question':'Does the highlighted instance behave differently?'}
        r._accept_stage('reconcile',Reconciliation.model_validate(review))
        review_id=m.stage_notes['reconcile'][0]
        before=deepcopy(m.notes)
        # Deliberately reproduce the original regression: the model resubmits its old hypothesis.
        r._accept_stage('understand',Understanding.model_validate(first))
        self.assertEqual({k:m.notes[k] for k in before},before)
        self.assertEqual(m.notes[outcome]['body']['changed_cell_count'],0)
        c=r._context('ground')
        self.assertEqual(c['last_review'][-1]['causal_notes'],'This instance may be inert.')
        self.assertNotIn('causal_notes',c)
        for key in m.stage_notes['understand']:
            self.assertIn(outcome,m.notes[key]['sources']);self.assertIn(review_id,m.notes[key]['sources'])

    def test_hierarchy_pagination_pin_and_remove_preserve_all_records(self):
        m=Memory(run_id='r',game_id='g')
        for i in range(9):append_note(m,'o','outcome',f'trial {i}',{'value':i},author='host')
        reader=start_reader('understand','o','Which trial matters?',m.notes)
        before=deepcopy(m.notes)
        navigate(reader,reader_options(reader,m.notes)['1'])  # experience heading
        page1=reader_options(reader,m.notes)
        self.assertEqual(len([v for v in page1.values() if v['kind']=='open_memory']),4)
        navigate(reader,page1['6'])
        key=reader_options(reader,m.notes)['1']['key']
        self.assertEqual(key,'n5')
        navigate(reader,reader_options(reader,m.notes)['1'])
        self.assertEqual(reader_context(reader,m.notes)['opened_record']['body'],{'value':4})
        navigate(reader,reader_options(reader,m.notes)['7'])
        self.assertEqual(reader['selected'],['n5'])
        self.assertNotIn('n5',[v.get('key') for v in reader_options(reader,m.notes).values()])
        navigate(reader,reader_options(reader,m.notes)['7'])  # selected set
        navigate(reader,reader_options(reader,m.notes)['1'])
        navigate(reader,reader_options(reader,m.notes)['7'])  # remove
        self.assertEqual(reader['selected'],[]);self.assertEqual(m.notes,before)

    def test_kept_records_leave_the_candidate_list_but_remain_reviewable(self):
        m=Memory(run_id='r',game_id='g')
        append_note(m,'o','hypothesis','claim',{'hypothesis':'maybe'},author='understand')
        reader=start_reader('ground','o','question',m.notes)
        navigate(reader,reader_options(reader,m.notes)['1'])
        navigate(reader,reader_options(reader,m.notes)['7'])
        self.assertEqual(set(reader_options(reader,m.notes)),{'7','8'})
        navigate(reader,reader_options(reader,m.notes)['7'])
        navigate(reader,reader_options(reader,m.notes)['1'])
        navigate(reader,reader_options(reader,m.notes)['7'])
        navigate(reader,reader_options(reader,m.notes)['5'])
        self.assertEqual(reader_options(reader,m.notes)['1']['key'],'n1')

    def test_small_library_skips_heading_and_limits_selected_body_size(self):
        m=Memory(run_id='r',game_id='g')
        append_note(m,'o','hypothesis','small',{'text':'small'},author='understand')
        append_note(m,'o','hypothesis','oversize',{'text':'x'*10000},author='understand')
        reader=start_reader('ground','o','question',m.notes)
        options=reader_options(reader,m.notes)
        self.assertEqual(options['1']['key'],'n2')
        navigate(reader,options['1'])
        self.assertNotIn('7',reader_options(reader,m.notes))
        self.assertIn('8',reader_options(reader,m.notes))

    def test_fast_reader_hands_only_selected_records_to_deliberation(self):
        r=self.runtime();m=r.memory
        for i in range(6):r._write_note('hypothesis',f'candidate {i}',{'hypothesis':f'claim {i}'},author='understand')
        r.outcome={'action':{'action':'ACTION1'},'changed_cell_count':0,'acknowledged':True}
        m.handoff_question='Which other group should be compared?'
        responses=iter(['1','2','7','8']);requests=[]
        def respond(model,payload):
            requests.append(payload);c=context(payload)
            self.assertEqual(c['work'],'read_memory');self.assertEqual(payload['max_tokens'],1)
            self.assertFalse(any(p.get('type')=='image_url' for p in payload['messages'][-1]['content']))
            self.assertEqual(c['last_result']['changed_cell_count'],0)
            self.assertEqual(c['question'],m.handoff_question)
            return token(next(responses))
        with patch.object(LocalVisionLlm,'_complete',respond):r.loop.run_until_complete(r._read_memory('understand'))
        self.assertEqual(len(requests),4)
        self.assertEqual(m.memory_brief['selected'],['n5'])
        c=r._context('understand')
        self.assertEqual([n['id'] for n in c['memory_brief']['records']],['n5'])
        self.assertNotIn('previous_understanding',c);self.assertNotIn('concept_catalogue',c)
        self.assertEqual(c['last_result'],{**r.outcome,'action':{'action':'UP'}})
        self.assertIsNone(m.pending);self.assertEqual(m.revision,1)

    def test_empty_or_expired_read_budget_preserves_mandatory_input(self):
        r=self.runtime()
        with patch.object(LocalVisionLlm,'_complete',side_effect=AssertionError('no optional reads')):
            r.loop.run_until_complete(r._read_memory('understand'))
        self.assertEqual(r.memory.memory_brief['end_reason'],'no_optional_records')
        r._write_note('question','old question',{'question':'old?'},author='understand')
        r.outcome={'changed_cell_count':0}
        with patch('agent.cognition.notebook.READ_SECONDS',0),patch.object(LocalVisionLlm,'_complete',side_effect=AssertionError('no time')):
            r.loop.run_until_complete(r._read_memory('understand'))
        self.assertEqual(r.memory.memory_brief['end_reason'],'read_time_budget')
        self.assertEqual(r._context('understand')['last_result'],r.outcome)
        self.assertGreater(r.time_left(),0);self.assertIsNone(r.result)

    def test_invalid_reader_output_keeps_partial_selection_and_does_not_stop_game(self):
        r=self.runtime()
        r._write_note('hypothesis','claim',{'hypothesis':'maybe'},author='understand')
        replies=iter(['1','7','bad'])
        with patch.object(LocalVisionLlm,'_complete',lambda m,p:token(next(replies))):
            r.loop.run_until_complete(r._read_memory('understand'))
        self.assertEqual(r.memory.memory_brief['selected'],['n1'])
        self.assertEqual(r.memory.memory_brief['end_reason'],'unreadable_selection')
        self.assertEqual(r.memory.phase,'understand');self.assertIsNone(r.result)

    def test_reading_restores_the_selected_procedure_version_for_reuse(self):
        r=self.runtime();s=skill();r.memory.skills['move']=s
        r._write_note('procedure','Move the actor',s,author='ground')
        self.assertEqual(r._retained_skills(),{})
        replies=iter(['1','7','8'])
        with patch.object(LocalVisionLlm,'_complete',lambda m,p:token(next(replies))):
            r.loop.run_until_complete(r._read_memory('ground'))
        self.assertEqual(r._retained_skills(),{'move':s})
        schema=StageTool(r,'ground')._get_declaration().parameters_json_schema
        self.assertEqual(schema['$defs']['GroundedPlan']['properties']['reuse']['items']['enum'],['move'])
        value=grounding(r._context('ground'))
        value['plan'].update(intent='probe',question='Does it move?',skills=[],reuse=['move'])
        r.validate_stage('ground',Grounding.model_validate(value))
        r.memory.skills['move']=skill(x=2)
        self.assertEqual(r._retained_skills(),{'move':s})
        r.memory.skills.clear()  # evicted cache entries remain retrievable
        r._accept_stage('ground',Grounding.model_validate(value))
        self.assertEqual(r.memory.skills['move'],s)

    def test_rerouting_preserves_the_question_and_never_executes_an_ignored_plan(self):
        r=self.runtime();v=grounding(r._context('ground'))
        v.update(next='understand',next_question='Which repeated group differs visually?')
        r._accept_stage('ground',Grounding.model_validate(v))
        self.assertEqual(r.memory.handoff_question,v['next_question'])
        self.assertIsNone(r.memory.plan)
        self.assertEqual(r.memory.phase,'understand')
        self.assertEqual(list(r.memory.notes.values())[-1]['body']['question'],v['next_question'])
        v['next_question']='';v['plan']['question']='Does the second group respond?'
        r._accept_stage('ground',Grounding.model_validate(v))
        self.assertEqual(r.memory.handoff_question,'Does the second group respond?')

    def test_actual_driver_path_runs_reader_before_reinterpretation(self):
        r=CognitiveRuntime('test','local/qwen3-vl-4b-instruct');self.addCleanup(r.close)
        works=[]
        def respond(model,payload):
            c=context(payload);works.append(c['work'])
            if c['work']=='ground':
                v=grounding(c);v['plan'].update(intent='probe',question='Does this control affect the actor?')
                return call('submit_grounding',v)
            if c['work']=='reconcile':
                from runtime_helpers import reconciliation
                v=reconciliation(c);v['next']='understand'
                return call('submit_reconciliation',v)
            if c['work']=='understand' and r.obs['step']==1:
                self.assertEqual(c['last_result']['changed_cell_count'],0)
                self.assertIn('not yet',c['last_review'][-1]['causal_notes'])
                self.assertTrue(c['memory_brief']['for_work']=='understand')
            return answer(model,payload)
        with patch.object(LocalVisionLlm,'_complete',respond):
            r.decide(obs());ack(r);result=r.decide(obs(1))
        self.assertEqual(result['status'],'action')
        index=works.index('reconcile')
        self.assertEqual(works[index:index+3],['reconcile','read_memory','understand'])
        self.assertEqual(sum(n['kind']=='outcome' for n in r.memory.notes.values()),1)
