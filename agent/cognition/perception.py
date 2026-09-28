"""Measured candidate changes and bounded semantic questions, not game rules."""
from copy import deepcopy
import json
import time
from .geometry import extract, compare

COLORS = ['white','light gray','gray','dark gray','very dark gray','black',
          'magenta','pink','red','blue','light blue','yellow','orange','maroon','green','purple']


def support(obj):
    if 'children' in obj:
        return {(x,y) for c in obj['children'] for y in range(c['bbox'][1],c['bbox'][3]+1)
                for x in range(c['bbox'][0],c['bbox'][2]+1)}
    pattern=obj['pattern']
    if isinstance(pattern,dict):
        pattern=[[v for v,n in row for _ in range(n)] for row in pattern['row_runs']]
    x0,y0=obj['bbox'][:2]
    return {(x0+x,y0+y) for y,row in enumerate(pattern) for x,v in enumerate(row) if v is not None}


def brief(obj):
    if obj is None:return None
    return dict(id=obj['id'],shape=obj['class_name'],bbox=obj['bbox'],size=obj['size'],
                colors=[COLORS[int(c)] for c in obj['colors']],pattern=obj['pattern'])


def measure(before, after, *, before_id=None, after_id, action=None, boundary=False):
    start=time.perf_counter()
    result=dict(before_observation_id=before_id,observation_id=after_id,action=deepcopy(action),
                status='unavailable',candidates=[],changes=[],unchanged=[],unresolved=[],
                uncovered_changed_pixels=None,changed_pixels=None,requires_review=True,
                identity_note='Geometric correspondence hypotheses, not physical identity or causal proof.')
    if not after:
        result['seconds']=time.perf_counter()-start;return result
    new=extract(after,'a');result['candidates']=[brief(o) for o in new['instances']]
    result['extraction_limited']=bool(new.get('extraction_limited'))
    if boundary or not before or len(before)!=len(after) or any(len(x)!=len(y) for x,y in zip(before,after)):
        result.update(status='initial' if not before else 'boundary',requires_review=True)
    else:
        old=extract(before,'b');comparison=compare(old,new)
        result['extraction_limited'] |= bool(old.get('extraction_limited'))
        old_by={o['id']:o for o in old['instances']};new_by={o['id']:o for o in new['instances']}
        matched_old=set();matched_new=set()
        for b,a,reason in comparison['correspondences']:
            ob,oa=old_by[b],new_by[a];matched_old.add(b);matched_new.add(a)
            flags=dict(position=ob['bbox'][:2]!=oa['bbox'][:2],
                       appearance=ob['appearance_signature']!=oa['appearance_signature'],size=ob['size']!=oa['size'])
            record=dict(before=brief(ob),after=brief(oa),changed=flags,correspondence=reason)
            if any(flags.values()):
                record['delta_xy']=[oa['bbox'][i]-ob['bbox'][i] for i in (0,1)]
                result['changes'].append(record)
            else:result['unchanged'].append(dict(before_id=b,after_id=a))
        result['unresolved']=[dict(side='before',candidate=brief(o)) for k,o in old_by.items() if k not in matched_old]
        result['unresolved'] += [dict(side='after',candidate=brief(o)) for k,o in new_by.items() if k not in matched_new]
        changed={(x,y) for y,(br,ar) in enumerate(zip(before,after)) for x,(b,a) in enumerate(zip(br,ar)) if b!=a}
        covered=set().union(*(support(o) for o in old['instances']+new['instances']))
        residual=sorted(changed-covered,key=lambda p:(p[1],p[0]))
        result.update(status='measured',changed_pixels=len(changed),uncovered_changed_pixels=len(residual),
                      uncovered_positions=residual,relations=comparison['same_pattern_after'],
                      requires_review=bool(residual or result['unresolved'] or result['extraction_limited']))
    result['seconds']=time.perf_counter()-start
    return result


def context_record(record, *, inventory=False):
    """No silent truncation: keep all facts in logs, flag bounded model summaries."""
    r={k:deepcopy(record[k]) for k in ['observation_id','before_observation_id','status','action',
        'changed_pixels','uncovered_changed_pixels','requires_review','identity_note']}
    for key in ['changes','unresolved']:
        r[key]=deepcopy(record[key][:12]);r[key+'_omitted']=max(0,len(record[key])-12)
    r['unchanged_count']=len(record['unchanged'])
    if inventory:
        if record.get('proposal_mode') == 'sam_initial':
            # All region IDs remain visible, including lower-ranked SAM masks.
            # Pixel patterns stay in the evidence log instead of flooding the LLM.
            r['candidate_index_columns']=['id','bbox_inclusive','color_ids','sources','track_id']
            r['candidate_index']=[[c['id'],c['bbox'],c['color_ids'],c['sources'],c['track_id']]
                                  for c in record['candidates']]
            r['color_names']=COLORS
            r['candidates_omitted']=0
            r['candidate_note']='Regions may overlap or include background; track IDs are correspondence hypotheses.'
        else:
            r['candidates']=deepcopy(record['candidates'][:24])
            r['candidates_omitted']=max(0,len(record['candidates'])-24)
    for key in ['proposal_mode','sam','proposal_coverage_incomplete','target_correspondence']:
        if key in record:r[key]=deepcopy(record[key])
    if 'relations' in record:r['same_pattern_pairs']=record['relations'][:24]
    return r


QUESTION_OPTIONS = {
    '1':dict(kind='supports',meaning='This measured change supports the stated expected effect on the intended target.'),
    '2':dict(kind='contradicts',meaning='This measured change contradicts the stated expected effect.'),
    '3':dict(kind='unrelated',meaning='This change is clearly unrelated to that expected effect.'),
    '8':dict(kind='unknown',meaning='Target correspondence or the relation to the expected effect is unclear.'),
}
QUESTION_INSTRUCTION = ('Classify the relation of ONE measured candidate change to the stated expected effect. '
    'The measured change is supplied evidence, not something to redetect. Target descriptions and roles are hypotheses. '
    'Do not infer causation or goal completion from temporal proximity. If target identity or the expected effect '
    'cannot be grounded, choose 8. Output exactly one offered digit.')


def questions(record, outcome, understanding, *, limit=4):
    if record['status']!='measured' or not outcome or not outcome.get('acknowledged') or not outcome.get('prediction'):
        return [],0
    targets=(understanding or {}).get('targets',[])
    all_questions=[dict(id=f"{record['observation_id']}:meaning:{i}",observation_id=record['observation_id'],
        before_observation_id=record['before_observation_id'],decision_id=outcome['decision_id'],
        invocation_id=(outcome.get('skill') or {}).get('invocation_id'),
        action=record['action'],expected_effect=outcome['prediction'],
        targets=targets,change=deepcopy(change),
        target_correspondence=deepcopy(record.get('target_correspondence',[])),
        question='How does this measured change relate to the expected effect?',choices=deepcopy(QUESTION_OPTIONS))
        for i,change in enumerate(record['changes'])]
    return all_questions[:limit],max(0,len(all_questions)-limit)


class PerceptionRuntime:
    def _measure_observation(self, boundary):
        engine=getattr(self,'proposal_perception',None)
        measure_frame=engine.measure if engine is not None else measure
        self.perception=measure_frame(self.previous.get('grid'),self.obs.get('grid'),
            before_id=self.previous.get('observation_id'),after_id=self.obs['observation_id'],
            action=(self.outcome or {}).get('action'),boundary=bool(boundary),
            **({'seconds_left':self.time_left()} if engine is not None else {}))
        self.semantic_answers=[];self.question_queue=[];self.question_deferred=0
        if self.outcome:self.outcome['measured_objects']=context_record(self.perception)
        self._record('artifacts','objects_measured',measurement=self.perception)

    def _route_perception(self):
        """Called after normal observation routing; probes still end in reconciliation."""
        m=self.memory
        if not self.outcome or not self.outcome.get('acknowledged') or self.outcome.get('boundary'):
            return
        if m.phase not in ('execute_step','reconcile'):return
        self.question_queue,self.question_deferred=questions(self.perception,self.outcome,m.understanding)
        if self.perception['requires_review'] and m.phase=='execute_step':
            self._queue_review('perception_uncertain','Object correspondence or changed pixels need review.',
                               'objects_uncertain')
        if self.question_queue:
            self.question_return=m.phase
            self._machine_transition('questions_for_review' if m.phase=='reconcile' else 'questions_for_execution')
            m.phase='answer_question'
        self._snapshot()

    async def _answer_semantic_question(self):
        from agent.fast_choice import choose
        q=self.question_queue.pop(0);self.work='answer_question'
        if q['observation_id']!=self.obs['observation_id']:
            raise ValueError('stale semantic question')
        context=dict(work=self.work,**q)
        self.calls+=1;self.memory.model_calls+=1
        self._record('states','state_entered',state='DECIDE',work=self.work,input=context)
        record=await choose(self.choice_model,instruction=QUESTION_INSTRUCTION,
            parts=[dict(type='text',text=json.dumps(context,separators=(',',':')))],options=QUESTION_OPTIONS,
            observe_request=self._request_record,timeout=self.time_left())
        self.http_requests+=record['http_requests']
        self._record('model','model_decision',state='DECIDE',work=self.work,call_index=self.calls,context=context,**record)
        self._record('states','state_exited',state='DECIDE',work=self.work,output=record)
        meaning=QUESTION_OPTIONS[record['label']]['kind'] if record['schema_valid'] else 'unknown'
        answer=dict(question_id=q['id'],observation_id=q['observation_id'],decision_id=q['decision_id'],
                    interpretation=meaning,schema_valid=record['schema_valid'],author='model_hypothesis')
        self.semantic_answers.append(answer)
        self._write_note('hypothesis','Measured change versus expected effect: '+meaning,answer,author='model')
        self._record('artifacts','semantic_question_answered',question=q,answer=answer)
        if self.question_queue:
            self._machine_transition('next_semantic_question')
        elif (self.question_return=='reconcile' or self.question_deferred or
              any(a['interpretation'] in ('unknown','contradicts') for a in self.semantic_answers)):
            self._queue_review('measured_effect_review','Interpret the measured changes; answers are hypotheses, not goal completion.',
                               'questions_to_review')
        else:
            self.memory.phase='execute_step';self._machine_transition('questions_to_execution')
        self._snapshot()
