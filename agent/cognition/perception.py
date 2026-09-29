"""Program measurements and compact current observations; no model routing."""
from copy import deepcopy
import time
from .geometry import extract, compare
from .region_masks import encode

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
                colors=[COLORS[int(c)] for c in obj['colors']],pattern=obj['pattern'],
                mask_runs=encode(support(obj)))

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

def without_masks(value):
    """Native supports are audit/execution data, not repeated model input."""
    if isinstance(value, list):
        return [without_masks(v) for v in value]
    if isinstance(value, dict):
        return {k: without_masks(v) for k, v in value.items() if k != 'mask_runs'}
    return value

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
    return without_masks(r)
