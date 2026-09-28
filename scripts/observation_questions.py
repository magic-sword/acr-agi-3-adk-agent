"""Generate atomic observation questions from measured candidates, without gold.

This is an experimental adapter, not a production detector. Candidate grouping
and correspondence remain hypotheses. Pixel coverage is audited independently
so an omitted large region cannot silently disappear from the question queue.
"""
import json
from scripts.instance_state import extract, compare

OPTIONS = {'A': 'unchanged', 'B': 'changed', 'X': 'cannot determine'}
PROPERTIES = {
    'position': 'Did the position of this candidate change between BEFORE and AFTER?',
    'appearance': 'Did the local shape or color pattern of this candidate change between BEFORE and AFTER?',
    'size': 'Did the width or height of this candidate change between BEFORE and AFTER?',
}


def grammar(letters):
    letters = list(letters)
    if not letters or len(set(letters)) != len(letters) or any(len(s) != 1 or s not in 'ABCDEX' for s in letters):
        raise ValueError('Expected unique supported single-letter labels')
    return 'root ::= ' + ' | '.join(json.dumps(s) for s in letters)


def support(obj):
    """Recover exact measured support, not the whole bounding rectangle."""
    if 'children' in obj:
        return {(x,y) for c in obj['children'] for y in range(c['bbox'][1],c['bbox'][3]+1) for x in range(c['bbox'][0],c['bbox'][2]+1)}
    pattern = obj['pattern']
    if isinstance(pattern,dict):
        pattern = [[v for value,count in runs for v in [value]*count] for runs in pattern['row_runs']]
    x0,y0 = obj['bbox'][:2]
    return {(x0+x,y0+y) for y,row in enumerate(pattern) for x,v in enumerate(row) if v is not None}


def observation(obj):
    if obj is None:return None
    return {k:obj[k] for k in ['id','class_name','bbox','size','pattern']}


def build_questions(before_pixels, after_pixels, action):
    if len(before_pixels)!=len(after_pixels) or any(len(b)!=len(a) for b,a in zip(before_pixels,after_pixels)):
        raise ValueError('Frame dimensions must agree')
    before=extract(before_pixels,'b');after=extract(after_pixels,'a');comparison=compare(before,after)
    old={o['id']:o for o in before['instances']};new={o['id']:o for o in after['instances']}
    matches={a:b for a,b,_ in comparison['correspondences']}
    matched_after=set(matches.values())
    ambiguities={e['before_id']:e.get('candidates',[]) for e in comparison['changes'] if e['kind']=='uncertain'}
    subjects=[(o,new.get(matches.get(o['id'])),ambiguities.get(o['id'],[])) for o in before['instances']]
    subjects += [(None,o,[]) for o in after['instances'] if o['id'] not in matched_after]
    questions=[]
    for b,a,alternatives in subjects:
        ref=(b or a)['id']
        for kind,text in PROPERTIES.items():
            questions.append(dict(id=f'{ref}:{kind}',subject_id=ref,property=kind,question=text,options=OPTIONS.copy(),
                context=dict(recorded_action=action,coordinate_convention='original pixels; x rightward, y downward',
                    before=observation(b),after=observation(a),
                    correspondence='measured_candidate_match' if b is not None and a is not None else 'unresolved',
                    alternative_after_candidates=[observation(new[i]) for i in alternatives],
                    caution='Grouping is hypothetical; local IDs alone do not prove identity. Missing correspondence is not evidence of no change.')))
    changed={(x,y) for y,(br,ar) in enumerate(zip(before_pixels,after_pixels)) for x,(b,a) in enumerate(zip(br,ar)) if b!=a}
    covered=set().union(*(support(o) for o in list(old.values())+list(new.values())))
    residual=sorted(changed-covered,key=lambda xy:(xy[1],xy[0]))
    unresolved=sum(b is None or a is None for b,a,_ in subjects)
    return dict(questions=questions,coverage=dict(before_candidates=len(old),after_candidates=len(new),subjects=len(subjects),
        properties_per_subject=len(PROPERTIES),changed_pixels=len(changed),uncovered_changed_pixels=len(residual),uncovered_positions=residual,
        unresolved_subjects=unresolved,requires_review=bool(residual or unresolved),
        limitation='Complete over extracted candidates only. Zero residual pixels does not prove correct object grouping, identity, or semantic coverage.'),
        independent_states=dict(before=before,after=after))


def request_for(question):
    content=json.dumps(question['context'],separators=(',',':'))+'\n'+question['question']+'\n'+'\n'.join(k+'. '+v for k,v in question['options'].items())
    return dict(model='qwen3-vl-4b-instruct',messages=[
        dict(role='system',content='Answer this one observation question only. Use X when correspondence or evidence is unresolved. Output only its option letter.'),
        dict(role='user',content=content)],temperature=0,max_tokens=1,grammar=grammar(question['options']),stream=False,cache_prompt=False)
