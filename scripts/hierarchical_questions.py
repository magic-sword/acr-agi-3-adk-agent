"""Question routing over measured candidate states; uncertainty never means no."""
import json

KINDS = ['position','appearance','size']
LABEL_FLAGS = {
    'A': (False,False,False), 'B': (True,False,False),
    'C': (False,True,False), 'D': (False,False,True),
    'E': (True,True,False), 'F': (True,False,True),
    'G': (False,True,True), 'H': (True,True,True), 'X': None,
}
DETAIL_OPTIONS = {
    'A':'no change', 'B':'position only', 'C':'appearance only', 'D':'size only',
    'E':'position and appearance', 'F':'position and size',
    'G':'appearance and size', 'H':'position, appearance and size',
    'X':'cannot determine',
}
SYSTEM = ('Answer one observation question from these measured candidate states. '
          'Position means top-left x,y. Appearance means the complete local shape/color pattern; '
          'a change in pattern dimensions counts as an appearance change too. Size means width,height. '
          'These properties can change together. Use X if correspondence is unresolved or either state is missing. '
          'Output only the option letter.')


def measured_flags(context):
    b,a=context['before'],context['after']
    if context['correspondence']=='unresolved' or b is None or a is None:return None
    return (b['bbox'][:2]!=a['bbox'][:2],b['pattern']!=a['pattern'],b['size']!=a['size'])


def encode_flags(flags):
    if flags is None:return 'X'
    return next(k for k,v in LABEL_FLAGS.items() if v==tuple(flags))


def flat_result(answers):
    if any(answers.get(k) not in ['A','B'] for k in KINDS):return 'X'
    return encode_flags(tuple(answers[k]=='B' for k in KINDS))


def needs_detail(gate_answer):
    # Invalid output also goes to detail/review; only explicit No skips it.
    return gate_answer!='A'


def make_request(context,stage):
    if stage=='gate':
        question='Did ANY of position, appearance or size change between BEFORE and AFTER?'
        options={'A':'no change in any property','B':'one or more properties changed','X':'cannot determine'}
    elif stage=='detail':
        question='Which set of properties changed between BEFORE and AFTER? Select the complete combination.'
        options=DETAIL_OPTIONS
    else:
        if stage not in KINDS:raise ValueError('unknown stage')
        question=f'Did {stage} change between BEFORE and AFTER?'
        options={'A':'unchanged','B':'changed','X':'cannot determine'}
    prompt=json.dumps(context,separators=(',',':'))+'\n'+question+'\n'+'\n'.join(k+'. '+v for k,v in options.items())
    return dict(model='qwen3-vl-4b-instruct',messages=[dict(role='system',content=SYSTEM),dict(role='user',content=prompt)],
        temperature=0,max_tokens=1,grammar='root ::= '+' | '.join(json.dumps(k) for k in options),stream=False,cache_prompt=False)
