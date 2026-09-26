from copy import deepcopy
import json


def obs(step=0, grid=None):
    grid = deepcopy(grid if grid is not None else [[0]*6])
    return {'game_id':'test','step':step,'observation_id':f'o{step}', 'grid':grid,
            'width':len(grid[0]),'height':len(grid),'state':'NOT_FINISHED',
            'levels_completed':0,'available_actions':['ACTION1','ACTION6'],'remaining_actions':30-step}


def context(payload):
    for message in reversed(payload['messages']):
        if message['role']=='user' and isinstance(message['content'],list):
            for part in reversed(message['content']):
                if part.get('type')=='text':
                    try:
                        result=json.loads(part['text'])
                        if isinstance(result,dict) and 'work' in result:return result
                    except ValueError:pass
    raise AssertionError('missing decision context')


def call(name, args):
    return {'choices':[{'finish_reason':'tool_calls','message':{'role':'assistant','content':'',
        'tool_calls':[{'id':'test-call','type':'function','function':{'name':name,'arguments':json.dumps(args)}}]}}]}


def token(label):
    return {'choices':[{'finish_reason':'length','message':{'role':'assistant','content':label}}],
            'usage':{'prompt_tokens':100,'completion_tokens':1,'total_tokens':101},
            'timings':{'cache_n':0}}


def skill(name='move', x=None):
    return {'name':name,'when_to_use':'The target is visible.','effect':'Move the actor to the marked target.',
        'steps':[{'purpose':'Move the actor into the target.', 'continue_when':'The actor is outside the target.',
                  'done_when':'The actor is inside the marked target.',
                  'reconsider_when':'The actor does not move as expected.',
                  'options':[{'action':({'action':'UP'} if x is None else {'action':'CLICK','target_query':'The marked cell.'}),'expected_effect':'The actor moves into the target.'}]}]}


def understanding(c):
    return {'observation_id':c['observation_id'],
            'concepts':[{'name':'square','description':'Small square objects with different roles.'}],
            'targets':[{'concept':'square','appearance':'Small square','role_hypothesis':'actor','relations':'Left of the marked square'},
                       {'concept':'square','appearance':'Marked square','role_hypothesis':'destination','relations':'Right of the actor'}],
            'observed':'The actor is left of the marked target.',
            'goal_hypothesis':'Enter the marked target.', 'causal_hypotheses':'Clicking may move the actor.',
            'question':'Does clicking move the actor?', 'next':'backchain'}


def backchain(c):
    return {'observation_id':c['observation_id'],'goals':[
        {'id':'exit','parent_id':None,'desired_state':'The exit admits the actor.',
         'target_query':'The actor and the marked destination.','requires':['approach']},
        {'id':'approach','parent_id':'exit','desired_state':'The actor is inside the target.',
         'target_query':'The actor and the marked destination.','requires':[]}],
        'selected_goal_id':'approach','rationale':'Entering the target enables testing whether it is an exit.'}


def grounding(c, x=None):
    return {'observation_id':c['observation_id'],'next':'execute','reason':'Ground the selected goal.','next_question':'',
        'plan':{'goal_id':(c.get('current_goal') or {}).get('id'),'intent':'achieve','question':'',
                'target_query':'The actor and the marked destination.','baseline':'The actor is left of the target.',
                'skills':[skill(x=x)],'reuse':[]}}


def reconciliation(c):
    return {'observation_id':c['observation_id'],'goal_id':c['plan']['goal_id'],
            'assessment':'unclear','evidence':'The intended target relation is not yet confirmed.',
            'goal_status':'active','causal_notes':'Clicking has not yet established target entry.',
            'next':'ground','reason':'Specify a new concrete attempt.',
            'next_question':'Which target or condition would distinguish the unresolved effect?' }


def answer(model, payload):
    c=context(payload)
    stages={'understand':('submit_understanding',understanding),
            'backchain':('submit_backchain',backchain),'ground':('submit_grounding',grounding),
            'reconcile':('submit_reconciliation',reconciliation)}
    if c['work'] in stages:
        tool, fn=stages[c['work']];return call(tool,fn(c))
    if c['work']=='read_memory':return token('8')
    if c['work']=='aim':
        if c['cursor']['mode']=='locate':return token('1')
        if c['cursor']['x']<2:return token('4')
        return token('7')
    return token('1')


def ack(runtime):
    runtime.record_execution('action_dispatched')
    runtime.record_execution('action_acknowledged')
