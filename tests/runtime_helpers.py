from copy import deepcopy
import json


def obs(step=0, grid=None):
    grid = deepcopy(grid if grid is not None else [[0]*6])
    return {'game_id':'test','step':step,'observation_id':f'o{step}', 'grid':grid,
            'width':len(grid[0]),'height':len(grid),'state':'NOT_FINISHED',
            'levels_completed':0,'available_actions':['ACTION6'],'remaining_actions':30-step}


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


def skill(name='move', x=0):
    return {'name':name,'when_to_use':'The target is visible.','effect':'Move the actor to the marked target.',
        'steps':[{'purpose':'Move the actor into the target.', 'continue_when':'The actor is outside the target.',
                  'done_when':'The actor is inside the marked target.',
                  'reconsider_when':'The actor does not move as expected.',
                  'options':[{'action':{'action':'CLICK','x':x,'y':0},'expected_effect':'The actor moves into the target.'}]}]}


def understanding(c):
    return {'observation_id':c['observation_id'],
            'targets':[{'id':'actor','appearance':'Small square','x':0,'y':0},
                       {'id':'target','appearance':'Marked square','x':2,'y':0}],
            'observed':'The actor is left of the marked target.',
            'goal_hypothesis':'Enter the marked target.', 'causal_hypotheses':'Clicking may move the actor.',
            'question':'Does clicking move the actor?', 'next':'backchain'}


def backchain(c):
    return {'observation_id':c['observation_id'],'goals':[
        {'id':'exit','parent_id':None,'desired_state':'The exit admits the actor.',
         'target_ids':['actor','target'],'requires':['approach']},
        {'id':'approach','parent_id':'exit','desired_state':'The actor is inside the target.',
         'target_ids':['actor','target'],'requires':[]}],
        'selected_goal_id':'approach','rationale':'Entering the target enables testing whether it is an exit.'}


def grounding(c, x=0):
    return {'observation_id':c['observation_id'],'next':'execute','reason':'Ground the selected goal.',
        'plan':{'goal_id':(c.get('current_goal') or {}).get('id'),'intent':'achieve','question':'',
                'target_ids':['actor','target'],'baseline':'The actor is left of the target.',
                'skills':[skill(x=x)],'reuse':[]}}


def reconciliation(c):
    return {'observation_id':c['observation_id'],'goal_id':c['plan']['goal_id'],
            'assessment':'unclear','evidence':'The intended target relation is not yet confirmed.',
            'goal_status':'active','causal_notes':'Clicking has not yet established target entry.',
            'next':'ground','reason':'Specify a new concrete attempt.'}


def answer(model, payload):
    c=context(payload)
    stages={'understand':('submit_understanding',understanding),
            'backchain':('submit_backchain',backchain),'ground':('submit_grounding',grounding),
            'reconcile':('submit_reconciliation',reconciliation)}
    if c['work'] in stages:
        tool, fn=stages[c['work']];return call(tool,fn(c))
    return token('1')


def ack(runtime):
    runtime.record_execution('action_dispatched')
    runtime.record_execution('action_acknowledged')
