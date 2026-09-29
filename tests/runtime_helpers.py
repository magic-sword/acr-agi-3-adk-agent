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


def answer(model, payload):
    c = context(payload)
    return call('submit_next_action', dict(observation_id=c['observation_id'],
        interpretation='', question='Does UP move the actor?', action='UP',
        object_id=None, expected_effect='The actor may move upwards.'))


def ack(runtime):
    runtime.record_execution('action_dispatched')
    runtime.record_execution('action_acknowledged')
