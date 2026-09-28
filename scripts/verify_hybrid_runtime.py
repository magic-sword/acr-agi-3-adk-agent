"""Replay frozen frames through CognitiveRuntime's actual observation hook.

Gold is used only after runtime measurement. --sam recorded isolates the
integration from detector variance; --sam live measures actual initial SAM.
No game actions or remote writes are performed.
"""
import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agent.cognition.workflow import CognitiveRuntime
from agent.cognition.perception import context_record, questions, QUESTION_INSTRUCTION, QUESTION_OPTIONS
from agent.cognition.sam_proposals import default_proposer
from agent.fast_choice import choose
from agent.local_vlm import LocalVisionLlm
from scripts.audit_sam_proposals import independent_count
from scripts.audit_temporal_proposals import eligible_integer, patch_signature
from scripts.temporal_proposal_tracker import native_boxes


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def half_open(candidate):
    a,b,c,d = candidate['bbox']
    return [a,b,c+1,d+1]


class RecordedProposer:
    def __init__(self, result):
        self.result = result
        self.calls = 0

    def propose(self, grid):
        self.calls += 1
        return dict(boxes=native_boxes(self.result['boxes']), seconds=0,
                    recorded_detector_seconds=self.result['seconds'], model='recorded_sam_vit_b')


def replay(args):
    source = args.source
    out = args.out
    out.mkdir(parents=True, exist_ok=False)
    sequences = json.loads((source/'sequences.json').read_text())
    baseline = {(r['sequence'],r['frame']): r for r in
                map(json.loads,(source/'results.jsonl').read_text().splitlines())
                if r['arm']=='sam_initial' and r['repeat']==0}
    detections = {r['sequence']: r for r in
                  map(json.loads,(source/'sam-results.jsonl').read_text().splitlines()) if r['frame']==0}
    code = list(Path('agent/cognition').glob('*.py'))+[Path(__file__)]
    save(out/'manifest.json', dict(sam=args.sam, source=str(source),
        frames_sha256=sha(source/'sequences.json'), detections_sha256=sha(source/'sam-results.jsonl'),
        code_sha256={str(p):sha(p) for p in code},
        checkpoint_sha256=sha(default_proposer().checkpoint) if args.sam=='live' else None,
        timing='Actual runtime _receive including initial cold SAM when live; logs and evidence storage included. No Qwen or gameplay. Recorded mode never charges saved SAM inference times.'))
    results = []
    with (out/'measurements.jsonl').open('x') as stream:
        for sequence in sequences:
            proposer = RecordedProposer(detections[sequence['name']]) if args.sam=='recorded' else default_proposer()
            runtime = CognitiveRuntime(sequence['name'], proposal_mode='sam_initial', sam_proposer=proposer,
                                       seconds=600, decision_seconds=120, log_dir=out/'logs')
            summaries = []
            motion_gold, motion_found = set(), set()
            previous = None
            calls = 0
            exact_links = 0
            try:
                for i, frame in enumerate(sequence['frames']):
                    observation = dict(game_id=sequence['name'], step=i, grid=frame['grid'],
                        state='NOT_FINISHED', levels_completed=0, available_actions=['ACTION1'], remaining_actions=100-i)
                    start = time.perf_counter()
                    assert runtime._receive(observation)
                    seconds = time.perf_counter()-start
                    record = runtime.perception
                    assert record['sam']['status']=='ready', record['sam']
                    calls += record['sam']['called_this_frame']
                    boxes = [half_open(c) for c in record['candidates']]
                    if args.sam=='recorded':
                        assert {tuple(b) for b in boxes}=={tuple(t['box']) for t in baseline[sequence['name'],i]['tracks']}
                    inventory = context_record(record, inventory=True)
                    assert {c['id'] for c in record['candidates']}=={r[0] for r in inventory['candidate_index']}
                    found = independent_count([[v/64*1000 for v in b] for b in boxes], frame['gold'], 1, 9)
                    if i:
                        before = sequence['frames'][i-1]
                        bg = {g['id']:g for g in before['gold']}; ag = {g['id']:g for g in frame['gold']}
                        movement = {}
                        for ident in bg.keys() & ag.keys():
                            delta = tuple(ag[ident]['bbox'][k]-bg[ident]['bbox'][k] for k in (0,1))
                            if delta != (0,0) and patch_signature(before,bg[ident])==patch_signature(frame,ag[ident]):
                                movement[ident] = delta; motion_gold.add((i,ident))
                        for change in record['changes']:
                            ob, oa = change['before'], change['after']
                            assert ob['track_id']==oa['track_id'] and ob['id']!=oa['id']
                            x,y,r,b = half_open(ob); xx,yy,rr,bb = half_open(oa)
                            assert [row[x:r] for row in before['grid'][y:b]]==[row[xx:rr] for row in frame['grid'][yy:bb]]
                            assert change['delta_xy']==[xx-x, yy-y]
                            exact_links += 1
                            old = eligible_integer(half_open(ob),before['gold']); new = eligible_integer(half_open(oa),frame['gold'])
                            if len(old)==1 and old==new:
                                ident = next(iter(old))
                                if movement.get(ident)==tuple(change['delta_xy']): motion_found.add((i,ident))
                        outcome = dict(acknowledged=True, prediction='Observe the selected region movement.',decision_id=f'd{i}')
                        queue, deferred = questions(record,outcome,None)
                        assert len(queue)+deferred==len(record['changes'])
                        for q in queue:
                            assert q['change']['after']['id'] in {c['id'] for c in record['candidates']}
                            assert q['before_observation_id']==previous['observation_id']
                    row = dict(sequence=sequence['name'], kind=sequence['kind'], frame=i,
                               recall=found, gold=len(frame['gold']), sam_called=record['sam']['called_this_frame'],
                               seconds=seconds, measurement_seconds=record['seconds'],
                               candidate_count=len(boxes), inventory_bytes=len(json.dumps(inventory)),
                               changes=len(record['changes']))
                    summaries.append(row)
                    stream.write(json.dumps(dict(**row, measurement=record))+'\n');stream.flush()
                    previous = record
                assert calls==1
                result = dict(sequence=sequence['name'],kind=sequence['kind'],frames=len(summaries),
                    recall=sum(r['recall'] for r in summaries),gold=sum(r['gold'] for r in summaries),
                    motion_correct=len(motion_found),motion_gold=len(motion_gold),exact_motion_links=exact_links,
                    sam_calls=calls,initial_seconds=summaries[0]['seconds'],
                    update_seconds=sum(r['seconds'] for r in summaries[1:]),
                    total_seconds=sum(r['seconds'] for r in summaries),
                    max_inventory_bytes=max(r['inventory_bytes'] for r in summaries))
                results.append(result)
                print(json.dumps(result),flush=True)
            finally:
                runtime.close()
    save(out/'summary.json',results)


async def qwen_smoke(path, out):
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    measurement = next(r['measurement'] for r in rows if r['sequence']=='rigid_multicolor' and r['frame']==1)
    change = next(c for c in measurement['changes'] if 'sam' in c['after']['sources'])
    expected = f"The selected region moves by {change['delta_xy']} grid pixels."
    origin = change['before']
    targets = [dict(appearance='The region with the supplied original pixel pattern.',candidate_refs=[origin['id']])]
    measurement['target_correspondence'] = [dict(target_index=0, candidate_refs=[origin['id']],
        current_candidate_refs=[change['after']['id']], track_ids=[origin['track_id']], missing_track_ids=[], status='tracked')]
    measurement['changes'] = [change]
    queue,_ = questions(measurement, dict(acknowledged=True,prediction=expected,decision_id='smoke'),dict(targets=targets))
    model = LocalVisionLlm(model='qwen3-vl-4b-instruct', api_base=os.getenv('VLM_API_BASE','http://vlm:8080/v1'),max_output_tokens=1)
    records = []
    for kind in ('matching','opposite','unresolved'):
        q = json.loads(json.dumps(queue[0]))
        if kind=='opposite': q['expected_effect']=f"The selected region moves by {[-v for v in change['delta_xy']]} grid pixels."
        if kind=='unresolved':
            q['targets']=[dict(appearance='An unidentified object',candidate_refs=[])]
            q['target_correspondence']=[]
            q['expected_effect']='The unidentified object reaches its destination.'
        payloads=[]
        def observe(payload):
            payloads.append(payload)
            return hashlib.sha256(json.dumps(payload).encode()).hexdigest()
        result=await choose(model,instruction=QUESTION_INSTRUCTION,
            parts=[dict(type='text',text=json.dumps(dict(work='answer_question',**q)))],
            options=QUESTION_OPTIONS,observe_request=observe,timeout=45)
        records.append(dict(kind=kind,question=q,payload=payloads,result=result))
        assert result['schema_valid'],result
    save(out,dict(scope='Three semantic interface smoke checks, not an accuracy benchmark.',records=records))
    print(json.dumps([dict(kind=r['kind'],label=r['result']['label'],seconds=r['result']['seconds']) for r in records]))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sam',choices=['recorded','live'],default='recorded')
    parser.add_argument('--source',type=Path,default=Path('outputs/temporal-proposals-20260928'))
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--qwen-input',type=Path)
    args=parser.parse_args()
    if args.qwen_input: asyncio.run(qwen_smoke(args.qwen_input,args.out))
    else: replay(args)
