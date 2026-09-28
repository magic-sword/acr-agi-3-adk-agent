"""Measure resident SAM + existing Qwen on one GPU; never starts/stops Qwen.

Small representative inputs, not a concurrency or maximum-context stress test.
Uses the runtime SAM provider and alternates its calls with image requests to
the already-running local Qwen service. Outputs include sampled whole-GPU usage
and PyTorch allocator peaks, which measure different things.
"""
import argparse
import base64
import io
import json
from pathlib import Path
import subprocess
import sys
import threading
import time
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agent.cognition.sam_proposals import default_proposer


def http(base, path, data=None):
    request = urllib.request.Request(base+path,
        data=json.dumps(data).encode() if data is not None else None,
        headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def gpu_state():
    output = subprocess.check_output(['nvidia-smi',
        '--query-gpu=uuid,name,memory.total,memory.used,memory.free',
        '--format=csv,noheader,nounits'], text=True)
    rows=[]
    for line in output.splitlines():
        uuid,name,total,used,free=[s.strip() for s in line.split(',')]
        rows.append(dict(uuid=uuid,name=name,total_mib=int(total),used_mib=int(used),free_mib=int(free)))
    return dict(monotonic=time.monotonic(),gpus=rows)


def run(args):
    out=args.out
    out.mkdir(parents=True,exist_ok=False)
    baseline=gpu_state()
    props=http(args.base,'/props')
    (out/'qwen-props-before.json').write_text(json.dumps(props,indent=2))
    import torch
    from agent.rendering import frame_image
    from agent.cognition.hybrid_perception import HybridPerception
    sequences=json.loads(Path('outputs/temporal-proposals-20260928/sequences.json').read_text())
    sequences=[next(s for s in sequences if s['name']==name) for name in ['ls20','vc33','ft09']]
    provider=default_proposer()
    assert provider.generator is None, 'fresh process required to measure cold load'
    loads=[]
    original_load=provider._load
    def measured_load():
        if provider.generator is None:
            start=time.perf_counter()
            original_load()
            torch.cuda.synchronize()
            loads.append(time.perf_counter()-start)
        else:
            original_load()
    provider._load=measured_load
    stop=threading.Event();samples=[]
    def sample():
        with (out/'gpu-samples.jsonl').open('x') as stream:
            while not stop.is_set():
                row=gpu_state();samples.append(row)
                stream.write(json.dumps(row)+'\n');stream.flush()
                stop.wait(.1)
    sampler=threading.Thread(target=sample,daemon=True);sampler.start()
    rounds=[];pointers=[]
    try:
        for sequence in sequences:
            grid=sequence['frames'][0]['grid']
            # Separate game runtimes use the same cached provider in this process.
            engine=HybridPerception()
            assert engine.proposer is provider
            torch.cuda.reset_peak_memory_stats()
            start=time.perf_counter()
            measurement=engine.measure(None,grid,after_id=sequence['name'])
            sam_seconds=time.perf_counter()-start
            assert measurement['sam']['status']=='ready',measurement['sam']
            model=provider.generator.predictor.model
            pointer=next(model.parameters()).data_ptr();pointers.append(pointer)
            sam_gpu=gpu_state()
            image=frame_image(grid).resize((384,384))
            buffer=io.BytesIO();image.save(buffer,format='PNG')
            payload=dict(model='qwen3-vl-4b-instruct',temperature=0,max_tokens=1,
                grammar='root ::= "1"',messages=[
                    dict(role='system',content='Inspect the game image. Reply with the single digit 1.'),
                    dict(role='user',content=[dict(type='image_url',image_url=dict(
                        url='data:image/png;base64,'+base64.b64encode(buffer.getvalue()).decode())),
                        dict(type='text',text='Acknowledge this image with 1.')])])
            start=time.perf_counter();response=http(args.base,'/v1/chat/completions',payload)
            qwen_seconds=time.perf_counter()-start
            assert response['choices'][0]['message']['content']=='1',response
            assert next(model.parameters()).data_ptr()==pointer
            round_result=dict(sequence=sequence['name'],sam_seconds=sam_seconds,
                qwen_seconds=qwen_seconds,sam_loads_so_far=len(loads),sam_weight_pointer=pointer,
                sam_allocated_mib=torch.cuda.memory_allocated()/2**20,
                sam_reserved_mib=torch.cuda.memory_reserved()/2**20,
                sam_peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                sam_peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,
                gpu_after_sam=sam_gpu,gpu_after_qwen=gpu_state(),qwen_response=response)
            rounds.append(round_result)
            print(json.dumps({k:round_result[k] for k in ['sequence','sam_seconds','qwen_seconds',
                'sam_loads_so_far','sam_allocated_mib','sam_reserved_mib']}),flush=True)
    finally:
        stop.set();sampler.join(timeout=10)
    assert len(loads)==1 and len(set(pointers))==1
    result=dict(status='passed',gpu_baseline=baseline,
        scope='Both models resident; three alternating SAM and Qwen image inferences. No simultaneous kernel/parallel-game or maximum-context stress test. Not a Kaggle run.',
        sam_weight_loads=len(loads),sam_weight_load_seconds=loads,
        sam_weight_pointer_unchanged=len(set(pointers))==1,
        rounds=rounds,samples=len(samples),
        sampled_peak_total_used_mib=max(g['used_mib'] for s in samples for g in s['gpus']),
        sampled_min_free_mib=min(g['free_mib'] for s in samples for g in s['gpus']))
    (out/'summary.json').write_text(json.dumps(result,indent=2))
    (out/'qwen-props-after.json').write_text(json.dumps(http(args.base,'/props'),indent=2))
    print(json.dumps({k:result[k] for k in ['status','sam_weight_loads','sam_weight_load_seconds',
        'sam_weight_pointer_unchanged','sampled_peak_total_used_mib','sampled_min_free_mib']}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--base',default='http://127.0.0.1:8080')
    run(parser.parse_args())
