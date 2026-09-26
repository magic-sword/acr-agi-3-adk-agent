"""Replay fixed real requests to measure inference overlap, not game performance.

Recorded reader requests are replayed without following new model choices. Treat
overlap as an optimistic scheduling experiment: useful future queries are known.
No game actions are sent and the production runtime is not modified.
"""
from __future__ import annotations

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import time
import urllib.request


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def restore_images(payload, directory):
    payload=deepcopy(payload)
    for message in payload['messages']:
        if not isinstance(message.get('content'),list):continue
        for part in message['content']:
            if part.get('type')!='image_url':continue
            image=part['image_url']
            if 'url' in image:continue
            path=(directory/image['path']).resolve()
            if not path.is_relative_to(directory.resolve()):raise ValueError('image outside recording directory')
            raw=path.read_bytes()
            if hashlib.sha256(raw).hexdigest()!=image['sha256']:raise ValueError('image hash mismatch')
            part['image_url']={'url':'data:'+image['mime_type']+';base64,'+base64.b64encode(raw).decode()}
            if image.get('detail') is not None:part['image_url']['detail']=image['detail']
    return payload


def read_lines(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def prepare(source, output, reads):
    cases=[]
    for directory in sorted(source.glob('*/cognition')):
        requests=[r for p in sorted(directory.glob('*.requests.jsonl')) for r in read_lines(p)]
        decisions=[r for p in sorted(directory.glob('*.model.jsonl')) for r in read_lines(p)]
        valid={r.get('request_sha256') for r in decisions if r.get('schema_valid')}
        valid.update(e.get('request_sha256') for r in decisions if r.get('schema_valid') for e in r.get('exchanges',[]))
        background=[]
        for r in requests:
            if r.get('work')=='read_memory' and r['request_sha256'] in valid:
                background.append(r)
            elif background:break
        if len(background)<reads:raise ValueError(f'{directory}: need {reads} consecutive successful reader requests')
        background=background[:reads]
        for kind,work in [('deliberation','understand'),('action','execute_step')]:
            foreground=next((r for r in requests if r['work']==work and r['request_sha256'] in valid),None)
            if foreground is None:continue
            case={'name':directory.parent.name+'-'+kind,'kind':kind,
                  'source':str(directory),'foreground':restore_images(foreground['request'],directory),
                  'background':[restore_images(r['request'],directory) for r in background],
                  'source_request_hashes':[foreground['request_sha256'],*[r['request_sha256'] for r in background]]}
            cases.append(case)
    if not cases:raise ValueError('no recorded cases')
    output.mkdir(parents=True,exist_ok=False)
    (output/'cases.json').write_text(json.dumps(cases,ensure_ascii=False))
    (output/'experiment.json').write_text(json.dumps({'source':str(source),'case_digest':digest(cases),
        'created_at':datetime.now(timezone.utc).isoformat(),'reader_calls_per_case':reads,
        'scope':'Fixed request replay; oracle future-query availability; no online prefetch hit-rate or game-score claim.'},indent=2))
    # Keep context capacity per slot fixed. These overrides are only for the experiment.
    for slots in (1,2):
        command=['--model','/models/Qwen3VL-4B-Instruct-Q4_K_M.gguf',
                 '--mmproj','/models/mmproj-Qwen3VL-4B-Instruct-Q8_0.gguf',
                 '--alias','qwen3-vl-4b-instruct','--jinja','--host','0.0.0.0','--port','8080',
                 '--n-gpu-layers','99','--ctx-size',str(16384*slots),'--parallel',str(slots)]
        (output/f'server-{slots}.json').write_text(json.dumps({'services':{'vlm':{'command':command}}},indent=2))
    print(f'Prepared {len(cases)} cases in {output}',flush=True)


def request_json(url, payload=None):
    req=urllib.request.Request(url,data=None if payload is None else json.dumps(payload).encode(),
                               headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=120) as response:return json.load(response)


def response_signature(response):
    message=deepcopy(response['choices'][0]['message'])
    for call in message.get('tool_calls',[]):
        call.pop('id',None)
        args=call['function'].get('arguments')
        if isinstance(args,str):
            try:call['function']['arguments']=json.loads(args)
            except ValueError:pass
    return digest(message)


def complete(base, payload, slot, origin, cache=True):
    payload={**payload,'id_slot':slot,'seed':123,'cache_prompt':cache}
    started=time.monotonic()
    response=request_json(base+'/chat/completions',payload)
    ended=time.monotonic()
    cached=(response.get('timings') or {}).get('cache_n',
        ((response.get('usage') or {}).get('prompt_tokens_details') or {}).get('cached_tokens',0))
    if not cache and cached:
        raise ValueError('server reused prompt KV despite cache_prompt=false; exclude this cache-off profile')
    choices=response.get('choices') or []
    if not choices:raise ValueError(f'no model response: {response}')
    if payload.get('max_tokens')==1:
        label=choices[0]['message'].get('content')
        if json.dumps(label) not in payload.get('grammar',''):raise ValueError(f'invalid choice: {label}')
    elif payload.get('tools') and not choices[0]['message'].get('tool_calls'):
        raise ValueError('expected structured foreground submission')
    return {'start':started-origin,'end':ended-origin,'seconds':ended-started,
            'slot':slot,'usage':response.get('usage'),'timings':response.get('timings'),
            'response_signature':response_signature(response),'response':response,
            'payload_digest':digest(payload)}


def measure(base, case, slots, mode, cache=True):
    origin=time.monotonic()
    def foreground():return complete(base,case['foreground'],0,origin,cache)
    def background():return [complete(base,p,slots-1,origin,cache) for p in case['background']]
    if mode=='serial':fg=foreground();bg=background()
    else:
        with ThreadPoolExecutor(max_workers=2) as pool:
            f=pool.submit(foreground);b=pool.submit(background)
            fg=f.result();bg=b.result()
    return {'case':case['name'],'kind':case['kind'],'mode':mode,'slots':slots,
            'cache_prompt':cache,'pair_seconds':max(fg['end'],bg[-1]['end']),
            'foreground_seconds':fg['seconds'],
            'background_seconds':bg[-1]['end']-bg[0]['start'],
            'background_wait_after_foreground':max(0,bg[-1]['end']-fg['end']),
            'foreground':fg,'background':bg}


def run(output, base, slots, repeats, modes, cache=True):
    cases=json.loads((output/'cases.json').read_text())
    props=request_json(base.removesuffix('/v1')+'/props')
    if props['total_slots']!=slots:raise ValueError('server slot count does not match experiment')
    capacity=props['default_generation_settings']['n_ctx']
    if capacity!=16384:raise ValueError(f'expected 16384 context per slot, got {capacity}')
    gpu=subprocess.run(['nvidia-smi','--query-gpu=name,memory.total,memory.used,utilization.gpu',
                        '--format=csv'],capture_output=True,text=True,check=True).stdout
    output_file=output/f'measurements-{slots}.jsonl'
    if output_file.exists():raise ValueError('refusing to mix measurements from separate runs')
    metadata={'server':props,'gpu':gpu,'case_digest':digest(cases),'repeats':repeats,'modes':modes,'cache_prompt':cache,
              'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (output/f'environment-{slots}.json').write_text(json.dumps(metadata,indent=2))
    with output_file.open('w') as log:
        for case in cases:
            # Fixed warmup for each mode; exclude it from statistics, retain it for audit.
            for mode in modes:
                r=measure(base,case,slots,mode,cache);r.update(warmup=True,repeat=-1)
                log.write(json.dumps(r)+'\n');log.flush()
            for repeat in range(repeats):
                # Reverse order on alternating repeats to reduce order/cache bias.
                for mode in modes if repeat%2==0 else list(reversed(modes)):
                    r=measure(base,case,slots,mode,cache);r.update(warmup=False,repeat=repeat)
                    log.write(json.dumps(r)+'\n');log.flush()
                    print(f'{case["name"]} slots={slots} {mode} #{repeat}: '
                          f'pair={r["pair_seconds"]:.3f}s foreground={r["foreground_seconds"]:.3f}s',flush=True)


def report(output):
    rows=[r for p in sorted(output.glob('measurements-*.jsonl')) for r in read_lines(p) if not r['warmup']]
    cases=json.loads((output/'cases.json').read_text())
    policies={r.get('cache_prompt',True) for r in rows}
    if len(policies)!=1:raise ValueError('cannot compare mixed cache policies')
    environments=[json.loads(p.read_text()) for p in sorted(output.glob('environment-*.json'))]
    if environments and any(e['case_digest']!=digest(cases) for e in environments):
        raise ValueError('recorded input digest changed between runs')
    if len({(e['server']['build_info'],e['server']['model_path']) for e in environments})>1:
        raise ValueError('model server changed between arms')
    result=[]
    for case in cases:
        summary={'case':case['name'],'arms':{}}
        for slots,mode in ((1,'serial'),(2,'serial'),(2,'overlap')):
            selected=[r for r in rows if r['case']==case['name'] and r['slots']==slots and r['mode']==mode]
            if not selected:raise ValueError(f'missing measurements for {case["name"]}/{slots}/{mode}')
            summary['arms'][f'{slots}-{mode}']={'n':len(selected),
                **{key:statistics.median(r[key] for r in selected) for key in
                   ('pair_seconds','foreground_seconds','background_seconds','background_wait_after_foreground')},
                'foreground_completion_tokens':statistics.median(r['foreground']['usage']['completion_tokens'] for r in selected),
                'foreground_response_variants':len({r['foreground']['response_signature'] for r in selected}),
                'pair_min':min(r['pair_seconds'] for r in selected),'pair_max':max(r['pair_seconds'] for r in selected)}
        a,b,c=[summary['arms'][key] for key in ('1-serial','2-serial','2-overlap')]
        summary.update(speedup_vs_original=a['pair_seconds']/c['pair_seconds'],
                       speedup_from_overlap=b['pair_seconds']/c['pair_seconds'],
                       foreground_slowdown=c['foreground_seconds']/b['foreground_seconds'],
                       foreground_variants_all_arms=len({r['foreground']['response_signature'] for r in rows if r['case']==case['name']}))
        result.append(summary)
    (output/'summary.json').write_text(json.dumps(result,indent=2))
    lines=['# 推論並列化の固定入力測定','',
        '実ゲームのHTTP入力を固定して再実行。先読み対象の問いが事前に分かる場合の測定であり、攻略成績・先読み命中率の測定ではない。',
        '背景は保存済みの読み出し要求を順番に再生する。今回の選択結果で次の要求を変えないため、オンラインの情報収集品質は評価しない。',
        '各条件の中央値。モデル重みは共通、1枠あたりのコンテキストは16384。ウォームアップを除外し、2枠の条件順は反転する。','',
        'cache_prompt='+str(next(iter(policies)))+'。Falseは各要求でプロンプトKVを再計算する条件であり、モデル起動や画像処理の全キャッシュを消すものではない。','',
        '|入力|1枠直列 秒|2枠直列 秒|2枠並列 秒|元構成比|並列化のみ|前景遅延倍率|',
        '|---|---:|---:|---:|---:|---:|---:|']
    for r in result:
        a,b,c=[r['arms'][key] for key in ('1-serial','2-serial','2-overlap')]
        lines.append(f'|{r["case"]}|{a["pair_seconds"]:.3f}|{b["pair_seconds"]:.3f}|{c["pair_seconds"]:.3f}|'
                     f'{r["speedup_vs_original"]:.2f}x|{r["speedup_from_overlap"]:.2f}x|{r["foreground_slowdown"]:.2f}x|')
    lines+=['','速度倍率は大きいほど全処理が速い。前景遅延倍率は大きいほど操作・熟考の応答が遅くなる。',
            'temperature=0でもバッチ構成によって応答と生成トークン数が変わる場合がある。この所要時間は実際に返された応答に対するもので、全条件で生成計算量が等しいとは限らない。',
            'ケース・要求全文・画像はcases.json、応答・トークン数・各要求の開始終了はmeasurements-*.jsonl、GPU・サーバ情報はenvironment-*.json。',
            'summary.jsonに反復数、最小最大、出力トークン数、応答の変動も保存する。Kaggleへの速度外挿は行わない。','']
    (output/'report.md').write_text('\n'.join(lines))
    print('\n'.join(lines),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['prepare','run','report'])
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--source',type=Path)
    parser.add_argument('--reads',type=int,default=8)
    parser.add_argument('--base',default='http://vlm:8080/v1')
    parser.add_argument('--slots',type=int,choices=[1,2],default=1)
    parser.add_argument('--repeats',type=int,default=3)
    parser.add_argument('--modes',nargs='+',choices=['serial','overlap'],default=['serial'])
    parser.add_argument('--no-cache',action='store_true',help='disable prompt KV reuse; keep this policy equal across arms')
    args=parser.parse_args()
    if args.repeats<1 or args.reads<1:parser.error('positive repeats and reads required')
    if args.command=='prepare':
        if args.source is None:parser.error('--source required')
        prepare(args.source,args.output,args.reads)
    elif args.command=='run':run(args.output,args.base,args.slots,args.repeats,args.modes,not args.no_cache)
    else:report(args.output)


if __name__=='__main__':main()
