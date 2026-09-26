"""Bounded local competition-gateway evaluation. Never uploads or submits to Kaggle."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import threading
import time
import traceback
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.eval_reporting import diagnostics, write_report, read_jsonl
from scripts.analyze_agent import render as render_decisions


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def git(*args):
    return subprocess.run(['git', *args], cwd=ROOT, text=True, capture_output=True).stdout.strip()


def model_server_info():
    url = os.getenv('VLM_API_BASE', 'http://vlm:8080/v1').rstrip('/')
    base = url.removesuffix('/v1')
    with urllib.request.urlopen(base + '/health', timeout=5) as response:
        health = json.load(response)
    info = {'health': health}
    for endpoint in ('/props', '/v1/models'):
        try:
            with urllib.request.urlopen(base + endpoint, timeout=5) as response:
                info[endpoint] = json.load(response)
        except Exception as exc:
            info[endpoint] = {'unavailable': type(exc).__name__}
    return info


def prepare(games):
    from arc_agi import Arcade, OperationMode
    # Only preparation can contact the public API. No online scorecard or Kaggle API.
    os.environ['OPERATION_MODE'] = 'normal'
    arcade = Arcade(operation_mode=OperationMode.NORMAL, environments_dir=str(ROOT / 'environment_files'))
    known = [e.game_id for e in arcade.get_environments()]
    for game in games:
        matches = [g for g in known if g == game or g.split('-')[0] == game]
        if len(matches) != 1:
            raise ValueError(f'Unknown or ambiguous game: {game}: {matches}')
        env = arcade.make(matches[0])
        if env is None:
            raise RuntimeError(f'Could not cache {game}')
        print(f'Cached {matches[0]}', flush=True)
    arcade.close_scorecard()


def resolve_games(games):
    metadata = []
    for path in (ROOT / 'environment_files').rglob('metadata.json'):
        data = json.loads(path.read_text())
        metadata.append((path, data))
    selected = []
    for game in games:
        matches = [(p, d) for p, d in metadata if d['game_id'] == game or d['game_id'].split('-')[0] == game]
        if len(matches) != 1:
            raise ValueError(f'{game}: cached version missing or ambiguous; run --prepare or specify full game ID')
        selected.append(matches[0])
    return selected


def run_worker(spec_path):
    spec = json.loads(spec_path.read_text())
    out = Path(spec['output'])
    # Load exactly the source snapshot used by the notebook packager.
    sys.path[:0] = [spec['package'], str(ROOT / 'vendor/ARC-AGI-3-Agents')]
    os.environ.update(ADK_MODEL=spec['model'], COGNITION_LOG_DIR=str(out / 'cognition'),
                      COGNITION_SECONDS=str(spec['seconds']), OPERATION_MODE='offline',
                      AGENTOPS_API_KEY='', WANDB_MODE='disabled')
    from arc_agi import Arcade, OperationMode
    from arc_agi.server import create_app
    from werkzeug.serving import make_server
    from flask import request
    from agent.my_agent import MyAgent
    import agent.my_agent as deployed
    assert Path(deployed.__file__).resolve().is_relative_to(Path(spec['package']).resolve())
    result = {'game_id': spec['game_id'], 'sdk_score_full_game': None, 'actions': 0,
              'levels_completed': 0, 'loaded_agent_file': deployed.__file__}
    started = time.monotonic()
    player = None
    server = None
    client = None
    card = None
    gateway_lock = threading.Lock()
    try:
        backend = Arcade(operation_mode=OperationMode.OFFLINE, arc_api_key='local-evaluation',
                         environments_dir=spec['environments'], recordings_dir=str(out / 'recordings'))
        assert backend.operation_mode is OperationMode.OFFLINE
        app, _ = create_app(backend, competition_mode=True, save_all_recordings=True, include_frame_data=True)

        @app.after_request
        def record_response(response):
            if request.path.startswith('/api/cmd/'):
                # Keep action ACKs even if the agent process later exceeds its hard limit.
                data = {'path': request.path, 'request': request.get_json(silent=True),
                        'http_status': response.status_code, 'response': response.get_json(silent=True)}
                with gateway_lock, (out / 'gateway.jsonl').open('a') as log:
                    log.write(json.dumps(data) + '\n')
            return response

        server = make_server('127.0.0.1', 0, app, threaded=True)
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        endpoint = f'http://127.0.0.1:{server.server_port}'
        client = Arcade(operation_mode=OperationMode.ONLINE, arc_api_key='local-evaluation',
                        arc_base_url=endpoint, environments_dir=str(out / 'no-client-environments'),
                        recordings_dir=str(out / 'client-recordings'))
        card = client.open_scorecard(tags=['agent', 'local-bounded-evaluation'])
        env = client.make(spec['game_id'], scorecard_id=card)
        if env is None:
            raise RuntimeError('local gateway could not initialize game')
        player = MyAgent(card_id=card, game_id=spec['game_id'], agent_name='MyAgent.evaluation',
                         ROOT_URL=endpoint, record=False, arc_env=env, tags=['local-evaluation'])
        player.MAX_ACTIONS = spec['steps'] - 1
        player.evaluation_max_levels = spec['levels'] or None
        player.main()
        memory = player._runtime.memory
        final = player.frames[-1]
        result.update(actions=player.action_counter, levels_completed=final.levels_completed,
                      win_levels=final.win_levels, game_state=final.state.name,
                      stop_reason=memory.stop_reason, lifecycle=memory.lifecycle,
                      run_id=memory.run_id, resets=memory.resets)
    except Exception as exc:
        traceback.print_exc()
        result.update(stop_reason='worker_error', error=f'{type(exc).__name__}: {exc}')
        if player is not None:
            result.update(actions=player.action_counter, levels_completed=player.frames[-1].levels_completed)
    finally:
        if client is not None and card is not None:
            try:
                scorecard = client.close_scorecard(card)
                if scorecard is not None:
                    payload = scorecard.model_dump(mode='json')
                    write_json(out / 'scorecard.json', payload)
                    messages = [run.get('message') for env in payload.get('environments', [])
                                for run in env.get('runs', []) if run.get('message')]
                    result['score_warnings'] = messages
                    result['sdk_score_full_game'] = None if messages else scorecard.score
            except Exception as exc:
                result['scorecard_error'] = f'{type(exc).__name__}: {exc}'
        if server is not None:
            server.shutdown()
            server.server_close()
        result['wall_seconds'] = time.monotonic() - started
        result.update(diagnostics(out / 'cognition'))
        write_json(out / 'result.json', result)
    return 0 if result.get('stop_reason') != 'worker_error' else 1


def recover_interrupted(out, game_id, wall_seconds):
    events = read_jsonl(out / 'gateway.jsonl')
    completed = [e for e in events if e.get('http_status') == 200 and isinstance(e.get('response'), dict)]
    last = completed[-1]['response'] if completed else {}
    return {'game_id': game_id, 'stop_reason': 'hard_timeout', 'sdk_score_full_game': None,
            'wall_seconds': wall_seconds, 'actions': max(0, len(completed) - 1),
            'levels_completed': last.get('levels_completed'), 'game_state': last.get('state'),
            'actions_note': 'Acknowledged gateway actions excluding initialization RESET; an in-flight action may be unconfirmed.',
            **diagnostics(out / 'cognition')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--games', default='ls20,vc33,ft09')
    parser.add_argument('--steps', type=int, default=12)
    parser.add_argument('--levels', type=int, default=1, help='stop after N cleared levels; 0 disables')
    parser.add_argument('--seconds', type=float, default=90)
    parser.add_argument('--hard-seconds', type=float, default=110, help='process limit per game including initialization/cleanup')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--prepare', action='store_true', help='download public environments, then exit; no model run')
    parser.add_argument('--offline-policy', action='store_true', help='test harness without VLM; not a model performance run')
    parser.add_argument('--worker', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        return run_worker(args.worker)
    games = args.games.split(',')
    if (len(set(games)) != len(games) or not all(re.fullmatch(r'[a-z0-9]+(?:-[a-z0-9]+)?', g) for g in games)
            or args.steps < 1 or args.levels < 0 or not math.isfinite(args.seconds)
            or not math.isfinite(args.hard_seconds) or args.seconds <= 0 or args.hard_seconds <= args.seconds):
        parser.error('invalid game list or budgets; hard-seconds must exceed seconds')
    if args.prepare:
        prepare(games)
        return 0
    selected = resolve_games(games)
    model = '' if args.offline_policy else 'local/qwen3-vl-4b-instruct'
    server_info = model_server_info() if model else {'mode': 'deterministic probe'}
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    out = (args.output or ROOT / 'outputs/evaluations' / stamp).resolve()
    out.mkdir(parents=True, exist_ok=False)
    from scripts.build_notebook import SOURCES
    package = out / 'package'
    hashes = {}
    for rel, source in SOURCES.items():
        target = package / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        hashes[rel] = sha256(target)
    manifest = {'observatory_schema': 1, 'created_utc': stamp, 'games': [d['game_id'] for _, d in selected],
                'policy': 'visual_attention',
                'limits': {'steps': args.steps, 'levels': args.levels, 'seconds': args.seconds, 'hard_seconds': args.hard_seconds},
                'agent_model': model, 'server_info': server_info, 'source_sha256': hashes,
                'git_revision': git('rev-parse', 'HEAD'), 'git_dirty': bool(git('status', '--porcelain')),
                'framework_revision': git('-C', 'vendor/ARC-AGI-3-Agents', 'rev-parse', 'HEAD'),
                'python': sys.version, 'packages': {n: importlib.metadata.version(n) for n in ('google-adk', 'google-genai', 'arc-agi', 'arcengine')},
                'cognition_settings': {k: os.getenv(k, default) for k, default in [
                    ('COGNITION_REPAIR_ATTEMPTS', '1'), ('COGNITION_MAX_RESETS', '2')]},
                'scoring_source_sha256': sha256(Path(importlib.metadata.distribution('arc-agi').locate_file('arc_agi/scorecard.py'))),
                'gateway': 'official arc-agi SDK, localhost HTTP, competition_mode=True, one game per scorecard',
                'seed': 0, 'execution': 'sequential fresh worker/session for each game',
                'differences_from_kaggle': [
                    'Public cached games only; limits shorten the game but scoring retains the full level denominator.',
                    'Local SDK gateway, not the private Kaggle gateway image; private scoring changes cannot be verified.',
                    'Local GPU and llama.cpp binary may differ from the attached Kaggle bundle.',
                    'Sequential workers instead of the framework Swarm concurrent game threads.'],
                'models': {}, 'environment_sha256': {}}
    bundle = ROOT / '.cache/model-cache/qwen3-vl-4b'
    for name in ('Qwen3VL-4B-Instruct-Q4_K_M.gguf', 'mmproj-Qwen3VL-4B-Instruct-Q8_0.gguf'):
        path = bundle / name
        manifest['models'][name] = {'size': path.stat().st_size, 'sha256': sha256(path)} if path.exists() else {'unavailable': True}
    gpu = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,driver_version', '--format=csv,noheader'], text=True, capture_output=True)
    manifest['gpu'] = gpu.stdout.strip()
    write_json(out / 'manifest.json', manifest)
    results = []
    for metadata, data in selected:
        game_id = data['game_id']
        dest = out / game_id
        env_dir = dest / 'environments' / game_id
        env_dir.mkdir(parents=True)
        for file in metadata.parent.iterdir():
            if file.is_file() and file.suffix in ('.py', '.json'):
                shutil.copyfile(file, env_dir / file.name)
                manifest['environment_sha256'][game_id + '/' + file.name] = sha256(file)
        write_json(out / 'manifest.json', manifest)
        spec = {'game_id': game_id, 'output': str(dest), 'package': str(package),
                'policy': 'visual_attention',
                'environments': str(dest / 'environments'), 'model': model,
                'steps': args.steps, 'levels': args.levels, 'seconds': args.seconds}
        write_json(dest / 'spec.json', spec)
        started = time.monotonic()
        print(f'Evaluating {game_id}: <= {args.steps} actions, <= {args.levels or "all"} levels, {args.seconds}s reasoning', flush=True)
        with (dest / 'worker.log').open('w') as log:
            proc = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--worker', str(dest / 'spec.json')],
                                    cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            try:
                code = proc.wait(timeout=args.hard_seconds)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)
                result = recover_interrupted(dest, game_id, time.monotonic() - started)
                write_json(dest / 'result.json', result)
            else:
                if (dest / 'result.json').exists():
                    result = json.loads((dest / 'result.json').read_text())
                else:
                    result = {'game_id': game_id, 'stop_reason': 'worker_error', 'exit_code': code,
                              'sdk_score_full_game': None, 'wall_seconds': time.monotonic() - started,
                              **diagnostics(dest / 'cognition')}
                    write_json(dest / 'result.json', result)
        if result.get('stop_reason') == 'budget_exhausted':
            result['limit_reached'] = 'actions' if result.get('actions', 0) >= args.steps else 'time'
        elif result.get('stop_reason') == 'level_limit':
            result['limit_reached'] = 'levels'
        write_json(dest / 'result.json', result)
        if (dest / 'cognition').is_dir():
            (dest / 'cognition' / 'decisions.html').write_text(render_decisions(dest / 'cognition'), encoding='utf-8')
        results.append(result)
        write_report(out, results)
        print(f"  levels={result.get('levels_completed')} actions={result.get('actions')} score={result.get('sdk_score_full_game')} stop={result['stop_reason']}", flush=True)
    print(f'Report: {out / "report.md"}', flush=True)
    return 1 if any(r['stop_reason'] in ('worker_error', 'hard_timeout') or r.get('sdk_score_full_game') is None for r in results) else 0


if __name__ == '__main__':
    raise SystemExit(main())
