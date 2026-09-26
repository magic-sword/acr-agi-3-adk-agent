"""Local evaluation summaries; missing scores are never silently converted to zero."""
from __future__ import annotations

from collections import Counter
import csv
import json
from pathlib import Path
import statistics


def read_jsonl(path: Path) -> list[dict]:
    records = []
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                # A killed worker may leave one incomplete final line.
                continue
    return records


def percentile(values, q):
    if not values:
        return None
    values = sorted(values)
    position = (len(values) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def tool_requests(exchange):
    """Count structured calls or successfully normalized native Qwen calls once."""
    structured = exchange.get('response', {}).get('tool_calls') or []
    if structured:
        return structured
    return [{'function': {'name': c['name'], 'arguments': c['arguments']}}
            for c in exchange.get('normalized_tool_calls', [])]


def diagnostics(directory: Path) -> dict:
    calls = [r for p in directory.glob('*.model.jsonl') for r in read_jsonl(p)]
    tools = [r for p in directory.glob('*.tools.jsonl') for r in read_jsonl(p) if r.get('event')=='tool_finished']
    learning = [r for p in directory.glob('*.learning.jsonl') for r in read_jsonl(p)]
    notebook = [r for p in directory.glob('*.notebook.jsonl') for r in read_jsonl(p)]
    artifacts = [r for p in directory.glob('*.artifacts.jsonl') for r in read_jsonl(p)]
    worlds = [r['after']['data'] for r in notebook
              if r.get('event')=='note_changed' and r.get('after',{}).get('kind')=='world']
    findings = {f['experiment_id']:f for w in worlds for f in w['conditional_findings']}
    target_checks = [r for r in artifacts if r.get('event')=='target_checked'
                     and r.get('source')=='host_region_and_pixel_mask']
    experiments = [r for p in directory.glob('*.experiments.jsonl') for r in read_jsonl(p)]
    reviews = list({(r.get('run_id'), r['experiment']['id']): r['experiment']['data']
                    for r in experiments if r['event'] == 'experiment_reviewed'}.values())
    auto_skills = [r for p in directory.glob('*.tools.jsonl') for r in read_jsonl(p)
                   if r.get('event') == 'skill_instructions_loaded']
    turns = [r for p in directory.glob('*.jsonl') if p.name.count('.')==1 for r in read_jsonl(p)]
    actions = [t for t in turns if t.get('action',{}).get('status')=='action']
    valid = sum(bool(c.get('schema_valid')) for c in calls)
    latency = [c['seconds'] for c in calls if 'seconds' in c]
    measured = sum(isinstance(c.get('usage'),dict) for c in calls)
    tokens = {k:sum((c.get('usage') or {}).get(k,0) for c in calls)
              for k in ('prompt_tokens','completion_tokens','total_tokens')}
    requests = [t['function']['name'] for c in calls for e in c.get('exchanges',[]) for t in tool_requests(e)]
    consecutive_repeats = identical_frame_repeats = 0
    for path in directory.glob('*.jsonl'):
        if path.name.count('.') != 1:
            continue
        run_actions = [r for r in read_jsonl(path) if r.get('action', {}).get('status') == 'action']
        for a, b in zip(run_actions, run_actions[1:]):
            same = all(a['action'].get(k) == b['action'].get(k) for k in ('action', 'x', 'y'))
            consecutive_repeats += same
            identical_frame_repeats += bool(same and a.get('frame_hash') and a['frame_hash'] == b.get('frame_hash'))
    errors = [e for t in turns for e in t.get('errors',[])]
    events = Counter(r.get('event') for r in learning)
    outcomes = Counter(r.get('outcome') for r in learning if r.get('event')=='skill_execution_finished')
    hints = []
    if consecutive_repeats:
        hints.append(f'同じ操作の連続箇所: {consecutive_repeats}。対象領域の実験判定と有限回の再試行理由を確認。')
    if valid<len(calls): hints.append('モデル要求・提出に失敗あり。model.jsonlを確認。')
    if events['skill_drafted'] and not events['skill_promoted']: hints.append('候補は未昇格。固定評価と実試行結果を確認。')
    return {'model_calls':len(calls), 'schema_valid_calls':valid,
        'model_http_requests':sum(c.get('http_requests',1) for c in calls),
        'schema_valid_rate':valid/len(calls) if calls else None,
        'model_seconds':sum(latency), 'model_latency_p50':percentile(latency,.5),
        'model_latency_p95':percentile(latency,.95),
        'decision_latency_p50':percentile([t['decision_seconds'] for t in turns if 'decision_seconds' in t],.5),
        'tokens':tokens if measured else None, 'usage_recorded_calls':measured,
        'state_visits':dict(Counter(s for t in turns for s in t.get('trace',[]))),
        'reasoning_calls_by_state':dict(Counter(c.get('state') for c in calls)),
        'reasoning_calls_by_work':dict(Counter(c.get('work') for c in calls)),
        'notebook_events':dict(Counter(r.get('event') for r in notebook)),
        'notebook_tool_calls':dict(Counter(n for n in requests if n in
            ('read_notebook','write_note','erase_note','set_bookmark'))),
        'host_loaded_skills':dict(Counter(r['skill'] for r in auto_skills)),
        'skill_tool_calls':dict(Counter(n for n in requests if n in ('load_skill','load_skill_resource','propose_skill','read_skill'))),
        'loaded_skills':dict(Counter(t['arguments'].get('skill_name') for t in tools if t['tool']=='load_skill' and t['status']=='success')),
        'tool_execution_errors':sum(t.get('status')=='error' for t in tools),
        'learning_events':dict(events), 'skill_execution_outcomes':dict(outcomes),
        'validation_errors':errors, 'consecutive_action_repeats':consecutive_repeats,
        'identical_frame_action_repeats':identical_frame_repeats,
        'experiment_events':dict(Counter(r['event'] for r in experiments)),
        'experiment_verdicts':dict(Counter(r['review']['verdict'] for r in reviews)),
        'experiment_reviewers':dict(Counter(r['reviewer'] for r in reviews)),
        'world_memory': {'versions':len(worlds), 'conditional_trials':len(findings),
            'questions_tested':len({f['question_id'] for f in findings.values() if f['resolved_test']}),
            'pixel_mask_checks':len(target_checks),
            'pixel_mask_matches':sum(r['verdict']=='matched' for r in target_checks),
            'last_open_questions':sum(q['status']=='open' for q in worlds[-1]['questions']) if worlds else 0,
            'note':'Mask matches measure geometric consistency, not semantic object-recognition accuracy.'},
        'bounded_experiment_retries':sum(r['event']=='experiment_started' and
            bool(r['experiment']['data']['plan']['retry_of']) for r in experiments),
        'committed_actions':len(actions), 'hints':hints}


def write_report(root: Path, results: list[dict]) -> dict:
    scores = [r['sdk_score_full_game'] for r in results if r.get('sdk_score_full_game') is not None]
    summary = {
        'games': results, 'requested_games': len(results),
        'scored_games': len(scores),
        'mean_sdk_score_full_game': statistics.mean(scores) if len(scores) == len(results) and scores else None,
        'mean_available_score': statistics.mean(scores) if scores else None,
        'completed_levels': sum(r.get('levels_completed') or 0 for r in results),
        'submitted_actions': sum(r.get('actions') or 0 for r in results),
        'total_worker_seconds': sum(r.get('wall_seconds') or 0 for r in results),
        'note': '公開ゲームの制限付きローカル評価。SDKの全レベル分母を維持。Kaggleリーダーボードスコアではない。',
    }
    (root / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    columns = ['game_id', 'levels_completed', 'actions', 'sdk_score_full_game', 'wall_seconds',
               'stop_reason', 'limit_reached', 'model_calls', 'schema_valid_rate', 'model_latency_p50', 'model_latency_p95']
    with (root / 'summary.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for result in results:
            writer.writerow({key: result.get(key, '') for key in columns})
    lines = ['# ローカル評価結果', '', summary['note'], '',
             '|ゲーム|到達レベル|操作数|SDKスコア|秒|停止理由|モデル呼出し|',
             '|---|---:|---:|---:|---:|---|---:|']
    for r in results:
        score = f"{r['sdk_score_full_game']:.3f}" if r.get('sdk_score_full_game') is not None else '未取得'
        lines.append(f"|[{r['game_id']}]({r['game_id']}/cognition/decisions.html)|{r.get('levels_completed', '—')}|{r.get('actions', '—')}|{score}|"
                     f"{r.get('wall_seconds', 0):.1f}|{r.get('stop_reason', 'error')} {r.get('limit_reached', '')}|{r.get('model_calls', 0)}|")
    lines += ['', '## 改善の調査箇所', '']
    for r in results:
        lines.append(f"### {r['game_id']}")
        lines.append('')
        for hint in r.get('hints', []) or ['この短時間評価だけでは改善点を特定できません。観測・操作ログを確認してください。']:
            lines.append('- ' + hint)
        lines.append('')
    lines += ['## ログ', '', '- `manifest.json`: 実行条件、ソースハッシュ、依存バージョン、モデル設定と本番との差分。',
              '- 各ゲームの `scorecard.json`: 公式SDKの生スコアカード。',
              '- `cognition/*.model.jsonl`: 入力記憶・モデル生応答・型検証・時間・トークン数。',
              '- `cognition/*.jsonl`: 各手の予測照合・遷移・検証エラー。',
              '- `cognition/*.tools.jsonl`: ツール実行前後の記録。読み込んだスキル名・成否・引数・結果。',
              '- `cognition/*.artifacts.jsonl`: 判断の更新と選択された操作。未実行も区別。',
              '- `cognition/*.learning.jsonl`: 経験・候補作成・実試行・評価・昇格・停止。',
              '- `cognition/*.notebook.jsonl`: 攻略ノートの入力ページ・参照・版の変更・撤回・しおり。',
              '- `cognition/*.experiments.jsonl`: 操作前の小目標・仮説・予測、実測、判定・更新・中断。',
              '- `cognition/<run>/notebook/`: ノートの各版としおりの保存先。',
              '- `cognition/<run>/skills/library.json`: 版と評価証拠を含むライブラリ。',
              '- `cognition/*.requests.jsonl` と `request-images/`: 実HTTP入力と画像参照。',
              '- `cognition/*.execution.jsonl`: ドライバの送信・受付・結果不明イベント。',
              '- `cognition/*.observations.jsonl` と `cognition/frames/`: 実観測の色ID・PNG。',
              '- `gateway.jsonl`: 公式ゲートウェイへの操作と実際の戻り値。',
              '- `worker.log`: 実行ログと例外。', '']
    (root / 'report.md').write_text('\n'.join(lines))
    return summary
