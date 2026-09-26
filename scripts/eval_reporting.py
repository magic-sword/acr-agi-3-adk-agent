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
    artifacts = [r for p in directory.glob('*.artifacts.jsonl') for r in read_jsonl(p)]
    feedback = [r for r in artifacts if r.get('event') == 'action_feedback']
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
    hints = []
    if consecutive_repeats:
        hints.append(f'同じ操作の連続箇所: {consecutive_repeats}。実測された変化とモデルの解釈・注目対象を確認。')
    if valid<len(calls): hints.append('モデル要求・提出に失敗あり。model.jsonlを確認。')
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
        'tool_calls':dict(Counter(requests)),
        'tool_execution_errors':sum(t.get('status')=='error' for t in tools),
        'validation_errors':errors, 'consecutive_action_repeats':consecutive_repeats,
        'identical_frame_action_repeats':identical_frame_repeats,
        'fast_slow': {
            'plans':sum(r.get('event')=='plan_created' for r in artifacts),
            'reconsiderations':sum(r.get('event')=='reconciliation_requested' for r in artifacts),
            'completion_candidates':sum(r.get('event')=='completion_candidate' for r in artifacts),
            'confirmed_goals':sum(r.get('event')=='stage_accepted' and r.get('work')=='reconcile'
                                  and r.get('result',{}).get('goal_status')=='confirmed' for r in artifacts),
            'probe_results':sum(r.get('event')=='reconciliation_requested' and r.get('trigger')=='probe_result' for r in artifacts),
            'no_visible_effect':sum(r['trial'].get('frame_changed') is False for r in feedback),
            'repairs':sum(c.get('attempt',0)>0 for c in calls if c.get('work') in ('understand','backchain','ground','reconcile')),
            'by_work':{work:{'calls':len(selected),
                'seconds':sum(c.get('seconds',0) for c in selected),
                'latency_p50':percentile([c['seconds'] for c in selected if 'seconds' in c],.5),
                'latency_p95':percentile([c['seconds'] for c in selected if 'seconds' in c],.95),
                'completion_tokens':sum((c.get('usage') or {}).get('completion_tokens',0) for c in selected)}
                for work in ('understand','backchain','ground','reconcile','choose_skill','execute_step')
                for selected in [[c for c in calls if c.get('work')==work]]}},
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
    lines += ['', '## 工程別の判断時間', '', '|ゲーム|工程|呼出し数|中央値（秒）|出力トークン合計|',
              '|---|---|---:|---:|---:|']
    for r in results:
        for work, metrics in r.get('fast_slow',{}).get('by_work',{}).items():
            median=metrics.get('latency_p50')
            latency=f'{median:.3f}' if median is not None else '—'
            lines.append(f"|{r['game_id']}|{work}|{metrics['calls']}|{latency}|{metrics['completion_tokens']}|")
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
              '- `cognition/*.tools.jsonl`: ツール実行前後の記録。計画の提出・検証結果。',
              '- `cognition/*.artifacts.jsonl`: 判断の更新と選択された操作。未実行も区別。',
              '- `cognition/*.artifacts.jsonl` の `plan_created` / `fast_selected` / `reconciliation_requested` / `stage_accepted` / `action_feedback`: 計画・高速選択・熟考への復帰・実測結果。',
              '- `cognition/*.requests.jsonl` と `request-images/`: 実HTTP入力と画像参照。',
              '- `cognition/*.execution.jsonl`: ドライバの送信・受付・結果不明イベント。',
              '- `cognition/*.observations.jsonl` と `cognition/frames/`: 実観測の色ID・PNG。',
              '- `gateway.jsonl`: 公式ゲートウェイへの操作と実際の戻り値。',
              '- `worker.log`: 実行ログと例外。', '']
    (root / 'report.md').write_text('\n'.join(lines))
    return summary
