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
    turns, calls = [], []
    for path in directory.glob('*.jsonl'):
        if path.name.endswith('.model.jsonl'):
            calls += read_jsonl(path)
        elif not path.name.endswith(('.observations.jsonl', '.tools.jsonl')):
            turns += read_jsonl(path)
    action_turns = [t for t in turns if t.get('action', {}).get('status') == 'action']
    counts = Counter(state for t in turns for state in t.get('trace', []))
    verification = Counter(v['result'] for t in turns for v in t.get('verification', []))
    valid = sum(bool(c.get('schema_valid')) for c in calls)
    tokens = {key: sum((c.get('usage') or {}).get(key, 0) for c in calls)
              for key in ('prompt_tokens', 'completion_tokens', 'total_tokens')}
    measured_usage = sum(isinstance(c.get('usage'), dict) for c in calls)
    http_requests = sum(c.get('http_requests', 1) for c in calls)
    skill_calls = Counter(
        call['function']['name'] for c in calls for exchange in c.get('exchanges', [])
        for call in tool_requests(exchange)
        if call.get('function', {}).get('name') in
        ('list_skills', 'load_skill', 'load_skill_resource', 'run_skill_script'))
    latency = [c['seconds'] for c in calls if 'seconds' in c]
    errors = [error for t in turns for error in t.get('errors', [])]
    perceptions = []
    for call in calls:
        if call.get('state') == 'OBSERVE' and call.get('schema_valid'):
            try:
                parsed = json.loads(call.get('response', ''))
                if isinstance(parsed, dict):
                    perceptions.append(parsed)
            except (ValueError, TypeError):
                pass
    interpretations = []
    reasoning_states = Counter()
    evidence_tools = Counter()
    for call in calls:
        reasoning_states[call.get('state', 'unknown')] += 1
        for exchange in call.get('exchanges', []):
            for tool in tool_requests(exchange):
                name = tool['function']['name']
                if name in ('list_observations', 'get_observation', 'compare_observations', 'observe_animation', 'observe_current', 'move_cursor'):
                    evidence_tools[name] += 1
        if call.get('schema_valid'):
            try:
                parsed = json.loads(call.get('response', ''))
                interpretation = parsed if call.get('state') == 'VERIFY' else parsed.get('interpretation')
                if isinstance(interpretation, dict):
                    interpretations.append(interpretation)
            except (ValueError, TypeError, AttributeError):
                pass
    empty_perceptions = sum(not p.get('facts') and not p.get('goal') for p in perceptions)
    repeats = 0
    for previous, current in zip(action_turns, action_turns[1:]):
        same_action = all(previous['action'].get(k) == current['action'].get(k) for k in ('action', 'x', 'y'))
        if previous.get('frame_hash') and previous['frame_hash'] == current.get('frame_hash') and same_action:
            repeats += 1
    hints = []
    if empty_perceptions:
        hints.append(f'解析可能な観測応答 {len(perceptions)}件中{empty_perceptions}件でfactsとgoalが空。画像中の対象・変化が記憶に取り込まれているか確認。')
    if calls and valid < len(calls):
        hints.append(f'モデル応答の型・通信エラー {len(calls)-valid}/{len(calls)}。model.jsonlのresponse/errorを確認。')
    if counts['PROBE'] and not counts['PLAN']:
        hints.append('PROBEのみでPLANに到達していない。目標仮説と重要なunknownsが更新されているか確認。')
    if repeats:
        hints.append(f'同じ盤面で同じ操作を選んだ連続箇所 {repeats}件。実験の重複または操作が効かない原因を確認。')
    if verification['contradicted']:
        hints.append(f'予測の反証 {verification["contradicted"]}件。観測差分とREVISE後の仮説を確認。')
    if any('stale' in e.lower() for e in errors):
        hints.append('観測ID・記憶revisionの不一致あり。モデルによるIDの転記と応答契約を確認。')
    if calls and valid == len(calls) and errors:
        hints.append('JSON型検証後の意味・操作検証エラーあり。判断ログerrorsとmodel.jsonlを照合。')
    executions = [tool for turn in turns for tool in turn.get('tool_executions', [])]
    loaded_skills = Counter(tool['arguments'].get('skill_name', 'unknown') for tool in executions
                            if tool['tool'] == 'load_skill' and tool['status'] == 'success')
    decisions = [t['decision'] for t in turns if t.get('decision')]
    missing_reasons = sum(not d.get('action', {}).get('reason', '').strip() for d in decisions)
    if missing_reasons:
        hints.append(f'判断理由の未記録 {missing_reasons}/{len(decisions)}件。予測だけでは選択理由を復元できません。')
    return {
        'loaded_skills': dict(loaded_skills),
        'tool_execution_errors': sum(t['status'] == 'error' for t in executions),
        'decisions_missing_reason': missing_reasons,
        'reasoning_calls_by_state': dict(reasoning_states),
        'evidence_tool_calls': dict(evidence_tools),
        'interpretation_count': len(interpretations),
        'interpretations_with_facts': sum(bool(p.get('facts')) for p in interpretations),
        'model_calls': len(calls), 'schema_valid_calls': valid,
        'model_http_requests': http_requests, 'skill_tool_calls': dict(skill_calls),
        'schema_valid_rate': valid / len(calls) if calls else None,
        'model_seconds': sum(latency), 'model_latency_p50': percentile(latency, .5),
        'model_latency_p95': percentile(latency, .95),
        'decision_latency_p50': percentile([t['decision_seconds'] for t in turns if 'decision_seconds' in t], .5),
        'tokens': tokens if measured_usage else None, 'usage_recorded_calls': measured_usage,
        'state_visits': dict(counts), 'verification': dict(verification),
        'validation_errors': errors, 'unchanged_action_repeats': repeats,
        'committed_actions': len(action_turns),
        'parsed_perceptions': len(perceptions), 'empty_perceptions': empty_perceptions,
        'hints': hints,
    }


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
        lines.append(f"|[{r['game_id']}]({r['game_id']}/result.json)|{r.get('levels_completed', '—')}|{r.get('actions', '—')}|{score}|"
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
              '- `cognition/*.observations.jsonl` と `cognition/frames/`: 実観測の色ID・PNG。',
              '- `gateway.jsonl`: 公式ゲートウェイへの操作と実際の戻り値。',
              '- `worker.log`: 実行ログと例外。', '']
    (root / 'report.md').write_text('\n'.join(lines))
    return summary
