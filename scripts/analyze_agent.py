"""Render recorded decisions and actual skill loads without a model or Docker."""
from __future__ import annotations

import argparse
from html import escape
import json
from pathlib import Path


def read_records(path):
    rows = []
    for line in path.read_text(encoding='utf-8').splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue  # An interrupted worker can leave a partial final line.
        if isinstance(row, dict):
            rows.append(row)
    return rows


def render(directory):
    sections = []
    for path in sorted(directory.glob('*.jsonl')):
        if path.name.endswith(('.model.jsonl', '.observations.jsonl', '.tools.jsonl')):
            continue
        rows = read_records(path)
        if not rows:
            continue
        sections.append('<h2>' + escape(path.stem) + '</h2>')
        for row in rows:
            decision = row.get('decision') or {}
            action = row.get('action') or {}
            tools = row.get('tool_executions', [])
            skills = [t.get('arguments', {}).get('skill_name', '?') for t in tools
                      if t.get('tool') == 'load_skill' and t.get('status') == 'success']
            def field(label, value):
                if not isinstance(value, str):
                    value = json.dumps(value, ensure_ascii=False, indent=2)
                return '<dt>' + escape(label) + '</dt><dd><pre>' + escape(value) + '</pre></dd>'
            sections.append('<details open><summary>Step ' + escape(str(row.get('step')))
                            + ' · ' + escape(' → '.join(row.get('trace', []))) + '</summary><dl>')
            sections.extend([
                field('観測ID', row.get('observation_id')),
                field('確定結果', action),
                field('判断理由（モデルの要約／ホスト判断）',
                      decision.get('action', {}).get('reason') or action.get('reason') or '未記録'),
                field('振り返り（モデルの解釈）', decision.get('reflection') or '未記録'),
                field('次の予測（未検証）', decision.get('prediction') or '未記録'),
                field('作業メモ（仮説）', row.get('notebook', decision.get('notebook', '未記録'))),
                field('実際に読み込んだスキル', skills if 'tool_executions' in row else '旧ログ：未記録'),
                field('前の操作の観測結果', row.get('previous_outcome', '未記録')),
            ])
            sections.append('</dl><details><summary>ツール実行・内部処理・エラー</summary><pre>'
                            + escape(json.dumps({k: row.get(k) for k in
                                      ('tool_executions', 'internal_trace', 'transitions', 'errors')},
                                      ensure_ascii=False, indent=2)) + '</pre></details></details>')
    # Include attempts even if the process stopped before COMMIT or model logging.
    for path in sorted(directory.glob('*.tools.jsonl')):
        sections.append('<details><summary>ツール実行イベント（未確定ターンを含む）: '
                        + escape(path.name) + '</summary><pre>'
                        + escape(json.dumps(read_records(path), ensure_ascii=False, indent=2))
                        + '</pre></details>')
    return ('<!doctype html><html lang="ja"><meta charset="utf-8"><title>Agent decision log</title>'
            '<style>body{font:16px system-ui;max-width:1100px;margin:32px auto;padding:0 16px;'
            'color:#172554;background:#f8fafc}details{background:white;padding:12px;margin:12px 0;'
            'border:1px solid #cbd5e1;border-radius:8px}summary{cursor:pointer;font-weight:bold}'
            'dt{font-weight:bold}dd{margin:6px 0 16px}pre{white-space:pre-wrap;overflow-wrap:anywhere}</style>'
            '<h1>エージェントの判断ログ</h1><p>観測 → 判断 → 確定を各手で表示します。'
            '理由・振り返り・メモはモデルが出力した要約であり、内部思考の再現ではありません。'
            'スキルのロード成功は、その手法の正しい適用を保証しません。'
            '画面の変化は予測の的中や進捗を意味しません。確定操作の実行結果は次の観測で確認します。</p>'
            + ''.join(sections or ['<p>判断ログがありません。</p>']) + '</html>')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if not args.directory.is_dir():
        parser.error('Log directory does not exist: ' + str(args.directory))
    output = args.output or args.directory / 'decisions.html'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render(args.directory), encoding='utf-8')
    print(output.resolve())


if __name__ == '__main__':
    main()
