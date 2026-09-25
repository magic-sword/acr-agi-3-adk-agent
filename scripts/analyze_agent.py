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
        if path.name.endswith(('.model.jsonl', '.observations.jsonl', '.tools.jsonl',
                               '.artifacts.jsonl', '.requests.jsonl', '.execution.jsonl', '.learning.jsonl')):
            continue
        rows = read_records(path)
        if not rows:
            continue
        sections.append('<h2>' + escape(path.stem) + '</h2>')
        for row in rows:
            def field(label, value):
                value = json.dumps(value, ensure_ascii=False, indent=2) if not isinstance(value,str) else value
                return '<dt>'+escape(label)+'</dt><dd><pre>'+escape(value)+'</pre></dd>'
            sections.append('<details open><summary>Step '+escape(str(row.get('step')))
                            +' · '+escape(' → '.join(row.get('trace',[])))+'</summary><dl>')
            for label,key in [('観測ID','observation_id'),('現在の課題','task'),('認識の更新','summary'),
                              ('実操作と観測結果','outcome'),('次の操作／停止','action')]:
                sections.append(field(label,row.get(key)))
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
    for suffix, label in (('artifacts', '中間成果物（未実行を含む）'), ('execution', '実操作の送信と受付'), ('learning', '経験・スキル作成・試験・採用')):
        for path in sorted(directory.glob(f'*.{suffix}.jsonl')):
            sections.append('<details><summary>' + label + ': ' + escape(path.name) + '</summary><pre>'
                            + escape(json.dumps(read_records(path), ensure_ascii=False, indent=2))
                            + '</pre></details>')
    for path in sorted(directory.glob('*.requests.jsonl')):
        sections.append('<p>実HTTP入力: <a href="' + escape(path.name, quote=True) + '">'
                        + escape(path.name) + '</a>（画像本体はrequest-images内の参照先）</p>')
    return ('<!doctype html><html lang="ja"><meta charset="utf-8"><title>Agent decision log</title>'
            '<style>body{font:16px system-ui;max-width:1100px;margin:32px auto;padding:0 16px;'
            'color:#172554;background:#f8fafc}details{background:white;padding:12px;margin:12px 0;'
            'border:1px solid #cbd5e1;border-radius:8px}summary{cursor:pointer;font-weight:bold}'
            'dt{font-weight:bold}dd{margin:6px 0 16px}pre{white-space:pre-wrap;overflow-wrap:anywhere}</style>'
            '<h1>エージェントの判断ログ</h1><p>判断と実行、およびスキル獲得の記録を表示します。'
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
