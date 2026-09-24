"""Render source-derived workflow and skill wiring with Python's standard library."""
from __future__ import annotations

import argparse
import ast
from collections import deque
from datetime import datetime, timezone
import hashlib
from html import escape
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def extract(root: Path) -> dict:
    paths = ['agent/cognition/workflow.py', 'agent/cognition/skills.py', 'agent/cognition/engine.py']
    sources = {p: (root / p).read_text() for p in paths}
    workflow, skills, engine = [ast.parse(sources[p]) for p in paths]
    runtime = next(n for n in workflow.body if isinstance(n, ast.ClassDef) and n.name == 'CognitiveRuntime')
    build = next(n for n in runtime.body if isinstance(n, ast.FunctionDef) and n.name == '_build_graph')
    local = {n.name for n in build.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    bindings = {}

    def resolve(n):
        if isinstance(n, ast.Constant) and isinstance(n.value, str):
            return n.value
        if isinstance(n, ast.Name):
            if n.id in bindings:
                return bindings[n.id]
            if n.id in local:
                return n.id.upper()
        if isinstance(n, (ast.List, ast.Tuple)):
            return [resolve(x) for x in n.elts]
        if isinstance(n, ast.Dict):
            return {resolve(k): resolve(v) for k, v in zip(n.keys, n.values)}
        raise ValueError(f'Unsupported workflow expression at line {n.lineno}: {ast.dump(n)}')

    for n in build.body:
        if isinstance(n, ast.Assign) and isinstance(n.value, ast.Dict):
            for target in n.targets:
                if isinstance(target, ast.Name):
                    bindings[target.id] = resolve(n.value)
    calls = [n for n in ast.walk(build) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'Workflow']
    if len(calls) != 1:
        raise ValueError('Expected exactly one Workflow constructor')
    specs = resolve(next(k.value for k in calls[0].keywords if k.arg == 'edges'))
    edges = []
    for chain in specs:
        for a, b in zip(chain, chain[1:]):
            if not isinstance(a, str):
                raise ValueError('Unsupported route source')
            for label, dest in (b.items() if isinstance(b, dict) else [('', b)]):
                if not isinstance(dest, str):
                    raise ValueError('Unsupported route destination')
                edges.append({'from': a, 'to': dest, 'label': label})
    names = dict.fromkeys(e[k] for e in edges for k in ('from', 'to'))
    constants = {}
    for n in skills.body:
        if isinstance(n, ast.Assign):
            for target in n.targets:
                if isinstance(target, ast.Name) and target.id in ('STATE_SKILLS', 'SKILL_NAMES'):
                    constants[target.id] = ast.literal_eval(n.value)
    # Run only the repo's pure selector; do not import ADK, model or game modules.
    selector = next(n for n in skills.body if isinstance(n, ast.FunctionDef) and n.name == 'selected_skills')
    namespace = dict(constants)
    exec(compile(ast.Module(body=[selector], type_ignores=[]), paths[1], 'exec'), namespace)
    selected = namespace['selected_skills']
    matrix = {}
    for state, ids in constants['STATE_SKILLS'].items():
        if not ids:
            matrix[state] = []
            continue
        matrix[state] = list(selected(state))
    missing = [sid for ids in matrix.values() for sid in ids if sid not in constants['SKILL_NAMES']]
    if missing:
        raise ValueError(f'Unknown skill IDs: {missing}')
    skill_files = {}
    for sid, name in constants['SKILL_NAMES'].items():
        p = f'agent/skills/{name}/SKILL.md'
        if not (root / p).is_file():
            raise ValueError(f'Missing skill file: {p}')
        sources[p] = (root / p).read_text()
        skill_files[sid] = p
    turn = next(n for n in engine.body if isinstance(n, ast.ClassDef) and n.name == 'CognitiveTurn')
    internal = []
    for method in turn.body:
        if not isinstance(method, ast.FunctionDef):
            continue
        for call in ast.walk(method):
            if (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
                    and isinstance(call.func.value, ast.Name) and call.func.value.id == 'self'
                    and call.func.attr == 'enter' and call.args and isinstance(call.args[0], ast.Constant)):
                state = call.args[0].value
                if state not in names:
                    callers = [m.name for m in turn.body if isinstance(m, ast.FunctionDef)
                               and any(isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute)
                                       and isinstance(c.func.value, ast.Name) and c.func.value.id == 'self'
                                       and c.func.attr == method.name for c in ast.walk(m))]
                    internal.append({'state': state, 'method': method.name, 'callers': callers})
    agent = next(n for n in runtime.body if isinstance(n, ast.FunctionDef) and n.name == '_agent')
    llm = next(n for n in ast.walk(agent) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'LlmAgent')
    tools = ast.unparse(next(k.value for k in llm.keywords if k.arg == 'tools'))
    completion_path = 'agent/cognition/completion.py'
    sources[completion_path] = (root / completion_path).read_text()
    instruction_path = 'agent/cognition/instructions.py'
    sources[instruction_path] = (root / instruction_path).read_text()
    planning_path = 'agent/cognition/planning.py'
    sources[planning_path] = (root / planning_path).read_text()
    completion_tree = ast.parse(sources[completion_path])
    completions = next(ast.literal_eval(n.value) for n in completion_tree.body
                       if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name)
                       and t.id == 'COMPLETION_TOOLS' for t in n.targets))

    details = {}
    for n in build.body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            details[f'Workflow: {n.name} (L{n.lineno})'] = ast.get_source_segment(sources[paths[0]], n)
    for n in turn.body:
        if isinstance(n, ast.FunctionDef) and n.name in ('route', 'update', 'consolidate', 'recover'):
            details[f'Engine: {n.name} (L{n.lineno})'] = ast.get_source_segment(sources[paths[2]], n)
    return {'generated_utc': datetime.now(timezone.utc).isoformat(), 'nodes': list(names),
            'edges': edges, 'skill_names': constants['SKILL_NAMES'], 'state_skills': constants['STATE_SKILLS'],
            'effective_skills': matrix, 'skill_files': skill_files, 'internal': internal,
            'completion_tools': completions, 'model_tools': tools, 'source_details': details,
            'sha256': {p: hashlib.sha256(s.encode()).hexdigest() for p, s in sources.items()}}


def render_svg(data: dict) -> str:
    # Shortest-path layers keep routing choices together; reverse edges are curved.
    depth = {data['nodes'][0]: 0}
    queue = deque(depth)
    while queue:
        a = queue.popleft()
        for e in data['edges']:
            if e['from'] == a and e['to'] not in depth:
                depth[e['to']] = depth[a] + 1
                queue.append(e['to'])
    layers = {}
    for name in data['nodes']:
        layers.setdefault(depth.get(name, max(depth.values()) + 1), []).append(name)
    width = max(1200, max(map(len, layers.values())) * 205 + 60)
    pos = {name: (width * (i + 1) / (len(names) + 1), 150 + level * 135)
           for level, names in layers.items() for i, name in enumerate(names)}
    table_y = max(y for x, y in pos.values()) + 125
    rows = data['effective_skills']
    legend_y = table_y + 80 + len(rows) * 35
    height = legend_y + len(data['skill_names']) * 26 + 140 + len(data['internal']) * 30
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
             '<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0L10 5L0 10Z" fill="#64748b"/></marker></defs>',
             '<style>text{font-family:Arial,sans-serif;fill:#172554} .small{font-size:13px} .edge{fill:none;stroke:#64748b;stroke-width:1.5;marker-end:url(#arrow)}</style>',
             f'<rect width="{width}" height="{height}" fill="#f8fafc"/>']

    def text(x, y, value, size=16, anchor='start'):
        parts.append(f'<text x="{x}" y="{y}" font-size="{size}" text-anchor="{anchor}">{escape(str(value))}</text>')

    text(30, 40, 'Cognitive agent · Workflow & skill wiring', 26)
    text(30, 68, data['generated_utc'] + ' · source-derived / static wiring', 14)
    text(30, 92, 'Blue: skill-enabled state   Gray: host logic   Arrows: Workflow edges (conditions in HTML)', 14)
    for e in data['edges']:
        x, y = pos[e['from']]; xx, yy = pos[e['to']]
        if yy > y:
            path = f'M{x},{y+26} C{x},{(y+yy)/2} {xx},{(y+yy)/2} {xx},{yy-27}'
        else:
            bend = min(width - 15, max(x, xx) + 135)
            path = f'M{x+85},{y} C{bend},{y-80} {bend},{yy-80} {xx+85},{yy}'
        parts.append(f'<path class="edge" d="{path}"><title>{escape(e["from"] + " → " + e["to"])}</title></path>')
    for name, (x, y) in pos.items():
        fill = '#dbeafe' if data['state_skills'].get(name) else '#e2e8f0'
        parts.append(f'<rect x="{x-85}" y="{y-26}" width="170" height="52" rx="10" fill="{fill}" stroke="#64748b"/>')
        text(x, y+5, name, 16, 'middle')
    text(30, table_y, 'Effective skill connections · available to model, not proof of invocation', 21)
    ids = list(data['skill_names'])
    cell = (width - 300) / max(1, len(ids))
    for i, sid in enumerate(ids):
        text(290 + i*cell, table_y+35, sid, 14, 'middle')
    for r, (state, connected) in enumerate(rows.items()):
        y = table_y + 65 + r*35
        text(30, y+5, state, 15)
        for i, sid in enumerate(ids):
            color = '#2563eb' if sid in connected else '#e2e8f0'
            parts.append(f'<circle cx="{290+i*cell}" cy="{y}" r="8" fill="{color}"><title>{escape(state+": "+data["skill_names"][sid]+(" connected" if sid in connected else " not connected"))}</title></circle>')
    used = {sid for ids in rows.values() for sid in ids}
    for i, (sid, name) in enumerate(data['skill_names'].items()):
        text(30, legend_y+i*26, f'{sid}  {name}' + ('  [no model connection]' if sid not in used else ''), 15)
    y = legend_y + len(data['skill_names'])*26 + 25
    text(30, y, 'Engine helpers (including legacy plan APIs; not independent Workflow nodes)', 19)
    for item in data['internal']:
        y += 30
        text(30, y, f'{item["state"]}: {item["method"]}() called by '+', '.join(item['callers']), 15)
    text(30, y+40, 'OBSERVE archives evidence; DECIDE reflects, updates working notes and chooses one action; COMMIT publishes.', 14)
    parts.append('</svg>')
    return '\n'.join(parts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/visualizations/agent')
    args = parser.parse_args()
    data = extract(ROOT)
    svg = render_svg(data)
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / 'architecture.svg').write_text(svg)
    (out / 'architecture.json').write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n')
    details = ''.join(f'<details><summary>{escape(k)}</summary><pre>{escape(v)}</pre></details>' for k,v in data['source_details'].items())
    edge_rows = ''.join(f'<tr><td>{escape(e["from"])}</td><td>→ {escape(e["to"])}</td><td>{escape(e["label"] or "sequence")}</td></tr>' for e in data['edges'])
    html = '<!doctype html><html lang="ja"><meta charset="utf-8"><title>Agent architecture</title><style>body{font:16px system-ui;margin:24px;background:#f8fafc;color:#172554}svg{width:100%;height:auto;min-width:900px}.diagram{overflow:auto}pre{overflow:auto;background:#e2e8f0;padding:16px}summary{cursor:pointer;padding:10px}td{padding:6px 20px}code{overflow-wrap:anywhere}</style>'
    html += '<h1>エージェントの構造とスキル接続</h1><p><code>make visualize</code>で最新ソースから再生成。青い点は利用可能なスキルで、実行済みを意味しません。モデルなしではLLMスキルは使いません。</p>'
    html += '<p>OBSERVEは観測・境界・予算を確認し、DECIDEが振り返りと次の一手をまとめて提出します。COMMITが記憶と操作を確定します。下部の内部処理一覧には旧計画APIの補助関数も含みます。</p>'
    html += '<p>状態の必須指示はinstructions.py、任意の専門手順はSKILL.mdです。スキル名・用途を初回に提示し、必要時だけ本文・資料を読みます。確定・予算制御・描画・操作変換はホストが担当します。</p>'
    html += '<h2>ステート完了ツール</h2><pre>'+escape(json.dumps(data['completion_tools'], ensure_ascii=False, indent=2))+'</pre>'
    html += '<div class="diagram">'+svg+'</div><h2>モデルに渡すツール（ソース抽出）</h2><pre>'+escape(data['model_tools'])+'</pre>'
    html += '<h2>遷移一覧</h2><table>'+edge_rows+'</table><h2>分岐条件・各ステートの処理（現在のソース）</h2>'+details
    html += '<h2>再現情報</h2><p>生成時刻: '+escape(data['generated_utc'])+'</p><pre>'+escape(json.dumps(data['sha256'],indent=2))+'</pre></html>'
    (out / 'index.html').write_text(html)
    print(f'HTML: {out / "index.html"}\nSVG:  {out / "architecture.svg"}\nJSON: {out / "architecture.json"}')


if __name__ == '__main__':
    main()
