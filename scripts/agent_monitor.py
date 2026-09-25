"""Read-only, incremental replay of agent journals. No ADK/model dependencies."""
from __future__ import annotations

import base64
from dataclasses import dataclass
from functools import lru_cache
from html import escape
import json
from pathlib import Path
import struct

JOURNALS = ('observations', 'states', 'requests', 'tools', 'model', 'artifacts', 'execution', 'learning')
PALETTE = ('#FFFFFF', '#CCCCCC', '#999999', '#666666', '#333333', '#000000', '#E53AA3', '#FF7BCC',
           '#F93C31', '#1E93FF', '#88D8F1', '#FFDC00', '#FF851B', '#921231', '#4FCC30', '#A356D6')


@dataclass(frozen=True)
class Run:
    directory: Path
    run_id: str

    @property
    def label(self):
        game = self.directory.parent.name if self.directory.name == 'cognition' else self.directory.name
        return f'{game} · {self.run_id[:10]} · {self.directory.parent.parent.name}'


def discover_runs(root):
    """A benchmark root, cognition directory, or a journal may be supplied."""
    root = Path(root).expanduser().resolve()
    if root.is_file():
        run_id = root.name.split('.')[0]
        return [Run(root.parent, run_id)] if (root.parent / f'{run_id}.states.jsonl').is_file() else []
    if not root.exists():
        return []
    paths = list(root.rglob('*.states.jsonl'))
    paths.sort(key=lambda p: (p.stat().st_mtime_ns, str(p)))
    return [Run(p.parent, p.name.removesuffix('.states.jsonl')) for p in paths]



class Journal:
    """Never consume an incomplete line. Truncation/replacement resets this file."""
    def __init__(self, path):
        self.path, self.offset, self.identity = Path(path), 0, None
        self.rows, self.invalid_lines = [], 0

    def refresh(self):
        try:
            stat = self.path.stat()
        except FileNotFoundError:
            return False
        identity = (stat.st_dev, stat.st_ino)
        reset = self.identity != identity or stat.st_size < self.offset
        if reset:
            self.rows, self.offset, self.invalid_lines = [], 0, 0
        self.identity = identity
        changed = reset
        with self.path.open('rb') as stream:
            stream.seek(self.offset)
            while True:
                start = stream.tell()
                line = stream.readline()
                if not line or not line.endswith(b'\n'):
                    self.offset = start
                    break
                self.offset = stream.tell()
                try:
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        raise ValueError('not an object')
                except (ValueError, UnicodeDecodeError):
                    self.invalid_lines += 1
                    continue
                self.rows.append(row)
                changed = True
        return changed


class Timeline:
    def __init__(self, run):
        self.run = run
        self.journals = {k: Journal(run.directory / f'{run.run_id}.{k}.jsonl') for k in JOURNALS}
        self.events = []

    def refresh(self):
        changed = [j.refresh() for j in self.journals.values()]
        if any(changed) or not self.events:
            rows = [dict(row, _journal=kind) for kind, journal in self.journals.items() for row in journal.rows]
            if any(type(r.get('sequence')) is not int or r['sequence'] < 1 for r in rows):
                raise ValueError('Every event must have a positive sequence number')
            if len({r['sequence'] for r in rows}) != len(rows):
                raise ValueError('Duplicate event sequence number')
            self.events = sorted(rows, key=lambda r: r['sequence'])

        return any(changed)

    def snapshot(self, index):
        if not self.events:
            return None
        index = max(0, min(index, len(self.events)-1))
        prefix = self.events[:index+1]
        selected = prefix[-1]
        observations, tools, learning = [], [], []
        state, stage_input, stage_output = '観測待ち', None, None
        phase, context, action, action_step, action_status = '', {}, None, None, '未選択'
        prediction = None
        incoming_action, incoming_status = None, None
        request = response = None
        for event in prefix:
            kind, name = event['_journal'], event.get('event')
            if kind == 'observations':
                incoming_action = action if action_step is not None and event.get('step') == action_step+1 else None
                incoming_status = action_status if incoming_action else None
                observations.append(event)
                state, phase, stage_input, stage_output = '観測受領', '', None, None
            if kind == 'states':
                if name == 'runtime_closed':
                    state, phase = '終了', event.get('stop_reason') or 'closed'
                else:
                    state = event['state']
                    phase = '処理中' if name == 'state_entered' else '完了'
                    if name == 'state_entered':
                        stage_input, stage_output = event.get('input'), None
                        if state == 'DECIDE':
                            context = stage_input or {}
                    else:
                        stage_output = event.get('output')
                        if state == 'RUN' and isinstance(stage_output, dict):
                            context = dict(context, **{k: stage_output[k] for k in ('task','hypotheses') if k in stage_output})
                            if 'summary' in stage_output:
                                context['previous_summary'] = stage_output['summary']
            if kind == 'requests':
                request = event
            if kind == 'model':
                response = event
            if name == 'action_selected':
                pending = event.get('pending', {})
                action, action_step = pending.get('action'), event.get('step')
                prediction = pending.get('prediction')
                action_status = '選択済み・未送信'
            if kind == 'execution':
                action, action_step = event.get('action'), event.get('step')
                action_status = {'action_dispatched': '送信済み・受付待ち',
                                 'action_acknowledged': '受付済み・次の観測待ち',
                                 'action_outcome_unknown': '実行結果不明'}.get(name, name)
            if kind == 'tools':
                tools.append(event)
            if kind == 'learning':
                learning.append(event)
        current = observations[-1] if observations else None
        before = observations[-2] if len(observations) > 1 else None
        if current and action_step is not None and current.get('step', -1) > action_step:
            if action_status.startswith('受付済み'):
                action_status = '受付済み・次の観測あり（効果の成功とは別）'
        return dict(event=selected, index=index, total=len(self.events), current=current, before=before,
                    state=state, phase=phase, input=stage_input, output=stage_output,
                    context=context, action=action, action_step=action_step, action_status=action_status,
                    prediction=prediction,
                    incoming_action=incoming_action, incoming_status=incoming_status,
                    request=request, response=response, tools=tools[-16:], learning=learning[-12:],
                    recent=prefix[-12:])


def safe_asset(directory, relative):
    path = (directory / relative).resolve()
    if not path.is_relative_to(directory.resolve()):
        raise ValueError('image path escapes the log directory')
    return path


@lru_cache(maxsize=96)
def png_asset(path, mtime_ns):
    data = Path(path).read_bytes()
    if data[:8] != b'\x89PNG\r\n\x1a\n':
        raise ValueError('not PNG')
    width, height = struct.unpack('>II', data[16:24])
    return base64.b64encode(data).decode(), width, height


def screen_html(observation, directory, action=None):
    if observation is None:
        return '<p>この時点の画面記録はありません。</p>'
    overlay = ''
    grid = observation.get('grid')
    body = ''
    if observation.get('image_path'):
        try:
            path = safe_asset(directory, observation['image_path'])
            data, width, height = png_asset(str(path), path.stat().st_mtime_ns)
            body = f'<image width="{width}" height="{height}" href="data:image/png;base64,{data}"/>'
            view = observation.get('viewport') or {}
            origin, scale = view.get('origin'), view.get('scale')
        except (OSError, ValueError, struct.error):
            body = ''
    if not body and grid:
        height, width = len(grid), len(grid[0])
        origin, scale = (0, 0), 1
        body = ''.join(f'<rect x="{x}" y="{y}" width="1" height="1" fill="{PALETTE[v]}"/>'
                       for y, row in enumerate(grid) for x, v in enumerate(row) if type(v) is int and 0 <= v < 16)
    if not body:
        return '<p>画像ファイル／色IDグリッドが見つかりません。</p>'
    # Draw only when the original-pixel-to-image transform is recorded.
    if action and action.get('action') in ('CLICK', 'ACTION6') and origin is not None and scale:
        x, y = action.get('x'), action.get('y')
        if type(x) is int and type(y) is int:
            cx, cy = origin[0]+(x+.5)*scale, origin[1]+(y+.5)*scale
            r = max(1, 1.8*scale)
            overlay = (f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="#fff" stroke-width="{max(.2, scale*.55)}"/>'
                       f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="#ff315b" stroke-width="{max(.1, scale*.25)}"/>')
    return (f'<svg role="img" aria-label="記録されたゲーム画面" viewBox="0 0 {width} {height}" '
            f'style="width:100%;max-height:420px;image-rendering:pixelated;background:#18202b" '
            f'xmlns="http://www.w3.org/2000/svg">{body}{overlay}</svg>')


def pretty(value):
    return json.dumps(value, ensure_ascii=False, indent=2) if not isinstance(value, str) else value


def json_html(value):
    return '<pre style="white-space:pre-wrap;overflow-wrap:anywhere;max-height:480px;overflow:auto">'+escape(pretty(value))+'</pre>'


def dashboard_html(snapshot, directory):
    if snapshot is None:
        return '<p>ログを待っています。推論サーバの確認・評価用ソースの準備中は画面がまだ届きません。</p>'
    s = snapshot
    e = s['event']
    action = s['action'] or {}
    action_label = action.get('action', '—')
    buttons = dict(zip(('ACTION1','ACTION2','ACTION3','ACTION4','ACTION5','ACTION6','ACTION7'),
                       ('UP','DOWN','LEFT','RIGHT','ACT','CLICK','UNDO')))
    if action_label in buttons:
        action_label = buttons[action_label] + ' / ' + action_label
    if 'x' in action:
        action_label += f" ({action.get('x')}, {action.get('y')})"
    cards = []
    for label, obs in [('直前の観測', s['before']), ('この時点の最新観測', s['current'])]:
        mark = action if obs and obs.get('step') == s['action_step'] else None
        if obs is s['before'] and s['incoming_action']:
            mark = s['incoming_action']
        cards.append('<div style="flex:1;min-width:220px"><b>'+label+'</b> · step '+
                     escape(str(obs.get('step') if obs else '—'))+screen_html(obs, directory, mark)+'</div>')
    badges = ' → '.join('<span style="padding:6px 14px;border-radius:5px;background:'+('#2563eb;color:white' if s['state']==state else '#e2e8f0;color:#334155')+'">'+escape(str(state))+'</span>' for state in ('DECIDE','RUN'))
    recent = ''.join('<tr><td>'+escape(str(r.get('step', '—')))+'</td><td>'+escape(r.get('state', ''))+'</td><td>'+escape(str(r.get('event', '')))+' '+escape(str(r.get('tool', '')))+'</td></tr>' for r in s['recent'])
    context = s['context']
    task = context.get('task', '記録なし')
    summary = context.get('previous_summary', context.get('summary', ''))
    incoming = s['incoming_action'] or {}
    incoming_name = incoming.get('action','記録なし')
    incoming_label = buttons.get(incoming_name,incoming_name)
    if 'x' in incoming:
        incoming_label += f" ({incoming.get('x')}, {incoming.get('y')})"
    incoming_status = s['incoming_status'] or ''
    if incoming_status.startswith('受付済み'):
        incoming_status = '受付済み'
    current, before = s['current'] or {}, s['before'] or {}
    a,b = before.get('grid'), current.get('grid')
    delta = '比較できる前後の色IDグリッドなし'
    if current.get('full_reset') or (before and current.get('levels_completed') != before.get('levels_completed')):
        delta = 'リセット／レベル境界のため通常の差分比較を保留'
    elif a and b and len(a)==len(b) and all(len(x)==len(y) for x,y in zip(a,b)):
        delta = str(sum(v!=w for x,y in zip(a,b) for v,w in zip(x,y)))+' セルが変化（成功判定ではありません）'
    return f'''<div style="font:14px system-ui;color:#172554;background:#f8fafc;padding:16px;border-radius:10px">
    <div style="display:flex;flex-wrap:wrap;justify-content:space-between;gap:16px"><b>{escape(str(e.get('game_id', 'ゲーム')))} · step {escape(str(e.get('step', '—')))}</b><div>{badges}</div></div>
    <p><b>{escape(s['state'])} · {escape(s['phase'])}</b> · イベント {s['index']+1}/{s['total']} · {escape(str(e.get('timestamp', '時刻なし')))}</p>
    <div style="display:flex;flex-wrap:wrap;gap:18px">{''.join(cards)}</div>
    <p><b>前画面に対する操作記録:</b> {escape(incoming_label)} · {escape(incoming_status)}<br>
    <b>この時点の選択操作（step {escape(str(s['action_step']))}）: {escape(action_label)}</b> · {escape(s['action_status'])}<br>赤丸は各画面に対して選択したクリック位置です。</p>
    <p><b>選択した操作の予測:</b> {escape(str(s['prediction'] or '記録なし'))}<br><b>表示中の前後差分:</b> {escape(delta)}</p>
    <p><b>保持している課題:</b> {escape(str(task))}<br><b>認識メモ:</b> {escape(str(summary)) or '更新なし／記録なし'}</p>
    <details><summary>直近のイベント</summary><table style="text-align:left;width:100%"><tr><th>step</th><th>状態</th><th>イベント／ツール</th></tr>{recent}</table></details></div>'''
