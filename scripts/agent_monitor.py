"""Read-only playback of saved agent journals. No ADK/model dependencies."""
from __future__ import annotations

import base64
from dataclasses import dataclass
from functools import lru_cache
from html import escape
import json
from pathlib import Path
import struct

JOURNALS = ('observations', 'states', 'requests', 'tools', 'model', 'artifacts', 'execution', 'learning', 'notebook', 'experiments')
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


def discover_evaluations(root):
    """List benchmark IDs with a finished, replayable game; never scan frame trees."""
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        return []
    candidates = [root] if (root/'manifest.json').is_file() else [p for p in root.iterdir() if p.is_dir()]
    if (root/'evaluations').is_dir():
        candidates += [p for p in (root/'evaluations').iterdir() if p.is_dir()]
    return sorted((p for p in candidates if (p/'manifest.json').is_file() and discover_runs(p)),
                  key=lambda p: p.name, reverse=True)


def discover_runs(evaluation):
    """Read only the shallow benchmark/game/cognition structure."""
    evaluation = Path(evaluation)
    return [Run(p.parent, p.name.removesuffix('.states.jsonl'))
            for p in sorted(evaluation.glob('*/cognition/*.states.jsonl'))
            if (p.parent.parent/'result.json').is_file()]


def read_journal(path):
    rows, invalid = [], 0
    if not path.exists():
        return rows, invalid
    with path.open('rb') as stream:
        for line in stream:
            if not line.endswith(b'\n'):
                break  # Preserve an interrupted run without inventing a completed event.
            try:
                row = json.loads(line)
                if not isinstance(row,dict):
                    raise ValueError('not an object')
                rows.append(row)
            except (ValueError,UnicodeDecodeError):
                invalid += 1
    return rows, invalid


class Timeline:
    def __init__(self, run):
        self.run = run
        self.events, self.snapshots = [], []
        self.invalid_lines = 0

    def load(self):
        """Load this game once and build the replay index in a single pass."""
        rows = []
        self.invalid_lines = 0
        for kind in JOURNALS:
            records, invalid = read_journal(self.run.directory/f'{self.run.run_id}.{kind}.jsonl')
            rows.extend(dict(row,_journal=kind) for row in records)
            self.invalid_lines += invalid
        if any(type(r.get('sequence')) is not int or r['sequence'] < 1 for r in rows):
            raise ValueError('Every event must have a positive sequence number')
        if len({r['sequence'] for r in rows}) != len(rows):
            raise ValueError('Duplicate event sequence number')
        self.events = sorted(rows,key=lambda r:r['sequence'])
        self.snapshots = list(self._index())
        return self

    def snapshot(self, index):
        if not self.snapshots:
            return None
        return self.snapshots[max(0,min(index,len(self.snapshots)-1))]

    def _index(self):
        observations, tools, learning = [], [], []
        state, stage_input, stage_output = '観測待ち', None, None
        phase, context, action, action_step, action_status = '', {}, None, None, '未選択'
        prediction = None
        experiment = review = None
        incoming_action, incoming_status = None, None
        request = response = None
        notebook_view = notebook_read = notebook_change = None
        notebook_opening_event = None
        notebook_reads, notebook_changes = (), ()
        notebook_events = []
        host_judgments = []
        machine = {'node': 'observe', 'phase': '初期観測待ち', 'work': None, 'job': None}
        for index,event in enumerate(self.events):
            kind, name = event['_journal'], event.get('event')
            if kind == 'observations':
                host_judgments = []
                notebook_view = notebook_read = notebook_change = notebook_opening_event = None
                notebook_reads, notebook_changes = (), ()
                incoming_action = action if action_step is not None and event.get('step') == action_step+1 else None
                incoming_status = action_status if incoming_action else None
                observations.append(event)
                state, phase, stage_input, stage_output = '観測受領', '', None, None
                machine = {'node': 'observe', 'phase': '観測受領', 'work': None, 'job': None}
            if kind == 'states':
                if name == 'runtime_closed':
                    state, phase = '終了', event.get('stop_reason') or 'closed'
                    machine = dict(machine, node='end', phase=phase)
                else:
                    state = event['state']
                    phase = '処理中' if name == 'state_entered' else '完了'
                    if name == 'state_entered':
                        stage_input, stage_output = event.get('input'), None
                        work = event.get('work')
                        job = (stage_input.get('job') or {}) if isinstance(stage_input, dict) else {}
                        job_kind = job.get('kind') or ('propose_skill' if 'spec' in job else 'submit_review' if 'verdict' in job else 'redesign' if 'blocked_action' in job else 'defer_skill' if 'reason' in job else None)
                        machine = {'node': ('build' if work == 'skill_creation' else
                                            work if work in ('attend','interpret_world','select_goal','assess_goal','design_experiment','inspect_target','judge_effect','choose_method','resolve_arguments') else
                                            'design' if work == 'experiment_design' else
                                            'review' if work == 'experiment_review' else 'decide')
                                   if state == 'DECIDE' else 'run',
                                   'phase': phase, 'work': work, 'job': job_kind}
                        if state == 'DECIDE':
                            context = stage_input if isinstance(stage_input, dict) else {}
                            notebook_view = context.get('notebook')
                            notebook_opening_event = event
                            notebook_read, notebook_reads = None, ()
                    else:
                        stage_output = event.get('output')
                        # RUN may change work for the NEXT invocation. Keep the
                        # work captured on entry until the next state is entered.
                        machine = dict(machine, phase=phase)
                        if state == 'RUN' and isinstance(stage_output, dict):
                            if 'notebook' in stage_output:
                                context = dict(context, notebook=stage_output['notebook'])
                            result = stage_output.get('result') or {}
                            if result.get('status') == 'stop':
                                machine = dict(machine, node='end', phase=result.get('reason', 'stop'))
                            elif result.get('status') == 'action':
                                machine = dict(machine, node='wait', phase='操作選択済み・未送信')
            if kind == 'experiments':
                if name in ('experiment_started', 'experiment_observed', 'experiment_review_requested'):
                    experiment = event.get('experiment')
                elif name in ('experiment_reviewed', 'experiment_interrupted'):
                    experiment = event.get('experiment')
                    if name == 'experiment_reviewed':
                        review = experiment
            if kind == 'notebook':
                notebook_events.append(event)
                if name == 'notebook_opened':
                    notebook_view = event.get('view')
                    notebook_opening_event = event
                    notebook_read, notebook_reads = None, ()
                elif name == 'task_opened':
                    context = event.get('task', {})
                    notebook_view = event.get('human_view')
                    notebook_opening_event = event
                    notebook_read, notebook_reads = None, ()
                elif name == 'notebook_read':
                    notebook_read = event
                    notebook_reads += (event,)
                elif name in ('note_changed', 'bookmark_changed', 'notebook_boundary'):
                    notebook_change = event
                    notebook_changes += (event,)
            if kind == 'artifacts' and name in ('target_checked','task_rejected','plan_delta'):
                host_judgments.append(event)
            if name == 'machine_transition':
                machine = dict(machine,node=event['target'],phase=event.get('condition',''))
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
                machine = dict(machine, node='wait', phase=action_status)
            if kind == 'tools':
                tools.append(event)
            if kind == 'learning':
                learning.append(event)
            current = observations[-1] if observations else None
            before = observations[-2] if len(observations) > 1 else None
            display_status = action_status
            if current and action_step is not None and current.get('step', -1) > action_step:
                if action_status.startswith('受付済み'):
                    display_status = '受付済み・次の観測あり（効果の成功とは別）'
            yield dict(event=event, index=index, total=len(self.events), current=current, before=before,
                        state=state, phase=phase, input=stage_input, output=stage_output,
                        machine=machine, host_judgments=host_judgments[-8:],
                        context=context, action=action, action_step=action_step, action_status=display_status,
                        prediction=prediction, experiment=experiment, review=review,
                        incoming_action=incoming_action, incoming_status=incoming_status,
                        request=request, response=response, tools=tools[-16:], learning=learning[-12:],
                        notebook={'opening': notebook_view, 'last_read': notebook_read,
                                  'opening_event': notebook_opening_event, 'reads': notebook_reads,
                                  'changes': notebook_changes,
                                  'last_change': notebook_change, 'recent': notebook_events[-12:]},
                        recent=self.events[max(0,index-11):index+1])


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


def dashboard_html(snapshot, directory, *, include_images=True):
    if snapshot is None:
        return '<p>評価IDとゲームを選択して「読み込む」を押してください。</p>'
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
    for label, obs in ([('直前の観測', s['before']), ('この時点の最新観測', s['current'])] if include_images else []):
        mark = action if obs and obs.get('step') == s['action_step'] else None
        if obs is s['before'] and s['incoming_action']:
            mark = s['incoming_action']
        cards.append('<div style="flex:1;min-width:220px"><b>'+label+'</b> · step '+
                     escape(str(obs.get('step') if obs else '—'))+screen_html(obs, directory, mark)+'</div>')
    recent = ''.join('<tr><td>'+escape(str(r.get('step', '—')))+'</td><td>'+escape(r.get('state', ''))+'</td><td>'+escape(str(r.get('event', '')))+' '+escape(str(r.get('tool', '')))+'</td></tr>' for r in s['recent'])
    context = s['context']
    notebook = context.get('notebook') or {}
    task = (notebook.get('goal') or {}).get('text', '記録なし')
    goal_path = ' → '.join(p['text'] for p in notebook.get('goal_path', []))
    summary = ' / '.join(b['name']+': '+b['title'] for b in notebook.get('bookmarks', []))
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
    experiment_panel = experiment_html(s)
    if context.get('work') == 'attend':
        memory_panel = attention_html(s)
    else:
        memory_panel = (f'<p><b>攻略ノートの目標:</b> {escape(str(task))}<br>'
            f'<b>現在の目標経路:</b> {escape(goal_path) or "記録なし"}<br>'
            f'<b>攻略ノートのしおり:</b> {escape(str(summary)) or "更新なし／記録なし"}</p>')
    return f'''<div style="font:14px system-ui;color:#172554;background:#f8fafc;padding:16px;border-radius:10px">
    <b>{escape(str(e.get('game_id', 'ゲーム')))} · step {escape(str(e.get('step', '—')))}</b>
    <p>実行ログ: <b>{escape(s['state'])} · {escape(s['phase'])}</b> · イベント {s['index']+1}/{s['total']} · {escape(str(e.get('timestamp', '時刻なし')))}</p>
    <div style="display:flex;flex-wrap:wrap;gap:18px">{''.join(cards)}</div>
    <p><b>前画面に対する操作記録:</b> {escape(incoming_label)} · {escape(incoming_status)}<br>
    <b>この時点の選択操作（step {escape(str(s['action_step']))}）: {escape(action_label)}</b> · {escape(s['action_status'])}<br>赤丸は各画面に対して選択したクリック位置です。</p>
    <p><b>選択した操作の予測:</b> {escape(str(s['prediction'] or '記録なし'))}<br><b>表示中の前後差分:</b> {escape(delta)}</p>
    {memory_panel}
    {focused_task_html(context)}
    {host_judgments_html(s.get('host_judgments', []))}
    {experiment_panel}
    <details><summary>直近のイベント</summary><table style="text-align:left;width:100%"><tr><th>step</th><th>状態</th><th>イベント／ツール</th></tr>{recent}</table></details></div>'''


def attention_html(snapshot):
    context = snapshot['context']
    response = snapshot.get('response') or {}
    decision = {}
    if response.get('step') == (snapshot.get('current') or {}).get('step') and response.get('schema_valid'):
        try:
            decision = json.loads(response.get('response') or '{}')
        except (TypeError, ValueError):
            pass
    rows = [('入力メモ（仮説）', context.get('working_notes_hypotheses', '')),
            ('直前の実測結果', context.get('last_result')),
            ('モデルの結果解釈', decision.get('interpretation')),
            ('今回の注目対象', decision.get('focus')),
            ('次に確かめること', decision.get('expectation')),
            ('更新した短い記憶（仮説）', decision.get('notes'))]
    body = ''.join('<tr><th style="text-align:left;vertical-align:top">'+escape(label)+
                   '</th><td>'+escape(pretty(value))+'</td></tr>' for label, value in rows if value is not None)
    return '<h4>画面からの注意と操作</h4><table>'+body+'</table><details><summary>最近の試行・同条件の無反応</summary>'+json_html({
        'recent_trials': context.get('recent_trials', []),
        'unchanged_state_trials': context.get('unchanged_state_trials', [])})+'</details>'


def experiment_html(snapshot):
    """Render only experiment versions already present at this replay position."""
    parts = []
    labels = {'supported': '支持', 'unsupported': '不支持', 'inconclusive': '判定不能'}
    seen = set()
    for label, page in [('現在／直近の実験', snapshot.get('experiment')),
                        ('直近の結果判定', snapshot.get('review'))]:
        if not page or (page['id'], page['revision']) in seen:
            continue
        seen.add((page['id'], page['revision']))
        data = page['data']; plan = data['plan']; verdict = data.get('review')
        rows = [('小目標', plan['subgoal']['text']), ('問い', plan['question']),
                ('仮説と条件', plan['hypothesis'] + ' / ' + plan['conditions']),
                ('操作前の予測', plan['expected']['description']),
                ('観測範囲', plan['expected'].get('region')),
                ('状態', data['status'])]
        revision = plan.get('revision')
        if revision:
            rows += [('再設計の根拠', f'{revision["experiment_id"]} v{revision["review_revision"]}'),
                     ('見直した前提', revision['reconsidered_assumption']),
                     ('変更したもの', revision['change']), ('変更の理由', revision['reason'])]
        if verdict:
            rows += [('実測', data.get('measurement')),
                     ('判定', labels.get(verdict['verdict'], verdict['verdict']) + ' / ' + verdict['finding']),
                     ('小目標の状態', verdict.get('subgoal_status', '別タスクで評価')), ('次への更新', verdict.get('update', '小目標の評価に渡す'))]
            understanding = verdict.get('understanding') or {}
            rows += [(label, understanding[key]) for key, label in (
                ('reconsider_assumption', '見直す前提'), ('open_question', '残る疑問'),
                ('subgoal_reason', '小目標の扱いの理由')) if key in understanding]
        if data.get('interruption'):
            rows += [('未判定の理由', data['interruption'])]
        cells = ''.join('<tr><th style="text-align:left;vertical-align:top;min-width:100px">' + escape(k) +
                        '</th><td>' + escape(pretty(v)) + '</td></tr>' for k, v in rows)
        parts.append('<p><b>' + label + ' · ' + escape(page['id']) + '</b></p><table>' + cells + '</table>')
    return ''.join(parts)


def focused_task_html(context):
    """Show the actual limited input, separately from the human notebook."""
    if not context.get('task_id'):
        return ''
    labels = {'task_id':'タスクID', 'work':'担当', 'observation_id':'観測', 'goal':'固定された小目標',
        'parent_goal':'親目標', 'previous_goal':'直前の小目標', 'question':'今回の問い', 'fact':'実験の事実',
        'latest_fact':'直近の事実', 'rejection':'差し戻し理由', 'target':'確認する対象', 'action':'提案された操作',
        'target_assessment':'対象確認の結果', 'last_execution':'直近の実行結果'}
    rows = ''.join('<tr><th style="text-align:left;vertical-align:top">'+escape(labels.get(k,k))+
                   '</th><td style="white-space:pre-wrap">'+escape(pretty(v))+'</td></tr>'
                   for k,v in context.items() if k != 'notebook' and v is not None)
    return '<details open><summary>この判断に渡した入力（攻略ノート全体とは別）</summary><table>'+rows+'</table></details>'


def host_judgments_html(events):
    labels={'target_checked':'対象矩形とクリック位置の照合', 'task_rejected':'差し戻し', 'plan_delta':'実験の変更点（ホスト比較）'}
    rows=''.join('<li><b>'+escape(labels[e['event']])+'</b> '+escape(pretty({k:v for k,v in e.items()
        if k in ('verdict','region','reason','details','recovery','changes')}))+'</li>' for e in events)
    return '<details open><summary>この観測でのホスト判定</summary><ul>'+rows+'</ul></details>' if rows else ''
