"""Readable notebook views from the exact versions in a replay snapshot.

No model calls, current-file lookups, Markdown execution or inferred summaries.
Labels are translated; the agent's recorded text stays verbatim.
"""
from html import escape

KINDS = {'goal': '大目標', 'subgoal': '小目標', 'hypothesis': '仮説', 'plan': '計画',
         'interpretation': '解釈', 'experiment': '実験', 'result': '観測結果'}
WORK = {'select_goal':'小目標選択', 'assess_goal':'小目標評価', 'design_experiment':'実験設計',
        'inspect_target':'対象確認', 'judge_effect':'効果判定', 'choose_method':'方法選択',
        'resolve_arguments':'引数決定', 'experiment_design': '実験設計', 'experiment_review': '結果判定', 'skill_creation': 'スキル作成'}
STATUS = {'active': '進行中', 'open': '有効', 'completed': '完了', 'abandoned': '取り下げ',
          'withdrawn': '撤回済み', 'awaiting_observation': '操作結果待ち',
          'awaiting_review': '判定待ち', 'reviewed': '判定済み', 'interrupted': '中断'}
VERDICT = {'supported': '支持', 'unsupported': '不支持', 'inconclusive': '判定不能'}
NEXT = {'continue': '継続', 'revise': '見直す', 'learn': 'スキル化を検討', 'stop': '停止'}
BOOKMARKS = {'current_goal': '現在の大目標', 'current_subgoal': '現在の小目標',
             'active_experiment': '進行中の実験', 'latest_review': '直近の判定', 'latest_result': '最新の観測結果'}
BUTTONS = dict(zip(('ACTION1','ACTION2','ACTION3','ACTION4','ACTION5','ACTION6','ACTION7'),
                   ('UP','DOWN','LEFT','RIGHT','ACT','CLICK','UNDO')))


def _text(value):
    return escape(str(value)) if value is not None else '記録なし'


def _field(label, value):
    if value is None or value == '':
        return ''
    return f'<div class="nbv-field"><dt>{_text(label)}</dt><dd>{_text(value)}</dd></div>'


def _region(region):
    if not region:
        return None
    return f'左上 ({region.get("x")}, {region.get("y")}) ・ 幅 {region.get("width")} × 高さ {region.get("height")}'


def _action(action):
    if not action:
        return None
    name = action.get('action', '記録なし')
    label = BUTTONS.get(name, name)
    if action.get('x') is not None:
        label += f' ({action["x"]}, {action.get("y")})'
    return label


def _review(data):
    verdict = data.get('review') or {}
    if not verdict:
        return ''
    fields = [_field('判定', VERDICT.get(verdict.get('verdict'), verdict.get('verdict'))),
              _field('分かったこと', verdict.get('finding')),
              _field('小目標の状態', STATUS.get(verdict.get('subgoal_status'), verdict.get('subgoal_status'))),
              _field('次の方針', NEXT.get(verdict.get('next_step'), verdict.get('next_step'))),
              _field('次の判断へ引き継ぐこと', verdict.get('update')),
              _field('判定した担当', {'host': 'システムの測定', 'agent': 'エージェントの解釈'}.get(data.get('reviewer')))]
    if verdict.get('evidence_ids'):
        fields.append(_field('判定の根拠', ' ・ '.join(verdict['evidence_ids'])))
    understanding = verdict.get('understanding') or {}
    fields += [_field('見直す前提', understanding.get('reconsider_assumption')),
               _field('残る疑問', understanding.get('open_question')),
               _field('小目標の扱いの理由', understanding.get('subgoal_reason'))]
    return ''.join(fields)


def _measurement(data):
    measured = data.get('measurement') or {}
    fields = [_field('測定した範囲', _region(measured.get('region'))),
              _field('範囲内で変わったセル数', measured.get('changed_cells_in_region'))]
    if measured.get('before_level') is not None or measured.get('after_level') is not None:
        fields.append(_field('到達レベル', f'{measured.get("before_level", "不明")} → {measured.get("after_level", "不明")}'))
    return ''.join(fields)


def page_html(page, *, label=None, segment=None):
    """Render full pages and intentionally abbreviated opening-view verdicts."""
    if not page:
        return '<p class="nbv-muted">この時点の記録はありません。</p>'
    data = page.get('data') or {}
    plan = data.get('plan') or {}
    kind = page.get('kind')
    title = label or page.get('title') or ('直近の判定' if data.get('review') else 'ノート')
    badge = KINDS.get(kind, '判定の要点' if data.get('review') else 'ノート')
    state = data.get('status') or page.get('status')
    if state:
        badge += ' · ' + STATUS.get(state, state)
    if segment is not None and page.get('segment', segment) != segment:
        badge += ' · 前の区間の記録'
    body = ''
    if kind in ('goal', 'subgoal', 'hypothesis', 'plan', 'interpretation') or not data:
        body += _field('内容', page.get('text'))
    if kind == 'subgoal':
        body += _field('完了の条件', data.get('done_when'))
        body += _field('親の目標', data.get('parent_id'))
        body += _field('直近の更新', data.get('update'))
        body += _field('小目標の扱いの理由', data.get('status_reason'))
    if plan:
        body += _field('小目標', (plan.get('subgoal') or {}).get('text'))
        body += _field('確かめたい問い', plan.get('question'))
        body += _field('仮説', plan.get('hypothesis'))
        body += _field('この仮説の条件', plan.get('conditions'))
        body += _field('試す操作', _action(data.get('action')))
    expected = plan.get('expected') or page.get('expected') or {}
    if expected:
        if not plan:
            body += _field('確かめた問い', page.get('question'))
        body += _field('操作前の予測', expected.get('description'))
        body += _field('予測を確かめる範囲', _region(expected.get('region')))
        body += _field('判定方法', {'region_changed': '指定範囲の画素変化', 'level_increased': '到達レベルの増加',
                                   'semantic': '観測内容の意味を比較'}.get(expected.get('kind')))
    if plan:
        body += _field('前提条件を見る範囲', _region(plan.get('context_region')))
        body += _field('試行回数', f'{data.get("attempt", 1)} / {plan.get("max_attempts", 1)} 回')
        body += _field('繰り返す理由', plan.get('repeat_reason'))
        body += _field('再試行の元', plan.get('retry_of'))
        revision = plan.get('revision') or {}
        if revision:
            body += _field('再設計の根拠', f'{revision.get("experiment_id")} · 第{revision.get("review_revision")}版')
            body += _field('変更したもの', {'action': '操作・対象座標', 'observation_scope': '観測範囲',
                                          'context': '前提条件・その観測'}.get(revision.get('change')))
            body += _field('見直した前提', revision.get('reconsidered_assumption'))
            body += _field('変更で疑問を解消できる理由', revision.get('reason'))
    body += _measurement(data) + _review(data)
    outcome = data.get('outcome') or {}
    if outcome:
        body += _field('実際の操作', _action(outcome.get('action')))
        ack = outcome.get('acknowledged')
        body += _field('操作の受付', '確認済み' if ack is True else '確認できず' if ack is False else '記録なし')
        body += _field('画面全体の変化', f'{outcome["changed_cell_count"]} セル（仮説の支持とは別）'
                       if 'changed_cell_count' in outcome else None)
        body += _field('実経験', outcome.get('experience_id'))
        body += _field('区間の切り替わり', {'level': '次のレベル', 'reset': 'リセット'}.get(outcome.get('boundary')))
    body += _field('見直しが必要な理由', data.get('review_request'))
    body += _field('中断した理由', data.get('interruption'))
    body += _field('撤回した理由', page.get('reason'))
    if kind == 'result' and not outcome:
        body += _field('観測', '初期観測、または対応する操作結果の記録なし')
    if page.get('evidence_ids'):
        body += _field('参照した根拠', ' ・ '.join(page['evidence_ids']))
    version = f'{page.get("id", "IDなし")} · 第{page.get("revision", "?")}版'
    author = {'host': 'システム記録', 'agent': 'エージェントの記述'}.get(page.get('author'), '')
    return (f'<article class="nbv-card"><div class="nbv-badge">{_text(badge)}</div>'
            f'<h4>{_text(title)}</h4><dl>{body}</dl>'
            f'<footer>{_text(version)}{(" · " + _text(author)) if author else ""}</footer></article>')


def _reads(events, segment):
    parts = []
    for event in events:
        result = event.get('result') or {}
        if result.get('page'):
            parts.append('<p class="nbv-muted">参照先：' + _text(event.get('reference')) + '</p>'
                         + page_html(result['page'], segment=segment))
        else:
            parts.append('<p>索引を検索（本文の参照ではありません） · 検索語：'
                         + _text(event.get('query') or '指定なし') + '</p>')
            notes = result.get('notes') or []
            if not notes:
                parts.append('<p class="nbv-muted">該当するノートなし。</p>')
            for note in notes:
                parts.append('<article class="nbv-card"><h4>' + _text(note.get('title'))
                             + '</h4><p>' + _text(note.get('preview')) + '</p><footer>'
                             + _text(note.get('id')) + ' · 第' + _text(note.get('revision')) + '版 · 抜粋のみ</footer></article>')
            if result.get('next_offset') is not None:
                parts.append('<p class="nbv-muted">検索結果には続きがあります。続きの本文はまだ提示されていません。</p>')
    return ''.join(parts) or '<p class="nbv-muted">この呼び出しでは、まだノートを追加で参照していません。</p>'


def _changes(events, segment):
    parts = []
    for event in events:
        if event['event'] == 'note_changed':
            before, after = event.get('before'), event.get('after') or {}
            description = ('撤回' if after.get('status') == 'withdrawn' else '改訂' if before else '作成')
            versions = f'第{before["revision"]}版 → 第{after.get("revision")}版' if before else f'第{after.get("revision")}版'
            parts.append('<details class="nbv-change"><summary>' + _text(description + '：' + after.get('title', after.get('id', 'ノート')) + ' · ' + versions)
                         + '</summary><div class="nbv-columns">'
                         + ('<div><h4>変更前</h4>' + page_html(before, segment=segment) + '</div>' if before else '')
                         + '<div><h4>変更後</h4>' + page_html(after, segment=segment) + '</div></div></details>')
        elif event['event'] == 'bookmark_changed':
            name = BOOKMARKS.get(event.get('name'), event.get('name'))
            parts.append('<p class="nbv-muted">しおり「' + _text(name) + '」 → '
                         + _text(event.get('after') or '解除') + '</p>')
        elif event['event'] == 'notebook_boundary':
            parts.append('<p>レベル／リセットの境界：局所的な目標としおりを切り替えました。</p>')
    return ''.join(parts) or '<p class="nbv-muted">このステップでは、まだノートの変更は記録されていません。</p>'


def notebook_html(snapshot):
    """Show received content separately from saved changes and bookmark indexes."""
    note = snapshot['notebook']
    opening = note.get('opening') or {}
    origin = note.get('opening_event') or {}
    segment = opening.get('segment')
    shown = origin.get('event') == 'notebook_opened'
    human_only = origin.get('event') == 'task_opened'
    work = WORK.get(origin.get('work'), origin.get('work') or '担当の記録なし')
    step = snapshot['event'].get('step', '—')
    intro = ('人間用の参照ビュー（LLMにはタスクの限定入力だけを提示）' if human_only else 'LLM呼び出し開始時の提示内容' if shown else '処理開始時のノート（LLMへの提示記録はまだありません）')
    parts = [f'<header><h3>攻略ノート · step {_text(step)}</h3><p>{_text(work)} · {_text(intro)}</p></header>',
             '<p class="nbv-muted">記録された文章を、その時点の版で表示しています。追加の推論や翻訳は行っていません。</p>']
    displayed = {}
    if opening:
        path = opening.get('goal_path') or ([opening['goal']] if opening.get('goal') else [])
        parts.append('<h3>目標の道筋</h3><ol class="nbv-path">')
        for depth, page in enumerate(path):
            label = '大目標' if depth == 0 else f'小目標 {depth}'
            parts.append('<li>' + page_html(page, label=label, segment=segment) + '</li>')
            displayed[(page.get('id'), page.get('revision'))] = '本文を提示' if shown else ('人間用の参照' if human_only else '処理の入力')
        parts.append('</ol>')
        for key, label in [('active_experiment', '今、検証していること'),
                           ('latest_review', '直前の実験から分かったこと'),
                           ('latest_result', '最新の実観測')]:
            page = opening.get(key)
            if page:
                reference = (page.get('id'), page.get('revision'))
                if reference in displayed:
                    parts.append('<p class="nbv-muted">' + _text(label) + '：上に表示した '
                                 + _text(page.get('id')) + ' の同じ版に含まれています。</p>')
                    continue
                parts.append(page_html(page, label=label, segment=segment))
                abbreviated = key == 'latest_review' and 'kind' not in page
                displayed[reference] = (
                    '要点のみ提示' if abbreviated else '本文を提示') if shown else ('人間用の参照' if human_only else '処理の入力')
    else:
        parts.append('<p>このステップでは、まだ開始時のノートが記録されていません。</p>')
    reads = note.get('reads', ())
    for event in reads:
        page = (event.get('result') or {}).get('page')
        if page:
            displayed[(page.get('id'), page.get('revision'))] = '本文を追加参照'
    parts.append('<h3>開始時のしおり</h3><p class="nbv-muted">索引にあるだけのページは、本文まで提示されたとは扱いません。</p>')
    bookmarks = opening.get('bookmarks') or []
    if bookmarks:
        parts.append('<table><thead><tr><th>しおり</th><th>ページ・版</th><th>参照の状態</th></tr></thead><tbody>')
        for mark in bookmarks:
            key = (mark.get('id'), mark.get('revision'))
            name = BOOKMARKS.get(mark.get('name'), mark.get('name'))
            parts.append('<tr><td>' + _text(name) + '</td><td>' + _text(mark.get('title'))
                         + '<br><small>' + _text(mark.get('id')) + ' · 第' + _text(mark.get('revision'))
                         + '版</small></td><td>' + displayed.get(key, '索引のみ・本文の参照記録なし') + '</td></tr>')
        parts.append('</tbody></table>')
    else:
        parts.append('<p class="nbv-muted">この時点のしおり一覧はありません。</p>')
    parts += ['<h3>この呼び出しで追加参照した内容</h3>', _reads(reads, segment),
              '<h3>このステップの書き込み</h3>',
              '<p class="nbv-muted">保存された変更です。開始時に提示された内容とは分けて表示します。</p>',
              _changes(note.get('changes', ()), segment)]
    return '''<div class="nbv"><style>
.nbv{font:14px/1.65 system-ui;color:#1e293b;background:#f8fafc;padding:18px;border-radius:12px}
.nbv h3{font-size:17px;margin:22px 0 10px}.nbv header h3{margin-top:0;font-size:21px}
.nbv h4{font-size:16px;margin:6px 0 12px}.nbv p{white-space:pre-wrap;overflow-wrap:anywhere}
.nbv-card{background:white;border:1px solid #dbe3ee;border-left:4px solid #60a5fa;border-radius:8px;padding:14px 18px;margin:12px 0}
.nbv-badge{color:#2563eb;font-size:12px}.nbv dl{margin:0}
.nbv-field{display:grid;grid-template-columns:145px minmax(0,1fr);gap:12px;margin:10px 0}
.nbv dt{font-size:12px;color:#64748b}.nbv dd{margin:0;white-space:pre-wrap;overflow-wrap:anywhere}
.nbv footer,.nbv-muted{font-size:12px;color:#64748b;overflow-wrap:anywhere}.nbv footer{margin-top:12px}
.nbv-path{border-left:2px solid #bfdbfe;padding-left:28px}.nbv-path li::marker{color:#2563eb}
.nbv table{width:100%;border-collapse:collapse}.nbv th,.nbv td{padding:9px;text-align:left;border-bottom:1px solid #dbe3ee;overflow-wrap:anywhere}
.nbv-change{border-top:1px solid #dbe3ee;padding:10px 0}.nbv summary{cursor:pointer;overflow-wrap:anywhere}
.nbv-columns{display:flex;gap:14px;flex-wrap:wrap}.nbv-columns>div{flex:1;min-width:220px}
.nbv-columns .nbv-field{display:block}
@media(max-width:650px){.nbv-field{display:block}.nbv dd{margin-top:3px}}
</style>''' + ''.join(parts) + '</div>'
