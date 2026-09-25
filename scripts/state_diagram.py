"""Small inline SVG for replay. No JavaScript, graph layout engine or polling."""
from html import escape

NODES = {
    'observe': (20, 165, 175, 78, '観測を受け取る', 'ゲーム画面・操作結果'),
    'design': (245, 75, 250, 78, '実験設計', '小目標・仮説・予測を保存'),
    'review': (245, 180, 250, 78, '結果判定', '予測と実測を照合・更新'),
    'build': (245, 285, 250, 78, 'スキル作成', '経験から手続きを作る'),
    'run': (575, 165, 215, 90, '実行・検証', '操作選択・判定保存・評価'),
    'wait': (860, 165, 215, 90, '操作・観測待ち', '送信 → 受付 → 次の画面'),
    'end': (860, 335, 215, 78, '終了・停止', '完了・予算上限など'),
}
JOBS = {'act': '実験の操作', 'invoke': 'スキル実行', 'trial': '実ゲームで試行',
        'evaluate': '採用判定', 'learn': '作成へ切り替え', 'stop': '停止',
        'propose_skill': '候補を保存', 'submit_review': '実験の判定を保存',
        'defer_skill': '不足情報を設計へ戻す', 'redesign': '反復した実験を見直す'}


def state_diagram_html(machine=None):
    machine = machine or {}
    if machine.get('work') in set(FOCUSED)-{'skill_creation'} or machine.get('node') in FOCUSED:
        return focused_diagram_html(machine)
    active = machine.get('node')
    label = NODES[active][4] if active in NODES else '判断（仕事の記録なし）' if active == 'decide' else '記録を選択してください'
    status = ' · '.join(str(v) for v in (label, machine.get('phase'),
                        JOBS.get(machine.get('job'))) if v)
    boxes = []
    for key, (x, y, width, height, title, detail) in NODES.items():
        selected = key == active
        fill, stroke = ('#dbeafe', '#1d4ed8') if selected else ('#ffffff', '#94a3b8')
        marker = '● 現在位置' if selected else ''
        boxes.append(f'''<g data-state="{key}" data-active="{str(selected).lower()}">
          <rect x="{x}" y="{y}" width="{width}" height="{height}" rx="12"
            fill="{fill}" stroke="{stroke}" stroke-width="{3 if selected else 1.5}"/>
          <text x="{x+14}" y="{y+23}" font-size="12" fill="#1d4ed8">{marker}</text>
          <text x="{x+14}" y="{y+44}" font-size="18" font-weight="600">{title}</text>
          <text x="{x+14}" y="{y+height-10}" font-size="12">{detail}</text></g>''')
    return f'''<div style="font:14px system-ui;color:#172554;background:#f8fafc;padding:12px;border-radius:10px">
      <b>状態遷移と現在位置</b><p role="status">{escape(status)}</p>
      <svg xmlns="http://www.w3.org/2000/svg" role="img" aria-label="{escape('状態遷移図。'+status, quote=True)}"
        viewBox="0 0 1100 490" style="width:100%;max-height:400px">
        <title>{escape('状態遷移と現在位置：'+status)}</title>
        <defs><marker id="replay-arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto">
          <path d="M0,0 L8,4 L0,8" fill="#64748b"/></marker></defs>
        <rect x="225" y="45" width="290" height="335" rx="16" fill="#eff6ff"
          stroke="{'#1d4ed8' if active in ('design','review','build','decide') else '#cbd5e1'}" stroke-dasharray="5 4"/>
        <g fill="#172554" font-family="system-ui,sans-serif">
          <text x="245" y="65" font-size="13">DECIDE · 判断する仕事を分ける</text>
          <text x="630" y="130" font-size="13">RUN</text>
          <g fill="none" stroke="#64748b" stroke-width="1.7" marker-end="url(#replay-arrow)">
            <path d="M195,200 H215 V219 H245"/>
            <path d="M195,182 H205 V114 H245"/>
            <path d="M495,114 H540 V185 H575"/>
            <path d="M495,219 H575"/>
            <path d="M495,324 H540 V242 H575"/>
            <path d="M625,165 V20 H535 V90 H495"/>
            <path d="M650,255 V397 H370 V363"/>
            <path d="M790,209 H860"/>
            <path d="M790,240 H825 V374 H860"/>
            <path d="M965,255 V445 H105 V243"/>
          </g>
          <text x="435" y="15" font-size="12">結果を戻す</text>
          <text x="395" y="395" font-size="12">learn：スキル作成へ</text>
          <text x="798" y="197" font-size="12">操作</text>
          <text x="830" y="303" font-size="12">停止</text>
          <text x="520" y="440" font-size="12">次の観測</text>
          {''.join(boxes)}
          <text x="225" y="478" font-size="13">攻略ノート：目標経路 → 実験 → 実観測 → 判定。画素条件の判定はホストが実行。</text>
        </g>
      </svg>
      <div style="color:#475569;font-size:12px">青枠と●が表示時点の位置です。観測待ち・終了を含む概略図です。
      「状態遷移」で再生すると、途中の判断やスキル作成も順に確認できます。</div>
    </div>'''


FOCUSED = {
    'select_goal': '小目標を選ぶ', 'assess_goal': '小目標を評価',
    'design_experiment': '実験を設計', 'inspect_target': '対象を確認',
    'judge_effect': '効果判定（実測優先）', 'choose_method': '方法を選ぶ',
    'resolve_arguments': 'スキル引数を決める', 'skill_creation': 'スキルを作成',
}


def focused_diagram_html(machine):
    positions = {'observe':(15,80), 'judge_effect':(205,20), 'assess_goal':(395,20),
        'select_goal':(585,20), 'choose_method':(585,125), 'design_experiment':(395,230),
        'inspect_target':(205,230), 'resolve_arguments':(775,125), 'skill_creation':(775,230),
        'run':(205,345), 'wait':(15,345), 'end':(395,345)}
    titles = {**FOCUSED, 'observe':'観測と機械測定', 'run':'検証・記録', 'wait':'操作・観測待ち', 'end':'終了・停止'}
    edges = [('observe','judge_effect'),('judge_effect','assess_goal'),('assess_goal','select_goal'),
        ('select_goal','choose_method'),('choose_method','design_experiment'),('choose_method','resolve_arguments'),
        ('choose_method','skill_creation'),('design_experiment','inspect_target'),('inspect_target','run'),('run','wait')]
    active=machine.get('node')
    if active == 'build':
        active = 'skill_creation'
    boxes=[]
    for key,(x,y) in positions.items():
        selected=active==key
        boxes.append(f'<g data-state="{key}" data-active="{str(selected).lower()}"><rect x="{x}" y="{y}" width="175" height="70" rx="10" fill="{"#dbeafe" if selected else "white"}" stroke="{"#1d4ed8" if selected else "#94a3b8"}"/><text x="{x+9}" y="{y+22}" font-size="11">{"● 現在位置" if selected else ""}</text><text x="{x+9}" y="{y+47}" font-size="14">{titles[key]}</text></g>')
    paths=[]
    for a,b in edges:
        x,y=positions[a];u,v=positions[b]
        paths.append(f'<path d="M{x+87},{y+35} L{u+87},{v+35}"/>')
    status=escape(titles.get(active,'判断')+' · '+str(machine.get('phase','')))
    return f'<div style="background:#f8fafc;padding:12px"><b>状態遷移と現在位置</b><p>{status}</p><svg xmlns="http://www.w3.org/2000/svg" role="img" aria-label="分割された判断タスク" viewBox="0 0 970 470" style="width:100%;max-height:450px"><g stroke="#94a3b8" fill="none">{"".join(paths)}</g>{"".join(boxes)}<text x="15" y="450" font-size="12">機械測定はLLM判定を省略。対象不一致は設計へ、同条件の反復は小目標評価へ差し戻します。</text></svg></div>'
