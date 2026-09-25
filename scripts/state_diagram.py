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
