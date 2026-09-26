"""Executable transitions shared by the runtime and replay diagram."""
STATES = {
    'observe': ('画面・操作受付・実測結果', 'host', 40, 50),
    'deliberate': ('熟考：因果・逆算・手続き構築', 'llm', 380, 50),
    'choose_skill': ('高速：スキル選択 / 8', 'llm', 740, 50),
    'execute_step': ('高速：操作 / 次の手順 / 8', 'llm', 740, 290),
    'wait': ('合法性確認・操作送信・観測待ち', 'host', 40, 290),
    'stop': ('終了・時間／受付による停止', 'terminal', 380, 530),
}
TRANSITIONS = {
    'received': ('observe','deliberate','初期観測','normal'),
    'need_plan': ('observe','deliberate','計画がない／熟考へ復帰','normal'),
    'planned': ('deliberate','choose_skill','計画と実行手続きを採用','normal'),
    'choose': ('observe','choose_skill','次のスキルを選ぶ','normal'),
    'execute': ('choose_skill','execute_step','現在の手順を継続','normal'),
    'accepted': ('execute_step','wait','合法な操作を選択','normal'),
    'next_observation': ('wait','observe','受付と新しい画面','normal'),
    'skill_reconsider': ('choose_skill','deliberate','8：適用できない／判断不能','recovery'),
    'step_reconsider': ('execute_step','deliberate','8：予想外／判断不能','recovery'),
    'step_done': ('execute_step','execute_step','7：次の手順へ','normal'),
    'procedure_done': ('execute_step','choose_skill','7：手続き完了','normal'),
    'repair': ('deliberate','deliberate','不正な計画の修正は最大1回','recovery'),
    'repair_exhausted': ('deliberate','stop','有効な計画が得られない','stop'),
}
GLOBAL_GATES = [
    '高速判断は1トークン。8は操作せず熟考へ戻る。7は手順を進める',
    '同じ手順の継続に計画の再生成は不要。反復操作は拒否しない',
    'ゲーム全体と1観測内の実時間予算を守る。勝利・評価終了・操作上限で終了',
    '古い観測ID・非合法操作・範囲外座標は拒否。受付不明の操作は再送しない',
    '画素差分は実測、因果・手順完了はモデル判断として別々に記録',
]


def destination(event):
    return TRANSITIONS[event][1]
