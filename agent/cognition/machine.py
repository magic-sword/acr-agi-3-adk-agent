"""Executable transitions shared with the replay diagram."""
STATES = {
    'observe': ('画面・受付・実測', 'host', 40, 50),
    'understand': ('熟考：状況理解', 'llm', 370, 50),
    'backchain': ('熟考：前提条件を逆算', 'llm', 700, 50),
    'ground': ('熟考：実行手順へ具体化', 'llm', 1030, 50),
    'reconcile': ('熟考：結果照合・更新', 'llm', 370, 300),
    'choose_skill': ('高速：スキル選択 / 8', 'llm', 1030, 300),
    'execute_step': ('高速：操作 / 7 / 8', 'llm', 700, 530),
    'wait': ('操作送信・観測待ち', 'host', 40, 300),
    'stop': ('終了・期限・受付による停止', 'terminal', 40, 530),
}
TRANSITIONS = {
    'received': ('observe','understand','初期観測／境界','normal'),
    'understood': ('understand','backchain','対象と疑問から逆算','normal'),
    'direct_ground': ('understand','ground','一操作の試行／既知手順','normal'),
    'backchained': ('backchain','ground','選んだ小目標を具体化','normal'),
    'ground_understand': ('ground','understand','対象の再解釈が必要','recovery'),
    'ground_backchain': ('ground','backchain','前提条件の修正が必要','recovery'),
    'grounded': ('ground','choose_skill','具体的な対象・条件・操作','normal'),
    'execute': ('choose_skill','execute_step','起動IDを付けて開始','normal'),
    'accepted': ('execute_step','wait','合法な操作を選択','normal'),
    'next_observation': ('wait','observe','受付と次の観測','normal'),
    'continue_execution': ('observe','execute_step','現在の手順を継続','normal'),
    'probe_observed': ('observe','reconcile','試行結果を取得（達成とは別）','normal'),
    'skill_reconsider': ('choose_skill','reconcile','8：適用できない／判断不能','recovery'),
    'step_reconsider': ('execute_step','reconcile','8：予想外／判断不能','recovery'),
    'step_done': ('execute_step','execute_step','7：手続き内の次の段階','normal'),
    'procedure_done': ('execute_step','reconcile','7：小目標の完了候補','normal'),
    'review_understand': ('reconcile','understand','対象・目標解釈を更新','recovery'),
    'review_backchain': ('reconcile','backchain','前提・小目標を更新','normal'),
    'review_ground': ('reconcile','ground','次の手順／局所修正','normal'),
    'review_resume': ('reconcile','execute_step','未完了の手順を継続','normal'),
    'repair_understand': ('understand','understand','理解の出力修正','recovery'),
    'repair_backchain': ('backchain','backchain','依存関係の出力修正','recovery'),
    'repair_ground': ('ground','ground','手順の出力修正','recovery'),
    'repair_reconcile': ('reconcile','reconcile','照合の出力修正','recovery'),
    'invalid_understand': ('understand','stop','理解の出力契約エラー','stop'),
    'invalid_backchain': ('backchain','stop','依存関係の出力契約エラー','stop'),
    'invalid_ground': ('ground','stop','手順の出力契約エラー','stop'),
    'invalid_reconcile': ('reconcile','stop','照合の出力契約エラー','stop'),
}
GLOBAL_GATES = [
    '熟考は理解・逆算・具体化・結果照合。必要な工程だけに戻る',
    '高速は1トークン。8は結果照合へ。7の小目標完了は確認前の候補',
    '一操作の試行は受付済みの結果で照合へ。試行終了と目標達成は別',
    '目標・対象・仮説を保持。操作結果は手続きの起動IDに対応付ける',
    '同条件の反復拒否なし。合法操作・観測ID・受付・実時間予算を検証',
]


def destination(event):
    return TRANSITIONS[event][1]
