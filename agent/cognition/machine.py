"""The active image-led policy, shared with the replay diagram."""
STATES = {
    'observe': ('画面と受付結果を受け取る', 'host', 40, 50),
    'measure': ('変化と短い履歴を更新', 'host', 350, 50),
    'attend': ('解釈・注目・次の操作', 'llm', 660, 50),
    'validate': ('合法性・同条件反復を検証', 'host', 1000, 210),
    'wait': ('操作送信・観測待ち', 'host', 350, 380),
    'stop': ('終了・停止', 'terminal', 1000, 560),
}
TRANSITIONS = {
    'received': ('observe', 'measure', '受付確認済みの操作結果', 'normal'),
    'ready': ('measure', 'attend', '現在画面・変化・最近の試行', 'normal'),
    'proposed': ('attend', 'validate', '1回で注目対象と操作を提出', 'normal'),
    'accepted': ('validate', 'wait', '操作を選択・予想を記録', 'normal'),
    'repair': ('validate', 'attend', '不正出力の修正は最大1回', 'recovery'),
    'repair_exhausted': ('validate', 'stop', '修正回数または時間を消費', 'stop'),
    'next_observation': ('wait', 'observe', '操作受付と新しい画面', 'normal'),
    'restart': ('observe', 'wait', '未開始／ゲームオーバー → RESET', 'normal'),
}
GLOBAL_GATES = [
    '勝利・評価終了・時間／操作上限 → 終了。操作前にも時間を確認',
    '通常は1 HTTP要求。無効な提出だけ最大1回修正',
    '同じ状態で無変化だった同じ操作は、事前に宣言した試行回数を超えて繰り返さない',
    '物体一覧・課題ID・意味的な対象分類は不要。注意の選択はLLMが行う',
    '受付不明の操作を再送しない。画素変化だけで因果や目標達成を断定しない',
]


def destination(event):
    return TRANSITIONS[event][1]
