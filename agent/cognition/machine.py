"""Current single-action lifecycle, shared with the replay diagram."""
STATES = {
    'observe': ('画面・受付・実測', 'host', 40, 50),
    'act': ('結果解釈＋次の1操作', 'llm', 370, 50),
    'wait': ('操作送信・観測待ち', 'host', 700, 50),
    'stop': ('終了・期限・受付による停止', 'terminal', 370, 300),
}
TRANSITIONS = {
    'direct_received': ('observe','act','現在の観測と直前の結果','normal'),
    'direct_accepted': ('act','wait','使用可能な操作・現在マスク内の座標','normal'),
    'repair_act': ('act','act','出力契約の修正は最大1回','recovery'),
    'invalid_act': ('act','stop','出力契約エラー','stop'),
}
GLOBAL_GATES = [
    '通常は結果解釈と次の1操作の選択を1回のモデル要求に統合',
    'クリック位置は現在の対象マスク内からホストが決定',
    '操作対象の無変化・不明と、別対象の実測された変化だけを履歴へ保存',
    '終了・受付不明・使用不可操作・観測ID・実時間予算はホストが検査',
    '操作選択・受付・実測とモデルの因果解釈を区別する',
]


def destination(event):
    return TRANSITIONS[event][1]
