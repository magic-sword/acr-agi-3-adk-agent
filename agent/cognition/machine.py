"""State/transition declarations shared by execution routing and the observatory.

Positions affect presentation only. Transition targets are used by runtime code.
The global stop gate is checked before further work; observed experiments are
reviewed first. ADK DECIDE/RUN is the dispatch mechanism, not a second task graph.
"""
STATES = {
    'observe': ('観測を受け取る', 'host', 40, 50),
    'measure': ('実験の機械判定', 'host', 40, 210),
    'judge_effect': ('意味的な効果判定', 'llm', 40, 380),
    'select_goal': ('小目標を選ぶ', 'llm', 660, 50),
    'assess_goal': ('小目標を評価', 'llm', 350, 210),
    'choose_method': ('方法を選ぶ', 'llm', 1000, 50),
    'design_experiment': ('実験を設計', 'llm', 660, 210),
    'inspect_target': ('対象の位置を認識', 'llm', 1000, 210),
    'validate': ('実験・反復を検証', 'host', 1000, 380),
    'recovery': ('差し戻しを振り分け', 'host', 1000, 560),
    'skill_creation': ('スキル候補を作る', 'llm', 350, 380),
    'resolve_arguments': ('スキル引数を決める', 'llm', 660, 380),
    'skill_run': ('手続き実行・効果検証', 'host', 350, 560),
    'wait': ('操作送信・次の観測待ち', 'host', 660, 560),
    'stop': ('終了・停止', 'terminal', 40, 560),
    'interpret_world': ('対象・関係・疑問を更新', 'llm', 350, 50),
}
# event: (source, destination, condition label, edge category)
TRANSITIONS = {
    'observed_experiment': ('observe','measure','実験結果あり','normal'),
    'initial_goal': ('observe','interpret_world','小目標なし／区間変更','normal'),
    'existing_goal': ('observe','interpret_world','新しい観測','normal'),
    'world_initial_goal': ('interpret_world','select_goal','対象と未解決の疑問を保存','normal'),
    'world_existing_goal': ('interpret_world','assess_goal','条件付きの知見を添付','normal'),
    'measured': ('measure','interpret_world','画素・レベルの判定済み','normal'),
    'semantic': ('measure','judge_effect','意味判断が必要','normal'),
    'effect_judged': ('judge_effect','interpret_world','判定を保存','normal'),
    'goal_selected': ('select_goal','design_experiment','探索のみ','normal'),
    'goal_methods': ('select_goal','choose_method','利用可能な方法あり','skill'),
    'goal_continue': ('assess_goal','design_experiment','継続','normal'),
    'goal_replace': ('assess_goal','select_goal','完了／置換','recovery'),
    'goal_deferred': ('assess_goal','stop','証拠不足で保留','stop'),
    'goal_choose_method': ('assess_goal','choose_method','継続・方法の選択肢あり','skill'),
    'method_explore': ('choose_method','design_experiment','探索','normal'),
    'method_learn': ('choose_method','skill_creation','経験から作成','skill'),
    'method_skill': ('choose_method','resolve_arguments','採用済み利用／候補試行','skill'),
    'build_missing': ('skill_creation','recovery','作成根拠が不足','recovery'),
    'arguments_missing': ('resolve_arguments','recovery','引数の根拠が不足','recovery'),
    'method_missing': ('choose_method','recovery','方法の根拠が不足','recovery'),
    'measure_boundary': ('measure','interpret_world','区間の切替','normal'),
    'judgment_boundary': ('judge_effect','interpret_world','区間の切替','normal'),
    'skill_built': ('skill_creation','design_experiment','候補を保存','skill'),
    'arguments_ready': ('resolve_arguments','skill_run','引数を検証済み','skill'),
    'skill_next': ('observe','skill_run','手続き継続・効果確認','skill'),
    'skill_action': ('skill_run','wait','開始条件を満たす次の手','skill'),
    'skill_finished': ('skill_run','assess_goal','完了／効果不一致','skill'),
    'inspect_click': ('design_experiment','inspect_target','未確認のクリック','normal'),
    'direct_test': ('design_experiment','validate','非クリック／確認を再利用','normal'),
    'target_found': ('inspect_target','validate','矩形と対象画素への命中を確認','normal'),
    'target_rejected': ('inspect_target','recovery','不一致／特定不能','recovery'),
    'test_rejected': ('validate','recovery','同条件で不支持の実験','recovery'),
    'test_accepted': ('validate','wait','予測を固定・操作を選択','normal'),
    'recover_target': ('recovery','design_experiment','対象／スキルの根拠不足','recovery'),
    'recover_goal': ('recovery','assess_goal','同じ失敗実験','recovery'),
    'recover_exhausted': ('recovery','stop','回復上限超過','stop'),
    'next_observation': ('wait','observe','受付確認と次フレーム','normal'),
    'restart': ('observe','wait','未開始／ゲームオーバー → RESET','normal'),
}
GLOBAL_GATES = [
    '勝利・評価終了・時間／操作／モデル予算の上限 → 終了・停止',
    '受付不明・不正な出力の訂正上限・実行例外 → 終了・停止',
    'タスク出力は適用前に契約を検証。不正出力は同タスクで最大2回訂正',
    'スキル試行の採点・採用はホスト処理。独立したLLMタスクではない',
]


def destination(event):
    return TRANSITIONS[event][1]
