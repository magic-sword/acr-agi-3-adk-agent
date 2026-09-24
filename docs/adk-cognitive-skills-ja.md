# 認知スキルの責務と専門手順

> 以下は詳細な認知機能を分割した旧設計の記録。現在の実行構成・提出契約は[一手ずつ学ぶ認知ループ](adk-cognitive-state-machine-ja.md)を参照。

更新日: 2026-09-24。現在の `agent/skills/` に対応する仕様。[設計レビュー](adk-design-review-20260924-ja.md)で指摘した、状態の必須指示やホスト制御との混同を解消した。[接続方法](adk-native-skills-ja.md)も参照。

## 区分

|層|責務|実装|
|---|---|---|
|状態別instruction|現在の課題、必須出力、共通の証拠・操作契約|`cognition/instructions.py`|
|Skill|問題に応じて読む専門的な判断手順・例|`skills/*/SKILL.md`|
|Tool|型付きの証拠取得・カーソル移動・提出|`cognition/workflow.py`、`completion.py`|
|ホスト|検証、遷移、予算、確定、描画、操作変換|`engine.py`、`validation.py`、`rendering.py`、`controls.py`|
|状態データ|観測事実、仮説、計画、結果待ち、手順候補|`cognition/state.py`、`evidence.py`|

スキルは独立したエージェントでも状態ノードでもない。現在の状態の判断を補助する任意の方法である。読み込みは「実行成功」を意味せず、スキルごとの別の戻り値も定義しない。結果は呼び出し元のInterpretation／Proposal契約に組み込む。

## 13個のスキル

|ID・本文|適用する問題|固有の判断手順|
|---|---|---|
|[S01 visual-observation](../agent/skills/visual-observation/SKILL.md)|画像の対象・関係が曖昧|問いに応じて現在・過去・差分・途中フレームを選び、表示座標と元座標を区別|
|[S02 occlusion-memory](../agent/skills/occlusion-memory/SKILL.md)|対象が隠れた、同一性が不明|最後の可視証拠と介在操作から候補を保ち、再出現時に同一性を再検査|
|[S03 prediction-verification](../agent/skills/prediction-verification/SKILL.md)|予測の意味的な照合が必要|効果と維持条件を別に確認し、遅延・不可視・介在操作を反証と区別|
|[S04 hypothesis-maintenance](../agent/skills/hypothesis-maintenance/SKILL.md)|複数の説明が成立する|条件と予測を分解し、区別できる証拠だけで仮説の状態を更新|
|[S05 goal-inference](../agent/skills/goal-inference/SKILL.md)|目標不明、見た目の完成で勝てない|関係・順序・同時条件を候補にし、欠けた条件を調べる|
|[S06 cause-diagnosis](../agent/skills/cause-diagnosis/SKILL.md)|予測と結果が食い違う|入力・対象・可視性・遅延・モードを調べ、影響範囲を限定して修正|
|[S07 discriminating-experiment](../agent/skills/discriminating-experiment/SKILL.md)|未知が次の行動を変える|既知の実験を避け、観測可能な予測の差と費用を比較して1操作を選ぶ|
|[S08 backward-planning](../agent/skills/backward-planning/SKILL.md)|前提や操作順序が重要|目標から前提を逆算し、干渉・退避場所・維持条件を検査して短い非循環計画へ。条件を破壊する順序は`check_plan_order`で前向き検査|
|[S09 plan-repair](../agent/skills/plan-repair/SKILL.md)|既存計画が継続できない|影響する依存先を特定し、有効な達成状態を保って残りの計画全体を再提案|
|[S10 temporal-reasoning](../agent/skills/temporal-reasoning/SKILL.md)|遅延・順序・時間窓がある|外部操作stepと再生フレーム順を区別し、因果が混ざる操作を明示|
|[S11 effect-revaluation](../agent/skills/effect-revaluation/SKILL.md)|以前の有害な作用が役立つ可能性|既知の作用と現在の有用性を別に評価し、因果規則を不用意に変更しない|
|[S12 action-grounding](../agent/skills/action-grounding/SKILL.md)|操作やクリック対象が曖昧|操作対応の証拠を確認し、元座標でカーソルをプレビューして入力を提案|
|[S14 procedure-reuse](../agent/skills/procedure-reuse/SKILL.md)|過去の成功手順を流用したい|制御・役割・前提を照合し、座標の盲目的再生を避けてチェック付き計画へ変換|

各本文に、適用条件、具体的な判断手順、反例や不確実な場合の扱いを含める。固有の補足が必要なS01/S06/S07/S08にだけ参照資料を置く。15個に複製されていた共通 `evidence-contract.md` は削除した。共通契約は状態instructionと型・ホスト検証で維持する。

スクリプト数はスキルの品質指標にしない。現在は上記を推論手順として提供し、観測比較・検証・描画など確実な処理は既存のTool／ホスト実装で行う。逆算計画の実モデル誤答を受け、条件の追加・削除を順に適用する純粋な `check_plan_order` を追加した。これは明示した仮定の整合性検査であり、ゲームルールの真偽を確定するソルバではない。

## ホストへ分類し直した旧項目

- **旧S13 decision-commit:** `CognitiveTurn.commit()` とWorkflowのCOMMITが担当。判断ID、pending、認知記憶の公開をスキル選択に依存させない。
- **旧S15 budget-recovery:** モデル判断数、HTTP要求数、残り時間、合法操作、RESET上限をコードで強制する。実験の費用・リスクの比較はS07、計画の実行可能性はS08で扱う。

この2個のSKILL.mdは削除し、利用可能な専門スキルに数えない。IDの欠番は旧設計との対応を保つためである。

## 検証の基準

形式検証は名前・説明・資料リンク・パッケージングを確認する。接続検証は実際のADK経由で本文と資料を必要時だけ読めることを確認する。専門手順の評価は、識別不能な実験を避ける、不可視状態を断言しない、壊れた計画をそのまま繰り返さない等の判断結果で行う。

ゲーム性能は同一モデル・ゲーム・操作数・時間予算で比較する。スキルをロードした回数や13個のファイルが存在することを、その能力を獲得した証拠にはしない。
