# 物体認識から行動計画への引継ぎ

更新：2026-09-29。採用理由・成功した検証・文献・主要ログは[統合資料](visual-recognition-adopted-ja.md)が正本。このページは次の作業に必要な状態だけを残す。

## 実装済み

既定は**初回SAM＋通常の画素更新**（`COGNITION_PROPOSALS=sam_initial`）。初回・RESET・レベル切替でSAMを使い、通常は一意の画素対応を追跡する。無変化なら抽出も再利用。全候補索引、観測内IDと追跡ID、受理した対象参照の継続／欠落を既存の意味質問へ接続済み。

89フレームの統合検証で候補回収は合成242/264・実ログ133/133、移動は104/127・6/6。実ログの初回後の観測処理は平均6.70ms（Qwenを除く）。全206テスト、実Qwenの1トークン接続3問、実SDKの2操作を確認した。数値の条件・限界は[統合資料](visual-recognition-adopted-ja.md)、実行設定は[ランタイム仕様](hybrid-perception-runtime-20260928-ja.md)を参照。

## 現在の判断

**前処理の精度だけで方式を決めず、後段の計画・因果推論に必要な情報から逆算する。** 物体の厳密な分割・全クラスの確定を必須にせず、曖昧な候補や対応仮説を許容する。動かない「青い四角」も後で役割を検討できるよう残す。

全候補・連動関係を保持することと、モデルへ一度にすべて提示することは別。大量の連動情報がノイズになる可能性があるため、どの情報を集計・要約・提示するかは未確定。連動グループを同一物体や因果関係と断定しない。

[直近の検証と判断](visual-recognition-adopted-ja.md#現時点の判断と直近の検証)に結果を集約した。差分を渡したモデルの39/51と、プログラムによる連動集計の8/8は別の評価範囲で、優劣は未確認。連動集計の本番統合、追跡器の削除、画像の一律省略、全連動の常時提示は採用していない。

## 再開時に確認すること

1. **後段の問いと必要情報を整理する。** 計画・因果推論が何を判断し、どの候補・変化・連動・不明点を必要とするかを先に定める。必要な情報を落とさず、不要な提示を減らす前処理を比較する。
2. **前処理から行動まで通して評価する。** 必要対象の考慮漏れ、誤った因果の断定、仮説更新・検証行動、入力情報量・呼出し数・総時間を確認する。認識だけの正答率や「全連動を計算できた」ことを最終成果にしない。
3. **既存の未解決例を残す。** 変化した64遷移すべてで対応保留が残った再確認の過剰発火、色変化・同形物体での対応切れ、測定と自由文の説明の矛盾を回帰対象にする。追跡器は候補参照の継続も担うため、質問の精度だけで削除を決めない。

以前の実ログでは、黄色いバー端の2画素だけが変わり、意味質問も「不明」なのに、後続の`reconcile`が「プレイヤーが下へ動いた」と説明した。SAM統合での解消は未確認。[比較HTML](../outputs/measured-runtime-20260928/gallery.html)のls20／measured／step 2、[生ログ](../outputs/measured-runtime-20260928/live-ls20-measured/ls20-9607627b/cognition/f93c11628b954bf2a4d4eb071d8dd71c.artifacts.jsonl)、[当時の原因分析](history/visual-recognition/cognition-handoff-analysis-20260928-ja.md)を参照。

## コード・ログの入口

| 作業 | 入口 |
|---|---|
| 候補生成・保持・移動測定 | [sam_proposals.py](../agent/cognition/sam_proposals.py)、[proposal_tracking.py](../agent/cognition/proposal_tracking.py)、[hybrid_perception.py](../agent/cognition/hybrid_perception.py) |
| 意味質問と再確認への分岐 | [perception.py](../agent/cognition/perception.py)、[workflow.py](../agent/cognition/workflow.py) |
| 対象参照の受理と計画 | [deliberation.py](../agent/cognition/deliberation.py) |
| 再生・回帰確認 | [verify_hybrid_runtime.py](../scripts/verify_hybrid_runtime.py)、[結合テスト](../tests/test_hybrid_perception.py)、[主要ログ一覧](visual-recognition-adopted-ja.md) |
| 人がログを読む | [可視化ノート](../notebooks/agent_observatory.ipynb)の「測定・計画を読む」 |

## 再開時の注意

未コミット変更があるため最初に`git status`を確認する。既存の実験データを上書きせず、新しい出力先で測定する。現在の認識精度、意味質問の正答、Qwenの呼出し数、ゲームの進展を別々に記録する。

過去の個別試験は[履歴フォルダ](history/visual-recognition/README.md)へ整理済み。「未実装」「次に試す」といった記述は当時の状態であり、現在の採用方針を上書きしない。
