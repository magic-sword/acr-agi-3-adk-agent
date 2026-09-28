# 物体認識から行動計画への引継ぎ

更新：2026-09-28。採用理由・成功した検証・文献・主要ログは[統合資料](visual-recognition-adopted-ja.md)が正本。このページは次の作業に必要な状態だけを残す。

## 実装済み

既定は**初回SAM＋通常の画素更新**（`COGNITION_PROPOSALS=sam_initial`）。初回・RESET・レベル切替でSAMを使い、通常は一意の画素対応を追跡する。無変化なら抽出も再利用。全候補索引、観測内IDと追跡ID、受理した対象参照の継続／欠落を既存の意味質問へ接続済み。

89フレームの統合検証で候補回収は合成242/264・実ログ133/133、移動は104/127・6/6。実ログの初回後の観測処理は平均6.70ms（Qwenを除く）。全206テスト、実Qwenの1トークン接続3問、実SDKの2操作を確認した。数値の条件・限界は[統合資料](visual-recognition-adopted-ja.md)、実行設定は[ランタイム仕様](hybrid-perception-runtime-20260928-ja.md)を参照。

## 次に検証すること

1. **再確認の過剰発火を減らす。** 現状は画面が変わった64遷移すべてで対応保留が残る。全体候補が追えているときの部品の曖昧さや背景枠の変化を調べる。真の色変化・出現・対応不明を隠さず、`requires_review`とQwen呼出し数を減らせるか測る。単純な必要時SAMへの切替は再実行が多くなるため、そのまま採用しない。
2. **測定と結果説明・計画の整合を確認する。** 候補参照の存在確認・追跡は実装済みだが、意味の一致や自由文の根拠は未保証。`reconcile`が測定事実を逸脱していないかを、候補・観測IDに結び付けて検証する。
3. **対応切れへの処理を個別に測る。** 色変化、同形物体、新しい多色物体を対象に、再検出・再同定が必要な場面を切り分ける。回収率とID継続率を混ぜない。

以前の実ログには、黄色いバー端の2画素だけが変わり、意味質問も「不明」なのに、後続の`reconcile`が「プレイヤーが下へ動いた」と説明した例がある。SAM統合で解消したとは未確認なので回帰ケースとして残す。[比較HTML](../outputs/measured-runtime-20260928/gallery.html)のls20／measured／step 2、[生ログ](../outputs/measured-runtime-20260928/live-ls20-measured/ls20-9607627b/cognition/f93c11628b954bf2a4d4eb071d8dd71c.artifacts.jsonl)、[当時の原因分析](history/visual-recognition/cognition-handoff-analysis-20260928-ja.md)を参照。

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
