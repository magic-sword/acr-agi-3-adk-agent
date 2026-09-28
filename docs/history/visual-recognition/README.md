# 物体認識の調査履歴

現在の採用構成・成果・主要ログは[統合資料](../../visual-recognition-adopted-ja.md)、実行方法は[ランタイム仕様](../../hybrid-perception-runtime-20260928-ja.md)を参照する。ここは過去の比較を再確認するための保管場所。本文の「未実装」「次に試す」は各実験時点の状態であり、現行の判断ではない。

## 採用判断に残す短いメモ

| 調べたこと | 今後も参考になる点 | 詳細 |
|---|---|---|
| Qwenによる候補生成 | 多色形状のまとまりは改善する例があるが、実画面の列挙漏れが残る。非標準座標での0/80を一般能力の評価に使わない | [概念による列挙](object-proposals-comparison-20260928-ja.md)・[標準座標での再測定](native-grounding-comparison-20260928-ja.md) |
| 反復・枠提示・マスク | 終了判断・重複が難しく、マスクは未検出物も隠す。一部の枠提示の改善を、全条件で有効だったと解釈しない | [反復](iterative-proposals-comparison-20260928-ja.md)・[一括反復とマスク](batch-mask-comparison-20260928-ja.md) |
| EdgeBoxes | 数十msで速いが単独の実画面回収ではSAMに届かない。色領域との事後結合は今後の比較候補で、運用速度・追跡は未検証 | [比較・原論文・公式実装](edgeboxes-comparison-20260928-ja.md) |
| 確率による対象との結び付け | 3分類の出力確率は取得できたが、主条件の完全正解は3/49、0.9閾値でも4/49。高確率を正しさの保証にしない | [単一・複数・不明のクイズ](target-binding-quiz-20260928-ja.md)・[Jevを参考にした確率取得](target-binding-confidence-probe-20260928-ja.md)・[3分類の検証](candidate-match-validation-20260928-ja.md) |
| 画像強調・説明・動画 | 一部の改善や閲覧上の利点はあるが、微小移動の安定検出を代替できなかった | 以下の「画像入力・観測方法」に保管 |
| 注意・特徴の可視化 | 特徴差があることと、正しい移動判断ができることは別。注意図だけで原因を断定しない | 以下の「診断」に保管 |
| 測定から計画への引継ぎ | 測定を保持しても、後段の説明がそれと矛盾する例があった。現在も回帰確認が必要 | [ログ分析](cognition-handoff-analysis-20260928-ja.md)・[引継ぎ](../../visual-recognition-handoff-ja.md) |

## 2026-09-29：候補の意味付けから連動情報へ

| 記録 | 検証結果・位置付け |
|---|---|
| [概念生成と候補紐づけの分離評価](concept-binding-split-20260929-ja.md) | 概念の列挙と個体の選択を分けて測定。注釈訂正を含む条件と採点を保存 |
| [先行研究とQwenの学習からの改善案](concept-binding-research-20260929-ja.md) | 文献調査。論文の成果と手元のモデルへの応用仮説を区別 |
| [候補ごとの外観記述・個体選択](region-descriptions-comparison-20260929-ja.md) | 従来22/36に対し10/36・6/36。全体／部品の判定による除外も原因で、採用せず |
| [画素差分・追跡情報・連動集計](delta-recognition-comparison-20260929-ja.md) | 差分を渡すモデルは39/51。プログラムの連動集計8/8は別の小規模問題で、直接比較不可 |

現在の判断は[統合資料](../../visual-recognition-adopted-ja.md)へ集約した。前処理では曖昧さを許容し、計画・因果推論に必要な情報とノイズの両方を見る。全連動の常時提示やプログラム解析への全面移行を採用した記録ではない。

これらは方式自体の一般的な失敗を断定するものではない。条件・注釈・採点の訂正は元の詳細記録に残している。毎手SAMと必要時SAMの比較は、現行採用の直接の根拠なので[時系列比較](../../temporal-proposals-comparison-20260928-ja.md)に残した。

## 成功した方式に至る過去の測定

現在の仕組みに取り入れた知見の詳しい経緯。現行ランタイムの成績と、試作時の成績は合算しない。

- [元画素から状態を取得して比較](instance-state-comparison-20260928-ja.md)
- [測定情報を短い質問で読む](simple-action-comparison-20260928-ja.md)
- [1トークン化と自動出題](one-token-comparison-20260928-ja.md)
- [無変化への質問を省く](hierarchical-comparison-20260928-ja.md)
- [整理前の採用経緯・ステートマシン設計・旧統合ベンチマーク](measured-perception-adoption-20260928-ja.md)

## 個別資料の索引

詳細が必要なときだけ展開する。上記の2026-09-29の4件と、以下の従来44件を合わせて48件を保管する。検証ログは移動せず、各文書から元の保存先へリンクしている。

<details>
<summary>2026-09-28までの44件の資料を表示</summary>

### 候補生成・対象との結び付け

- [画像からの物体候補列挙とプログラム抽出の比較](object-proposals-comparison-20260928-ja.md)
- [Qwen標準座標・盤面のみ・対象指定の比較](native-grounding-comparison-20260928-ja.md)
- [カテゴリーなしの一括検出と、枠を提示する反復検出の比較](iterative-proposals-comparison-20260928-ja.md)
- [一括検出の反復と、検出済み領域のマスク比較](batch-mask-comparison-20260928-ja.md)
- [EdgeBoxesとSAM：速度と候補の取りこぼしの比較](edgeboxes-comparison-20260928-ja.md)
- [対象説明と測定候補の結び付け：小さなクイズの精度測定（2026-09-28）](target-binding-quiz-20260928-ja.md)
- [対象の結び付けに確信度を使う案：API動作確認（2026-09-28）](target-binding-confidence-probe-20260928-ja.md)
- [候補別の一致・不一致・情報不足と出力確率の検証](candidate-match-validation-20260928-ja.md)

### 状態取得・短い質問・意味推論

- [概念・インスタンス・時点状態を分ける比較（2026-09-28）](instance-state-comparison-20260928-ja.md)
- [操作に紐づく単純な観測質問の比較](simple-action-comparison-20260928-ja.md)
- [問題の自動構築と1トークン回答の検証](one-token-comparison-20260928-ja.md)
- [物体ごとの階層クイズ比較（2026-09-28）](hierarchical-comparison-20260928-ja.md)
- [対象ごとの1トークン判定とYES後の説明の比較（2026-09-28）](object-gate-comparison-20260928-ja.md)
- [静止画の事前認識と対象名の比較（2026-09-27）](object-label-comparison-20260927-ja.md)
- [座標を要求しない「動いた物体」の認識比較](object-recognition-comparison-20260927-ja.md)
- [観測→比較→分類の比較検証（2026-09-28）](observe-compare-results-20260928-ja.md)
- [公式動画処理と相対関係質問の比較（2026-09-28）](official-relations-comparison-20260928-ja.md)
- [測定済みの変更・個体関係からQwenへルール推論を渡す比較](rule-inference-comparison-20260928-ja.md)
- [画面認識の採用方針・検証結果・参考文献](measured-perception-adoption-20260928-ja.md)

### 診断

- [認識から行動計画への引継ぎ分析（2026-09-28）](cognition-handoff-analysis-20260928-ja.md)
- [粗い中間説明の根拠を調べるXAI診断（2026-09-28）](explanation-xai-results-20260928-ja.md)
- [XAIによる移動見落としの診断案（2026-09-28）](xai-motion-diagnosis-20260928-ja.md)
- [XAIによる移動認識の診断結果（2026-09-28）](xai-motion-results-20260928-ja.md)

### 画像入力・観測方法

- [直前操作の発光・クリック位置表示の比較（2026-09-27）](action-cue-comparison-20260927-ja.md)
- [高速選択で対象を絞り、熟考で統合する構成の比較](attention-selection-comparison-20260927-ja.md)
- [不変部分を薄めて差分を強調する表示](change-emphasis-preview-20260927-ja.md)
- [差分の視覚的強調と白い対象の見やすさ：先行研究](change-emphasis-research-20260927-ja.md)
- [「幾何図形で構成されたゲーム」という事前説明の比較](geometry-context-comparison-20260927-ja.md)
- [Qwenへの入力形式と変更部分の認識：比較結果](motion-input-comparison-20260927-ja.md)
- [動きの対応候補と過去の役割説明：比較結果](motion-perception-comparison-20260927-ja.md)
- [動きと能動的知覚を使う画面認識：文献と改善案](motion-perception-research-ja.md)
- [座標を要求しない、動く物体の認識への方針修正](moving-object-recognition-research-20260927-ja.md)
- [重ね合わせ加工：暗色化と色混合の比較（2026-09-28）](overlay-ablation-comparison-20260928-ja.md)
- [Qwenの動き認識：学習・評価形式からの改善案](qwen-motion-input-research-20260927-ja.md)
- [相対関係の記述と差分境界画像の比較（2026-09-28）](relation-overlay-results-20260928-ja.md)
- [ARC PrizeのレトロUIを付けた認識比較](retro-ui-comparison-20260927-ja.md)
- [前後重ね合わせ3方式のQwen比較（2026-09-28）](temporal-overlay-comparison-20260928-ja.md)
- [BEFORE・AFTERを重ねる微小変化の表示試作](temporal-overlay-preview-20260928-ja.md)
- [動きの可視化による認識補助：先行研究と次の比較候補](temporal-overlay-research-20260928-ja.md)
- [微小変化をタイムバーと解釈するための調査](timer-recognition-research-ja.md)
- [4フレーム動画による移動認識の比較（2026-09-28）](video-comparison-20260928-ja.md)
- [差分を強調する5方式の比較](visual-emphasis-comparison-20260927-ja.md)
- [画面認識の比較結果：部分拡大と根拠付き出力](visual-recognition-comparison-20260927-ja.md)
- [ゲーム画面の認識改善：先行研究と比較実験案](visual-recognition-research-ja.md)

</details>
