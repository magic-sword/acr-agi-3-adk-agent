# ADK 2.0 スキル接続と責務

更新日: 2026-09-24。[設計レビュー](adk-design-review-20260924-ja.md)に基づき、状態の必須指示、ホスト制御、任意の推論スキルを分離した。専門手順の一覧は[スキル仕様](adk-cognitive-skills-ja.md)を参照。

## 実行構造

```text
Workflow（遷移・ホスト検証・確定）
  → 状態別LlmAgent（reasoner_plan / probe / revise / verify）
      instruction: 状態の課題・必須契約
      tools: 完了ツール + 観測ツール + ReasoningSkillToolset
          L1: 名前・用途を初回から提示
          L2: 必要な本文だけ load_skill
          L3: 必要な固有事例だけ load_skill_resource
  → LocalVisionLlm → Qwen3-VL
```

`agent/cognition/instructions.py` が状態の目的・出力契約を担当する。`skills.py` は13個の任意の専門手順と、その状態別の公開範囲を担当する。スキルを読まなくても、観測ツールを使って有効な結果を提出できる。状態機械がスキルの順番を実行する構成ではない。

`LocalVisionLlm` はADKの `BaseLlm` を継承し、画像・テキスト・function callingをローカルAPIへ変換する。ADKがツールを実行して結果を返す。[Qwenプロトコルの接続仕様](qwen-adk-tool-protocol-ja.md)も参照。

## 標準機能とプロジェクト側の設定

`ReasoningSkillToolset` はADK `SkillToolset` のサブクラス。標準の `load_skill_from_dir`、本文・資料ロード、結果イベント、スキル有効化の状態管理を利用する。公開フックの `get_tools` と `process_llm_request` で次だけを変更する。

- 状態内で使えるスキルの名前・用途だけをJSONカタログとして最初から渡す。本文と資料はロードまでモデルへ渡さない。
- `list_skills` の往復を省く。任意の `load_skill` と `load_skill_resource` のみを公開する。
- スクリプト実行を広告する標準案内を、このプロジェクトで利用可能な知識ロードの案内に置き換える。
- スキルに `scripts` や `adk_additional_tools` が追加された場合は構築時に拒否する。接続しない機能を黙って公開しない。

ADK本体のプライベート属性は変更しない。この2ツール限定・任意ロードの方針は本プロジェクトの設定であり、ADK共通の必須仕様ではない。

## 状態ごとの公開範囲

|状態|スキルID|数|
|---|---|---:|
|VERIFY|S01 / S02 / S03 / S04 / S10|5|
|PLAN|S01 / S02 / S04 / S05 / S08 / S09 / S10 / S11 / S12 / S14|10|
|PROBE|S01 / S02 / S04 / S07 / S10 / S12|6|
|REVISE|S01 / S02 / S04 / S05 / S06 / S07 / S08 / S09 / S10 / S11 / S12|11|
|OBSERVE / UPDATE / ACT / COMMIT / CONSOLIDATE / RECOVER|ホスト処理|0|

旧S13（決定確定）とS15（予算制御）はスキル定義から除外し、ホスト責務として扱う。残りのIDは既存の分析との対応のため保持する。登録数は技能の実証数ではなく、利用可能な手順の数である。

## ツールとホスト実装

全モデル状態に `observe_current`、`list_observations`、`get_observation`、`compare_observations`、`observe_animation`、`move_cursor` を直接登録する。スキルのロード有無で観測機能は変わらない。画像は画像入力として返し、外部ゲームは進めない。

描画は `agent/rendering.py`、操作名変換は `agent/controls.py` に配置した。どちらもホストの通常のPython実装で、スキルの実行ファイルではない。CLIは次のとおり。

```bash
python agent/rendering.py current EVENT.json current.png
python agent/rendering.py replay EVENT.json history.gif
```

`run_skill_script` は公開しない。以前の構成はexecutorなしでこのツールを公開していたため `NO_CODE_EXECUTOR` を返していた。将来、実行スキルが必要になった場合は、知識スキルとは別にexecutor・依存関係・入出力を実装して検証する。

PLAN／REVISEには `check_plan_order` も直接登録する。`initial_conditions`、順序付き `steps`（requires/adds/removes）、`required_final_conditions` を受け、最初に欠ける前提または未達の最終条件と計算過程を返す。入力が真実かを判断せず、ゲーム操作・記憶更新は行わない。S08は条件を破壊する操作順の検査にこのツールを使う。

## 予算・提出・保存

- 各状態の必須契約は `instructions.py`、型と参照の検証は完了ツールとホスト、遷移はWorkflowが所有する。スキル本文には状態完了の共通定型文を複製しない。
- 1判断につき最大8 HTTP要求。最終枠はその状態の完了ツールだけを公開し、`tool_choice=required` にする。受理時は追加のモデル応答なしでWorkflowへ戻る。
- スキルを必ず読む指示と「最大2個」というソフトな制限は撤廃した。必要な専門手順だけ選び、証拠取得と提出に予算を残す。全体の要求数・時間・判断数はホストが強制する。
- `include_contents="none"` により、本文の再利用は現在の判断内に限る。前回ロード済みというだけで次回に本文があるとは扱わない。
- 認知記憶はCOMMITの `Event(state=...)` で公開する。作業中の `self.turn`、提出結果、観測ストアはプロセス内にあり、障害後の自動再開は保証しない。

## 確認方法

```bash
docker compose run --rm --no-deps dev python scripts/check_skills.py
docker compose run --rm --no-deps dev python -m unittest discover -s tests -v
docker compose run --rm --no-deps dev python scripts/check_reasoning_skills.py
make visualize
```

`check_skills.py` は本番と同じtoolsetを使い、参照資料にだけ存在するランダム値を実モデルが回答できるかを確認する。新構成の接続確認では `load_skill` → `load_skill_resource` → 回答の3 HTTP要求で成功した。記録: `outputs/skill-checks/dd11f4bbc42a4f3aa83a9ebed4645e92/result.json`。

回帰テストはL1の初回提示、L2/L3の遅延ロード、結果ID、公開ツールの限定、状態外スキルの拒否、未接続実行機能の構築時拒否、ロードなしでの有効提出、上限、ノートブック同梱を確認する。これらはゲーム攻略能力の実証とは区別する。

公式資料: [Skills](https://adk.dev/skills/)、[SkillToolset 2.0.0](https://github.com/google/adk-python/blob/v2.0.0/src/google/adk/tools/skill_toolset.py)。

再設計後の85件の回帰テスト、実モデルの手順評価とゲーム測定の結果は[検証記録](local-evaluation-skill-redesign-ja.md)を参照。ツール利用と計画推論には未解決のモデル挙動がある。
