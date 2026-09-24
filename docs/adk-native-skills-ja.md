# ADK 2.0ネイティブスキル接続

`LocalVisionLlm`はADKの`BaseLlm`を継承し、画像・テキストに加え、OpenAI互換APIのfunction callingを変換する。ADKがツールを実行し、その結果をモデルへ返す。アダプタ自身はゲーム操作やスクリプトを実行しない。

応答の変換は`QwenToolCallAdapter`へ分離した。構造化された`tool_calls`とQwen3-VLの`<tool_call>`形式の両方をADKへ接続する。[形式・実行制約・検証結果](qwen-adk-tool-protocol-ja.md)を参照。

## 構成

```text
Workflow → 状態に対応したLlmAgent + SkillToolset
  → LocalVisionLlm → llama.cpp (--jinja) → Qwen3-VL-4B
  ← function_call ← tool_calls
  → ADKによるスキルツール実行
  → function_response → role=tool → Qwenの次の判断
```

S01〜S15を`agent/skills/<name>/SKILL.md`と`references/evidence-contract.md`へ移した。状態ごとに`STATE_SKILLS`に登録された候補だけを公開する。本文を一括でシステム指示に連結せず、LLMが`list_skills`でL1の名前・説明、`load_skill`でL2の手順、必要に応じて`load_skill_resource`でL3の資料を取得する。フォルダはホストで読み込まれるが、LLMへの提示は段階的に行う。

OBSERVE・UPDATE・COMMIT等の決定的処理はPythonで実行する。VERIFYは視覚的解釈とPythonの述語照合を組み合わせる。15スキル全部を毎手モデルで実行する構成ではない。モデルによるスキル選択の適切さ・指示への準拠は別の評価対象で、ツールを登録するだけでは保証されない。

## 状態ごとのスキル接続

候補の登録と本文の読み込みを区別する。各状態には必要になり得る推論スキルを登録し、LLMが必要な本文・参照資料だけをロードする。登録数の削減を理由に推論機能を省かない。

|状態・用途|公開するスキル|数|
|---|---|---:|
|OBSERVE / UPDATE / ACT / COMMIT / CONSOLIDATE / RECOVER|ホスト処理のみ|0|
|VERIFY|S01 / S02 / S03 / S04 / S10|5|
|PLAN|S01 / S02 / S04 / S05 / S08 / S09 / S10 / S11 / S12 / S14|10|
|PROBE|S01 / S02 / S04 / S07 / S10 / S12 / S15|7|
|REVISE|S01 / S02 / S04 / S05 / S06 / S07 / S08 / S09 / S10 / S11 / S12 / S15|12|

OBSERVEは画像・観測ID・操作対応の記録だけを行う。VERIFYはInterpretation、PLAN・PROBE・REVISEは必要なInterpretationを含むProposalを返す。目標未確定でもPLANへ進む。REVISEは独立した推論状態で、旧実装の遷移先との動的な合成は廃止した。

全モデル状態に`observe_current`、`list_observations`、`get_observation`、`compare_observations`、`observe_animation`、`move_cursor`を公開する。画像付きの取得結果はテキストへbase64を埋め込まず、画像入力としてモデルへ返す。呼出しは外部ゲームを進めない。

モデルログの`state`と`available_skills`で候補、`exchanges`で実際の読込・観測参照を確認する。登録済みでも未使用のスキルがあることは正常であり、全候補を毎回ロードしない。現行実装では関連する最大2スキルの選択を指示する。最終HTTP枠では`submit_plan`／`submit_revision`／`submit_experiment`／`submit_interpretation`のうち、その状態の完了ツールだけを公開して`tool_choice=required`とする。受理した提出は追加のモデル応答なしでWorkflowへ戻る。

S01〜S12、S14、S15は実際に呼び出されるLLM状態へ接続済み。S13（決定の確定・記録）だけはCOMMITのPython処理が担い、モデルへ登録しない。S13の定義ファイルは保持する。その他のスキルについて、関数による処理があることだけを理由に意味解釈・計画判断の候補から除外しない。

## 接続契約と制限

- ツール宣言のJSON Schema、assistantの`tool_calls`、対応する`tool_call_id`付き結果を変換する。複数呼び出しの履歴もIDで対応付けるが、モデルへの要求は`parallel_tool_calls=false`とする。
- 不正な引数JSON、未登録ツール、重複ID、対応しない結果は拒否する。ゲーム操作は従来の操作検証・COMMITを経由する。
- 応答は非ストリーミング。画像入力は維持する。function_response内のマルチモーダルpartsは未対応で明示的に拒否する。画像付きツール結果は専用フィールドから通常の画像メッセージへ変換する。
- 1回の状態判断につき最大8 HTTPリクエスト。`begin_invocation()`で計数をリセットし、既存の判断回数・ゲーム全体の時間上限も維持する。スキル読み込みも推論予算を消費する。
- `SkillToolset`にはスクリプト実行用ツールがある。視覚描画・操作変換のスクリプトもパッケージに含めるが、ゲーム操作の実行権限は与えない。スクリプト実行や動的追加ツールの実モデル検証は今回の範囲外。
- スキル本文は各判断の会話内で利用する。全ゲームの会話履歴を無制限に持ち越す構成にはせず、再判断では再ロードが発生し得る。
- ローカルとノートブック内のllama.cpp起動に`--jinja`を追加。ノートブック生成器はスキルのMarkdown資料も同梱する。

## ログと確認

`cognition/*.model.jsonl`の各判断記録に`http_requests`と`exchanges`を追加した。exchangesには各HTTP呼び出しのツール名・引数、生応答、ツール結果、usage、所要秒数を記録する。上位usageはその判断内の合計。集計の`model_calls`は状態判断の呼び出し数、`model_http_requests`はツール往復を含むHTTP回数、`skill_tool_calls`はツール別の要求数で、成功数ではない。

```bash
# 実モデルで一覧→本文→参照資料の接続を確認（最大60秒）
docker compose up -d vlm
docker compose run --rm --no-deps dev python scripts/check_skills.py

# 単体・統合テスト
docker compose run --rm --no-deps dev python -m unittest discover -s tests -v
```

実際のQwen3-VL-4Bで3段階のツール呼び出しと、参照資料にだけ存在するランダムな検証値の回答を確認済み。記録は`outputs/skill-checks/c053c7446c2e499a891a9b9084c9b179/result.json`。この確認はモデルがツール結果を読めることを検証し、ゲーム攻略能力の改善は主張しない。

回帰テストでは、ADK Runnerによる実ツール実行、本文・資料がロード前のリクエストに含まれないこと、結果ID、異常応答、往復上限、提出用パッケージへの同梱を確認する。

参考: [ADK Skills](https://adk.dev/skills/)、[llama.cpp function calling](https://github.com/ggml-org/llama.cpp/blob/master/docs/function-calling.md)。

## 改訂前のゲーム統合確認

ls20を1手・判断時間60秒・プロセス上限80秒に制限し、約8.73秒で1手を実行、手数上限で正常終了した。OBSERVEとPROBEの2判断で計8 HTTPリクエスト、list_skills / load_skill / load_skill_resourceを各2回記録し、検証エラーは0件。ログは`outputs/evaluations/20260924T095948206701Z/`。

最初の統合試行では既存のADK `max_llm_calls=3`が読み込みだけで尽きたため、状態判断数とHTTP往復の上限を分けて修正した。このケースを含む45件のテストが通過している。既定は最大3判断×各8 HTTPリクエストで、既存のゲーム時間上限も適用する。

この1手の観測応答ではfactsとgoalが空のままで、レベルクリアは0。接続は動作したが、知覚・仮説更新・計画の品質改善は未確認。Kaggleへの提出は行っていない。

現行の観測分離後の検証は[リファクタ検証記録](local-evaluation-cognitive-refactor-ja.md)を参照。
