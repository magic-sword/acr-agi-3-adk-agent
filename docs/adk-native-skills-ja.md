# ADK 2.0ネイティブスキル接続

`LocalVisionLlm`はADKの`BaseLlm`を継承し、画像・テキストに加え、OpenAI互換APIのfunction callingを変換する。ADKがツールを実行し、その結果をモデルへ返す。アダプタ自身はゲーム操作やスクリプトを実行しない。

## 構成

```text
Workflow → 状態に対応したLlmAgent + SkillToolset
  → LocalVisionLlm → llama.cpp (--jinja) → Qwen3-VL-4B
  ← function_call ← tool_calls
  → ADKによるスキルツール実行
  → function_response → role=tool → Qwenの次の判断
```

S01〜S15を`agent/skills/<name>/SKILL.md`と`references/evidence-contract.md`へ移した。状態ごとに`STATE_SKILLS`に登録された候補だけを公開する。本文を一括でシステム指示に連結せず、LLMが`list_skills`でL1の名前・説明、`load_skill`でL2の手順、必要に応じて`load_skill_resource`でL3の資料を取得する。フォルダはホストで読み込まれるが、LLMへの提示は段階的に行う。

VERIFYやCOMMIT等の決定的処理は引き続きPythonで実行する。15スキル全部を毎手モデルで実行する構成ではない。モデルによるスキル選択の適切さ・指示への準拠は別の評価対象で、ツールを登録するだけでは保証されない。

## 接続契約と制限

- ツール宣言のJSON Schema、assistantの`tool_calls`、対応する`tool_call_id`付き結果を変換する。複数呼び出しの履歴もIDで対応付けるが、モデルへの要求は`parallel_tool_calls=false`とする。
- 不正な引数JSON、未登録ツール、重複ID、対応しない結果は拒否する。ゲーム操作は従来の操作検証・COMMITを経由する。
- 応答は非ストリーミング。画像入力は維持する。function_response内のマルチモーダルpartsは未対応で明示的に拒否する。現在のスキルはテキスト資料のみ。
- 1回の状態判断につき最大8 HTTPリクエスト。`begin_invocation()`で計数をリセットし、既存の判断回数・ゲーム全体の時間上限も維持する。スキル読み込みも推論予算を消費する。
- `SkillToolset`にはスクリプト実行用ツールもあるが、今回の15スキルに実行スクリプトは同梱しない。スクリプト実行や動的追加ツールの実モデル検証は今回の範囲外。
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

## ゲーム統合確認

ls20を1手・判断時間60秒・プロセス上限80秒に制限し、約8.73秒で1手を実行、手数上限で正常終了した。OBSERVEとPROBEの2判断で計8 HTTPリクエスト、list_skills / load_skill / load_skill_resourceを各2回記録し、検証エラーは0件。ログは`outputs/evaluations/20260924T095948206701Z/`。

最初の統合試行では既存のADK `max_llm_calls=3`が読み込みだけで尽きたため、状態判断数とHTTP往復の上限を分けて修正した。このケースを含む45件のテストが通過している。既定は最大3判断×各8 HTTPリクエストで、既存のゲーム時間上限も適用する。

この1手の観測応答ではfactsとgoalが空のままで、レベルクリアは0。接続は動作したが、知覚・仮説更新・計画の品質改善は未確認。Kaggleへの提出は行っていない。
