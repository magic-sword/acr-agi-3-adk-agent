# SAM 3による対象抽出の検証準備（2026-09-30）

状態: **推論未実行。精度結果はまだない。**

`outputs/sam3-click-20260930/` に従来と同じ26問（一意18・否定8）と事前計画を固定した。原文の対象説明を使う比較を主評価とし、短い概念語による抽出は診断として別集計する。短い語では位置関係を省くため、正解候補を含むことと正しく選択することを混同しない。

` scripts/benchmark_sam3_click.py ` は公式Sam3Processorを使う実行準備コード。元画像を最近傍で6倍に拡大して入力し、返ったマスクを64×64に戻す。スコア最大のマスクを主選択、候補が1個の場合だけクリックする方式も補助集計する。正解注釈を使って最良マスクを選ぶことはしない。候補のいずれかに正解がある割合は別の診断指標。信頼度閾値は0.5に事前固定し、各要求で画像状態を初期化する。

専用テスト3件で座標復元、最高スコア選択と正解候補存在の分離、否定問題の採点を確認した。実モデルとの接続は未検証であり、環境が整った後に確認が必要。

現時点の障害:

- SAM 3パッケージ・重みがローカルにない。
- Hugging Faceへのログインは完了。ただしconfig取得は403で、アカウントがfacebook/sam3の許可リストに含まれていないという応答。モデルの利用申請・承認待ち。制限を回避するミラーは使用していない。
- GPU接続は復旧済み。ユーザー承認後に `docker compose restart dev` を実行し、CUDA利用可能・RTX A2000 12GB認識・GPU上の加算結果523776.0を確認。ホストGPUは正常で、先のホスト障害という記述はサンドボックス制限による誤判定だった。Hugging Face認証は再起動後も保持されているが、SAM 3取得は依然GatedRepoError。

既存CLIでの認証は、リポジトリのターミナルから `docker compose exec dev hf auth login`。トークンは対話入力し、チャットやコードには書かない。認証キャッシュはcomposeのHF_HOMEに保存される。

実行準備が整ったら、SAM 3公式コードを隔離環境に導入してから次を実行する。

```bash
docker compose exec -T dev python -m scripts.benchmark_sam3_click run \
  --output outputs/sam3-click-20260930 --checkpoint /path/to/authorized/sam3.pt
```

CPU指定も引数に用意したが、SAM 3のCPU動作・速度をこの環境で確認したわけではない。本番依存関係やランタイムは変更していない。

公式: https://github.com/facebookresearch/sam3 、 https://huggingface.co/facebook/sam3
