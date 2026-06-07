# pipeline-supervisor — 自己修復型ナイトリーパイプライン監督スキル

このスキルは `claude -p` ヘッドレスモードでの実行を想定している。
launchd / cron からトリガーされ、factfull のナイトリーパイプラインを監督付きで走らせる。

監督の中身（preflight・失敗分類・Ollama 再起動・--regen 再試行・strict ビルド
gate・メール通知）は `supervisor/nightly_supervisor.py` に実装済み。この
supervisor は既存の `nightly_pipeline.py`（arXiv + RSS 自動検出 → 記事生成 →
homupe 投稿）を **`--push` なしで** 包み、push 前に必ず `mkdocs build --strict`
を通す。エピソード検出ロジックは nightly_pipeline.py 側にあり、ここでは二重実装
しない。

## 実行方法

```bash
cd /path/to/factfull
claude -p "$(cat .claude/skills/pipeline-supervisor/SKILL.md)" \
  --allowedTools Bash,Edit,Task
```

---

## 監督手順

### Phase 0: Preflight

```bash
python supervisor/health.py
```

- ✅ 正しい venv（`.venv/bin/python` が PATH に現れること）
- ✅ 必須 env vars（PODCAST_OUTPUT_DIR / OLLAMA_URL / FACTFULL_LLM_BACKEND / FACTFULL_OLLAMA_MODEL）
- ✅ PODCAST_OUTPUT_DIR が実在
- ✅ Ollama が `"done":true` を返す

ひとつでも失敗したら supervisor は自動でエスカレーションメールを送って終了する。

### Phase 1: パイプライン実行（自己修復つき）

`nightly_supervisor.py` を走らせるだけでよい。SIGPIPE 安全（`head`/`tail` に
パイプしない、`tee` でログに複製する）:

```bash
python supervisor/nightly_supervisor.py 2>&1 \
  | tee supervisor/logs/nightly-$(date +%Y%m%d).log
```

supervisor が内部で行う自己修復（あなたが手動でやる必要はないが、挙動の把握用）:

| 失敗パターン | 症状 | 対処 |
|---|---|---|
| Ollama timeout / SIGPIPE | `connection refused` / `broken pipe` | `pkill -f "ollama runner"` → 8秒待機 → 再実行（最大3回）|
| JSONDecodeError | `json.decoder.JSONDecodeError` | `--regen` を付けて再実行（1回）|
| ImportError | `ModuleNotFoundError` | venv パスを記録してエスカレーション |
| auth / 401 | `401` / `unauthorized` | 即エスカレーション（自動修復不可）|
| rate-limit | `rate limit` / `usage limit` | クリーン停止してエスカレーション |

supervisor が Escalate で終了した（rc≠0）場合のみ、ログ末尾を読んで根本原因を
判断し、env / venv の修正など人手が要る対応を行うこと。自動修復可能な範囲は
supervisor が既に試行済みなので、同じことを繰り返さない。

**よく使うオプション:**
```bash
python supervisor/nightly_supervisor.py --dry-run       # preflight + 検出のみ
python supervisor/nightly_supervisor.py --max 1         # 1 件だけ
python supervisor/nightly_supervisor.py --channel lex_fridman
python supervisor/nightly_supervisor.py --skip-arxiv
```

### Phase 2: Publish

supervisor が自動で行う:

1. homupe の `docs/blog/posts/ docs/data/kg/ docs/data/synthesis/` に差分があるか確認
2. `uv run mkdocs build --strict`（失敗したら push せずエスカレーション）
3. `git add` → `git commit` → `git push origin main`

手動で publish 部分だけ確認したいときは homupe 側で:
```bash
cd $HOMUPE_DIR && uv run mkdocs build --strict
```

### Phase 3: 通知

supervisor が自動で送る（Telegram プッシュ + `supervisor/logs/notify.log` 記録。
Telegram 未設定時は macOS 通知にフォールバック）:
- ✅ publish 完了（成功）→ `notify_success`
- ✅ 自動修復不能な失敗 → `notify_escalation`
- ❌ 正常な途中経過・Ollama 再起動・regen リトライ → 送らない

手動でテスト送信:
```bash
python supervisor/notify.py --test
```

---

## 再開について

エピソードのチェックポイントは `nightly_pipeline.py` の `registry.json` が管理する。
rate-limit 等で中断しても、次回 `python supervisor/nightly_supervisor.py` を再実行
すれば未処理 (`pending`) から再開され、`done` 済みはスキップされる。supervisor 側に
別途の state ファイルは持たせていない。

---

## 環境変数リファレンス

| 変数 | 用途 | 必須 |
|---|---|---|
| `PODCAST_OUTPUT_DIR` | transcript/metadata の格納先 | ✅ |
| `OLLAMA_URL` | Ollama エンドポイント（`localhost`。Docker でない限り `host.docker.internal` 不可） | ✅ |
| `FACTFULL_OLLAMA_MODEL` | 使用モデル（e.g. `gemma4:e4b`） | ✅ |
| `FACTFULL_LLM_BACKEND` | `ollama` 固定 | ✅ |
| `HOMUPE_DIR` | homupe リポジトリパス | ✅（publish時） |
| `TELEGRAM_BOT_TOKEN` | bot トークン（nanoclaw の @Nanoikm_bot を流用） | ✅（通知時） |
| `TELEGRAM_CHAT_ID` | 送信先 chat id（DM は `8552913958`） | ✅（通知時） |
