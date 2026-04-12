# factfull

Truth ソース群に対してドキュメントのクレームを検証するファクトチェッカー。

## アーキテクチャ

```
Truth Sources → チャンク化 → BM25 インデックス
Target Doc → Claim 抽出 (LLM) → 各 Claim で BM25 検索 → 証拠パッセージ取得
                                                         → LLM 判定
                                                         → Markdown レポート
```

**判定種別:**
| 記号 | 値 | 意味 |
|------|----|------|
| ✅ | supported | Truth ソースに支持されている |
| ❌ | contradicted | Truth ソースと矛盾している |
| ⚠️ | partial | 部分的に正しい / 誇張・誤記あり |
| ❓ | unverifiable | 証拠が不十分 |

**信頼度スコア** = (supported × 1.0 + partial × 0.5) / verifiable件数 × 100

## セットアップ

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## 使い方

```bash
# Ollama（デフォルト）
python run.py \
  --truth transcript.txt \
  --target summary.md \
  --output report.md

# ディレクトリ全体を Truth ソースに
python run.py \
  --truth truth_docs/ \
  --target article.md \
  --output report.md

# Anthropic API を使う場合
FACTFULL_LLM_BACKEND=anthropic \
ANTHROPIC_API_KEY=sk-... \
python run.py --truth transcript.txt --target summary.md
```

## 環境変数

| 変数 | デフォルト | 説明 |
|------|------------|------|
| `FACTFULL_LLM_BACKEND` | `ollama` | `ollama` または `anthropic` |
| `FACTFULL_OLLAMA_URL` | `http://localhost:11434/api/generate` | Ollama エンドポイント |
| `FACTFULL_OLLAMA_MODEL` | `glm-4.7-flash:latest` | 使用するモデル |
| `FACTFULL_ANTHROPIC_MODEL` | `claude-sonnet-4-6` | Anthropic モデル ID |

## ポッドキャストパイプラインとの連携

```bash
python run.py \
  --truth output/podcasts/VIDEO_ID/transcript_en.txt \
  --target output/podcasts/VIDEO_ID/summary_ja.md \
  --output output/podcasts/VIDEO_ID/fact_check.md
```
