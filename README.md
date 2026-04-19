# factfull

Truth ソース群に対してドキュメントのクレームを検証・自動修正する LLM ベースのファクトチェックライブラリ。
ポッドキャスト翻訳記事生成パイプライン（`factfull.podcast`）を内蔵しています。

## アーキテクチャ

```
Truth Sources → チャンク化 → BM25 インデックス
Target Doc → Claim 抽出 (LLM) → BM25 検索 → 証拠パッセージ取得
                                             → LLM 判定
                                             → Markdown レポート
                                             → 外科的修正（corrector）
                               ↑__________________________|
                               スコア < 閾値なら再ループ
```

**判定種別:**

| 記号 | 値 | 意味 |
|------|----|------|
| ✅ | supported | Truth ソースに支持されている |
| ❌ | contradicted | Truth ソースと矛盾している |
| ⚠️ | partial | 部分的に正しい / 誇張・誤記あり |
| ❓ | unverifiable | 証拠が不十分 |

**信頼度スコア** = (supported × 1.0 + partial × 0.5) / verifiable 件数 × 100

詳細は [`docs/architecture.md`](docs/architecture.md) を参照。

## セットアップ

```bash
# 通常インストール
pip install -e .

# ポッドキャスト機能も使う場合
pip install -e ".[podcast]"

# Anthropic バックエンドも使う場合
pip install -e ".[anthropic]"
```

他リポジトリから依存する場合（uv / pyproject.toml）:

```toml
dependencies = [
    "factfull @ file:///path/to/factfull",
]
```

## 使い方

### 1. 単発ファクトチェック（`run.py`）

```bash
python run.py \
  --truth transcript.txt \
  --target summary.md \
  --output report.md

# ディレクトリ全体を Truth ソースに
python run.py \
  --truth truth_docs/ \
  --target article.md

# Anthropic API を使う場合
FACTFULL_LLM_BACKEND=anthropic ANTHROPIC_API_KEY=sk-... \
python run.py --truth transcript.txt --target summary.md
```

### 2. 自己改善ループ（`refine_loop.py`）

スコア閾値を超えるまで「ファクトチェック → 修正」を繰り返す。

```bash
python refine_loop.py \
  --truth ~/podcasts/VIDEO_ID/transcript_en.txt \
  --target ~/podcasts/VIDEO_ID/summary_ja.md \
  --threshold 95 \
  --max-iter 5 \
  --editorial \
  --editorial-model gemma4:e4b
```

### 3. ポッドキャスト記事生成パイプライン

YouTube URL から日本語記事を自動生成する。

```python
from factfull.podcast.pipeline import PipelineConfig, run_pipeline

config = PipelineConfig(
    translate_model="translategemma:12b",
    analyze_model="gemma4:26b",
    factcheck_model="gemma4:e4b",
    threshold=95.0,
    max_iter=5,
    blog_name="SoryuNews",
    reader_persona="英語圏情報にアクセスしたい日本語話者のエンジニア・研究者",
)

result = run_pipeline(config, "https://www.youtube.com/watch?v=VIDEO_ID")
# result.summary_path  → ~/podcasts/VIDEO_ID_.../summary_ja.md
# result.score         → ファクトチェックスコア (0–100)
# result.title         → YouTube タイトル（英語）
# result.episode_dir   → エピソードディレクトリ
```

Pass 1（チャンク別要点抽出）をスキップして記事だけ再生成したい場合は `regen=True`:

```python
result = run_pipeline(config, url, regen=True)
```

## 環境変数

| 変数 | デフォルト | 説明 |
|------|------------|------|
| `FACTFULL_LLM_BACKEND` | `ollama` | `ollama` または `anthropic` |
| `FACTFULL_OLLAMA_URL` | `http://localhost:11435/api/generate` | Ollama エンドポイント |
| `FACTFULL_OLLAMA_MODEL` | `glm-4.7-flash:latest` | デフォルトモデル |
| `FACTFULL_ANTHROPIC_MODEL` | `claude-sonnet-4-6` | Anthropic モデル ID |
| `ANTHROPIC_API_KEY` | — | Anthropic バックエンド使用時に必要 |

## パッケージ構成

```
factfull/
├── claim_extractor.py   # LLM によるクレーム抽出
├── corrector.py         # セクション単位の外科的修正
├── editorial.py         # 編集後記生成
├── indexer.py           # BM25 インデックス構築
├── llm.py               # LLM バックエンド抽象化（Ollama / Anthropic）
├── reporter.py          # Markdown レポート生成
├── retriever.py         # BM25 証拠検索
├── verifier.py          # LLM 判定
└── podcast/
    ├── archiver.py      # YouTube メタデータ取得・翻訳・記事生成（Map-Reduce）
    └── pipeline.py      # PipelineConfig / PipelineResult / run_pipeline()
```
