# factfull — システムアーキテクチャ（統合版）

## 概要

factfull を中心に、複数ソースの知識を統合・合成・公開するパイプライン群。

```
[L1: Ingestion]  → [L2: Processing] → [L3: Knowledge Store]
                                              ↓
[L5: Consumption] ←── [L4: Synthesis] ───────┘
```

---

## 5層アーキテクチャ

### L1: Ingestion（ソース別取り込み）

ソースを問わず `SourceDoc` を出力する。

```python
@dataclass
class SourceDoc:
    source_type: str        # "podcast" | "paper" | "book" | "web"
    source_id: str          # video_id / DOI / ISBN / URL
    title: str
    text: str               # 生テキスト（英語または原文）
    text_ja: str            # 日本語テキスト（翻訳済み）
    chunks: list[str]       # チャンク分割済み
    metadata: dict          # ソース固有メタデータ
    created_at: str         # ISO 8601
```

| モジュール | ソース | 移植元 |
|-----------|--------|--------|
| `factfull/ingest/podcast.py` | YouTube / Podcast | 現行 `podcast/archiver.py` |
| `factfull/ingest/paper.py` | arXiv / PDF | `kg-builder` |
| `factfull/ingest/book.py` | 書籍テキスト / EPUB | `cogito` (book-research) |
| `factfull/ingest/web.py` | URL / HTML | 新規 |

---

### L2: Processing（ソース非依存の変換）

`SourceDoc` を受け取り `ProcessedDoc` を返す。

```python
@dataclass
class ProcessedDoc:
    source: SourceDoc
    summary: str            # 日本語要約記事
    triples: list[Triple]   # (subject, predicate, object)
    entities: list[Entity]  # 固有名詞・概念
    score: float            # ファクトチェックスコア (0–100)
```

| モジュール | 機能 | 移植元 |
|-----------|------|--------|
| `factfull/process/summarizer.py` | 要約生成（Map-Reduce） | 現行 `podcast/archiver.py` |
| `factfull/process/extractor.py` | エンティティ抽出 | `kg-builder/extractor/entity_extractor.py` |
| `factfull/process/relations.py` | 関係抽出（トリプル） | `kg-builder/extractor/relation_extractor.py` |
| `factfull/process/factcheck.py` | ファクトチェックループ | 現行 `pipeline.py` 内 |

---

### L3: Knowledge Store（永続化）

| ストア | 用途 | 管理リポジトリ |
|--------|------|---------------|
| Neo4j | エンティティ・関係グラフ | `kg-builder`（独立維持） |
| Chroma (vector) | 意味検索 | `localsearch-mcp`（独立維持） |
| BM25 index | キーワード検索 | `localsearch-mcp`（独立維持） |
| `~/podcasts/` | エピソードファイル群 | factfull（現行） |

`localsearch-mcp` は MCP プロトコルサーバーのため**独立プロセスとして分離を維持**。
factfull が生成した記事・SourceDoc を localsearch-mcp が索引化する関係。

---

### L4: Synthesis（合成・生成）

複数の `ProcessedDoc` を横断して新しいコンテンツを生成する。

| モジュール | 機能 | フェーズ |
|-----------|------|---------|
| `factfull/synthesis/critique.py` | 批評的分析 | 実装済み |
| `factfull/synthesis/cross_source.py` | 複数ソース横断記事 | Plan A |
| `factfull/synthesis/tracker.py` | 予測・発言の時系列追跡 | Plan B |

---

### L5: Consumption（出力先）

| モジュール | 出力先 | 移植元 |
|-----------|--------|--------|
| `factfull/publishers/homupe.py` | MkDocs ブログ記事 | `homupe/pipelines/publish.py` |
| `factfull/publishers/twitter.py` | X (Twitter) 投稿 | 同上（Selenium部分） |
| `localsearch-mcp` | MCP エージェント向け検索 | 独立維持 |

---

## リポジトリ整理方針

### factfull に統合するもの

| リポジトリ | 統合対象 | 除外（残置） |
|-----------|---------|------------|
| `kg-builder` | ingest/paper, extract/, graph/neo4j | FastAPI サーバー、可視化 |
| `infoseeker` | arXiv 収集ロジック | Slack 通知（不要なら廃止） |
| `docchat` | RAG 検索ロジック | Gradio UI（廃止候補） |
| `graphragmi` | GraphRAG クエリ | Microsoft フレームワーク依存部（要判断） |
| `cogito`(book-research) | ingest/book, ConceptGraph 生成 | VOICEVOX 音声生成（独立維持） |
| `homupe/pipelines/` | publish.py → publishers/ | |

### 独立維持するもの

| リポジトリ | 理由 |
|-----------|------|
| `localsearch-mcp` | MCP プロトコルサーバー（別プロセス必須） |
| `homupe` | サイトコンテンツ・Cloudflare Pages デプロイ対象 |
| `nanoclaw` | メッセージング層（別ランタイム） |
| `cogito` | VOICEVOX 音声生成パイプライン（異質な依存関係） |

---

## データフロー全体図

```
YouTube URL ─────────────────────────────────────────────────┐
arXiv DOI / PDF ─────────────────────────────────────────────┤
書籍テキスト ────────────────────────────────────────────────┤
Web URL ──────────────────────────────────────────────────── ┤
                                                             ↓
                                               [L1: Ingestion]
                                                SourceDoc
                                                             ↓
                                              [L2: Processing]
                                   summary / triples / entities
                                                  ↙         ↘
                                    [L3a: Neo4j]    [L3b: Chroma+BM25]
                                    entities/triples  full-text search
                                                  ↘         ↙
                                              [L4: Synthesis]
                                     cross-source article / critique
                                                             ↓
                                    ┌────────────────────────┤
                                    ↓                        ↓
                             [homupe blog]          [localsearch-mcp]
                             Markdown 記事          エージェント検索
```

---

## 共有基盤

現在は各リポジトリが独自実装しているが、factfull に一本化する:

| 機能 | 現状 | 統合後 |
|------|------|--------|
| LLM クライアント | llm.py / llm_client.py / LangChain | `factfull/llm.py`（現行を拡張） |
| テキスト分割 | indexer.py / chunking.py / pdf_extractor | `factfull/ingest/chunker.py` |
| BM25 検索 | factfull/retriever.py | `factfull/retriever.py`（現行） |
