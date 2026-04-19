# factfull — 統合ロードマップ

作業ブランチ: `feat/unified-knowledge-pipeline`

---

## Phase 0 — 整理・地図作成 ✅

- [x] `critique.py` — 批評的読みパス追加
- [x] `docs/system-architecture.md` — 5層アーキテクチャ定義
- [x] `docs/roadmap.md` — このファイル
- [x] `factfull/publishers/homupe.py` — `homupe/pipelines/publish.py` を移植
- [ ] `homupe/pipelines/` — 移植完了後に削除（homupe 側で確認後）

---

## Phase 1 — 共有基盤の整備 ✅

目標: すべての ingest/process モジュールが共通インターフェースを使う

- [x] `factfull/core/types.py` — `SourceDoc` / `ProcessedDoc` / `Triple` / `Entity` データクラス定義
- [x] `factfull/llm.py` — Ollama / Anthropic のみ対応（OpenAI 不要）
- [x] `factfull/ingest/chunker.py` — テキスト分割ロジックを一本化（chunker.py に委譲、indexer.py 後方互換維持）
- [x] `PipelineResult.to_processed_doc()` — 既存 podcast パイプラインのアダプター追加

---

## Phase 2 — Ingestion 拡張

目標: Podcast 以外のソースを factfull に取り込む

- [x] `factfull/ingest/paper.py` — arXiv / PDF 取り込み（`kg-builder` + `infoseeker` から移植）
  - PDF テキスト抽出（pdfplumber）、arXiv ダウンロード、キーワード検索 → SourceDoc
- [x] `factfull/ingest/book.py` — Gutenberg/URL/ローカルファイル → SourceDoc（cogito から移植）
- [x] `factfull/ingest/web.py` — URL → HTML 取得・本文抽出 → SourceDoc（新規）
- [x] `factfull/extract/entity.py` — エンティティ抽出（kg-builder から移植・汎用化）
- [x] `factfull/extract/relation.py` — 関係抽出 / トリプル生成（kg-builder から移植・汎用化）

---

## Phase 3 — Knowledge Store 統合

目標: 生成した SourceDoc / ProcessedDoc を L3 に自動書き込み

- [ ] `factfull/graph/neo4j.py` — Neo4j クライアント（`kg-builder/graph/neo4j_client.py` から移植）
- [ ] pipeline 完了時に triples を Neo4j へ自動書き込み
- [ ] pipeline 完了時に summary を localsearch-mcp が監視するディレクトリへ自動配置

---

## Phase 4 — Synthesis（Plan A: 複数ソース横断）

目標: 同テーマの複数ソースから独自記事を生成する

- [ ] `factfull/synthesis/cross_source.py` — Synthesis エンジン
  - 同テーマ SourceDoc 群を受け取り「合意・矛盾・欠落」マップを生成
  - Map-Reduce で横断要約 → critique → editorial の流れ
- [ ] `factfull/synthesis/tracker.py` — 予測・発言の時系列追跡（Plan B）
- [ ] `pipelines/synthesis.py` — Synthesis パイプラインの CLI エントリポイント

---

## Phase 5 — 既存リポジトリの廃止・統合完了

- [ ] `kg-builder` — L1/L2 機能を factfull に移植済み確認後、KG 管理 API のみ残置
- [ ] `infoseeker` — 収集ロジックを `factfull/ingest/paper.py` に統合、リポジトリ廃止
- [ ] `docchat` — RAG ロジックを `factfull/synthesis/` に統合、Gradio UI 廃止
- [ ] `graphragmi` — 判断保留（Microsoft フレームワーク依存度次第）

---

## 各フェーズの依存関係

```
Phase 0 ──→ Phase 1 ──→ Phase 2 ──→ Phase 3
                                         ↓
                              Phase 4 ←──┘
                                         ↓
                              Phase 5
```

---

## 参照

- アーキテクチャ詳細: [system-architecture.md](system-architecture.md)
- 技術詳細（既存）: [architecture.md](architecture.md)
