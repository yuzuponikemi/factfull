# factfull — 統合ロードマップ

---

## Phase 0 — 整理・地図作成 ✅

- [x] `critique.py` — 批評的読みパス追加
- [x] `docs/system-architecture.md` — 5層アーキテクチャ定義
- [x] `factfull/publishers/homupe.py` — `homupe/pipelines/publish.py` を移植
- [x] `main.py`（Tweepy 残骸）削除
- [x] `pipelines/dwarkesh.py`（lex.py の重複）削除

---

## Phase 1 — 共有基盤の整備 ✅

- [x] `factfull/core/types.py` — `SourceDoc` / `ProcessedDoc` / `Triple` / `Entity`
- [x] `factfull/llm.py` — Ollama / Anthropic バックエンド
- [x] `factfull/ingest/chunker.py` — テキスト分割を一本化
- [x] `PipelineResult.to_processed_doc()` — podcast パイプラインのアダプター

---

## Phase 2 — Ingestion 拡張 ✅

- [x] `factfull/ingest/paper.py` — arXiv / PDF → SourceDoc
- [x] `factfull/ingest/book.py` — Gutenberg/ローカル → SourceDoc
- [x] `factfull/ingest/web.py` — URL → SourceDoc
- [x] `factfull/ingest/pluralistic.py` — Cory Doctorow RSS → SourceDoc
- [x] `factfull/extract/entity.py` — 汎用エンティティ抽出
- [x] `factfull/extract/relation.py` — 汎用関係抽出
- [x] `factfull/extract/podcast_extract.py` — サマリーベース高品質抽出（speaker 帰属付き）

---

## Phase 3 — Knowledge Store 統合 ✅

- [x] `factfull/graph/neo4j.py` — Neo4j クライアント
- [x] `factfull/registry.py` — バッチ処理レジストリ（SQLite）
- [x] `scripts/batch_process.py` — 多ソース一括処理（podcast / paper / web）
- [x] `factfull/podcast/archiver.py` — YouTube → 翻訳/要約/ファクトチェック
- [x] `pipelines/lex.py` — Podcast パイプライン CLI
- [ ] pipeline 完了時に summary を localsearch-mcp が監視するディレクトリへ自動配置

---

## Phase 4 — Synthesis（複数ソース横断）✅（基本実装完了）

目標: 同テーマの複数ソースから独自記事を生成する

- [x] `factfull/synthesis/cross_source.py` — 話者別 claim + 共通エンティティ → 論文スタイル記事生成
- [x] `scripts/e2e_synthesis.py` — Synthesis CLI
- [ ] `factfull/synthesis/tracker.py` — 予測・発言の時系列追跡（Plan B、未着手）
- [ ] speaker 帰属クエリの品質向上（`[Speaker Name]` prefix が全エピソードに揃うよう抽出改善中）

---

## Phase 5 — 既存リポジトリの廃止・統合完了（未着手）

- [ ] `kg-builder` — KG 管理 API のみ残置、L1/L2 は factfull に統合済み
- [ ] `infoseeker` — `factfull/ingest/paper.py` に統合、リポジトリ廃止
- [ ] `docchat` — RAG ロジックを `factfull/synthesis/` に統合、Gradio UI 廃止
- [ ] `graphragmi` — 判断保留（Microsoft フレームワーク依存度次第）

---

## 現在の運用フロー

```
YouTube URL
  → pipelines/lex.py          （翻訳・要約・ファクトチェック → summary_ja.md）
  → batch_process.py --graph-only
  → extract/podcast_extract.py（サマリーから speaker 帰属付きエンティティ抽出）
  → graph/neo4j.py            （Neo4j 書き込み）
  → synthesis/cross_source.py （複数エピソード横断 → 論文スタイル記事）
  → publishers/homupe.py      （ブログ投稿）
```

---

## 参照

- アーキテクチャ詳細: [system-architecture.md](system-architecture.md)
- 旧アーキテクチャ: [architecture.md](architecture.md)
