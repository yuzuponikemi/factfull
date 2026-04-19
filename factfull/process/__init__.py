"""
factfull/process/
==================
SourceDoc → ProcessedDoc の変換パイプライン。

L1（Ingestion）が返した SourceDoc を受け取り、
要約・エンティティ抽出・関係抽出を行って ProcessedDoc を返す。

使い方:
    from factfull.ingest.paper import ingest_arxiv
    from factfull.process import process

    doc = ingest_arxiv("2403.11996")
    pdoc = process(doc)
    # pdoc.summary   → 日本語要約記事
    # pdoc.entities  → 抽出エンティティ
    # pdoc.triples   → 抽出トリプル

    # Neo4j へ書き込む場合
    from factfull.graph.neo4j import Neo4jClient
    with Neo4jClient() as g:
        g.write_processed_doc(pdoc)
"""
from __future__ import annotations

from factfull.core.types import ProcessedDoc, SourceDoc


def process(
    doc: SourceDoc,
    model: str | None = None,
    summarize: bool = True,
    extract: bool = True,
    max_chunks_for_summary: int = 20,
    max_chunks_for_extract: int = 10,
) -> ProcessedDoc:
    """SourceDoc を ProcessedDoc に変換する。

    Args:
        doc: ingest_* が返した SourceDoc
        model: Ollama モデル名（省略時は FACTFULL_OLLAMA_MODEL 環境変数）
        summarize: True のとき日本語要約を生成する
        extract: True のときエンティティ・トリプルを抽出する
        max_chunks_for_summary: 要約に使うチャンク上限
        max_chunks_for_extract: 抽出に使うチャンク上限（重い処理なので上限を絞る）

    Returns:
        ProcessedDoc（summary / entities / triples が設定済み）
    """
    summary = ""
    if summarize:
        from factfull.process.summarizer import summarize as _summarize
        summary = _summarize(doc, model=model, max_chunks=max_chunks_for_summary)

    entities = []
    triples = []
    if extract:
        from factfull.extract.entity import extract_entities
        from factfull.extract.relation import extract_relations

        chunks = (doc.chunks or [doc.text])[:max_chunks_for_extract]
        entities = extract_entities(chunks, source_id=doc.source_id, model=model)
        if entities:
            triples = extract_relations(chunks, entities, source_id=doc.source_id, model=model)

    return ProcessedDoc(
        source=doc,
        summary=summary,
        entities=entities,
        triples=triples,
    )
