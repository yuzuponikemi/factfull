"""
factfull/podcast/steps/graph.py
---------------------------------
Step 7: ナレッジグラフ書き込み + エンティティ正規化

summary_ja.md から Grounded Extraction でエンティティ・トリプルを抽出し
Neo4j に書き込む。抽出後に Wikidata でエンティティを正規化する。
"""
from __future__ import annotations


def write_to_graph(result, config) -> None:
    """
    summary_ja.md → podcast_extract → Neo4j → Wikidata 正規化
    の一連のステップを実行する。
    """
    from factfull.graph.neo4j import Neo4jClient
    from factfull.core.types import SourceDoc, ProcessedDoc
    from factfull.extract.podcast_extract import (
        extract_from_summary,
        _extract_speakers_from_summary,
    )
    from factfull.normalize.wiki_linker import WikiLinker
    from factfull.normalize.entity_normalizer import normalize_entities

    _header("ナレッジグラフ書き込み")

    summary_text = result.summary_path.read_text(encoding="utf-8")
    canonical_speakers = _extract_speakers_from_summary(summary_text)
    print(f"  正規スピーカー: {canonical_speakers}", flush=True)

    extract_model = getattr(config, "extract_model", config.analyze_model)

    # エンティティ・トリプル抽出
    print("  エンティティ抽出中...", flush=True)
    entities, triples = extract_from_summary(
        summary_text,
        source_id=result.video_id,
        model=extract_model,
        canonical_speakers=canonical_speakers or None,
    )

    source = SourceDoc(
        source_type="podcast",
        source_id=result.video_id,
        title=result.title,
        text="",
        chunks=[],
        metadata={"channel": result.channel, **result.metadata},
    )
    pdoc = ProcessedDoc(source=source, entities=entities, triples=triples)

    # Neo4j 書き込み
    try:
        with Neo4jClient() as g:
            g.setup_schema()
            g.upsert_source(source)
            g.write_processed_doc(pdoc, clear_old=True)
            stats = g.get_statistics()
            print(f"  グラフ統計: {stats}", flush=True)
    except Exception as e:
        print(f"  [warn] Neo4j 書き込み失敗（スキップ）: {e}", flush=True)
        return

    # Wikidata 正規化（person / organization）
    _header("エンティティ正規化 (Wikidata)")
    try:
        with Neo4jClient() as client, WikiLinker() as linker:
            stats = normalize_entities(
                client, linker,
                types=("person", "organization"),
                limit=200,
            )
        print(f"  正規化結果: {stats}", flush=True)
    except Exception as e:
        print(f"  [warn] 正規化失敗（スキップ）: {e}", flush=True)


def _header(text: str) -> None:
    line = "=" * 60
    print(f"\n{line}\n  {text}\n{line}", flush=True)
