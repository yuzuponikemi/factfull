#!/usr/bin/env python3
"""
E2E テスト: arXiv 論文 → SourceDoc → ProcessedDoc → Neo4j

使い方:
    uv run scripts/e2e_paper.py
    uv run scripts/e2e_paper.py --arxiv 2403.11996
    uv run scripts/e2e_paper.py --no-graph   # Neo4j 書き込みをスキップ
    uv run scripts/e2e_paper.py --no-summary # 要約生成をスキップ
"""
import argparse
import os
from pathlib import Path

# デフォルト: Attention Is All You Need (transformer の原論文)
DEFAULT_ARXIV = "1706.03762"

MODEL = "gemma4:e4b"


def main() -> None:
    parser = argparse.ArgumentParser(description="E2E: arXiv → ProcessedDoc → Neo4j")
    parser.add_argument("--arxiv", default=DEFAULT_ARXIV, help="arXiv ID または URL")
    parser.add_argument("--no-graph", action="store_true", help="Neo4j 書き込みをスキップ")
    parser.add_argument("--no-summary", action="store_true", help="要約生成をスキップ")
    args = parser.parse_args()

    os.environ["FACTFULL_OLLAMA_MODEL"] = MODEL
    os.environ["FACTFULL_LLM_BACKEND"] = "ollama"
    os.environ.setdefault("NEO4J_URI", "bolt://localhost:7687")
    os.environ.setdefault("NEO4J_USER", "neo4j")
    os.environ.setdefault("NEO4J_PASSWORD", "factfull123")

    print("=" * 60)
    print("E2E テスト: arXiv → SourceDoc → ProcessedDoc → Neo4j")
    print("=" * 60)

    # ── Step 1: Ingestion ────────────────────────────────────────────────────
    print(f"\n[Step 1] arXiv 取得: {args.arxiv}")
    from factfull.ingest.paper import ingest_arxiv
    doc = ingest_arxiv(args.arxiv, output_dir=Path.home() / "papers" / "arxiv")
    print(f"  タイトル : {doc.title}")
    print(f"  テキスト : {len(doc.text)} 字")
    print(f"  チャンク : {len(doc.chunks)} 件")
    if doc.metadata.get("authors"):
        print(f"  著者     : {', '.join(doc.metadata['authors'][:3])}")
    if doc.metadata.get("published"):
        print(f"  発表日   : {doc.metadata['published'][:10]}")

    # ── Step 2: Processing ───────────────────────────────────────────────────
    print(f"\n[Step 2] process() 実行 (summarize={not args.no_summary}, extract=True)")
    from factfull.process import process
    pdoc = process(
        doc,
        model=MODEL,
        summarize=not args.no_summary,
        extract=True,
        max_chunks_for_summary=5,
        max_chunks_for_extract=5,
    )

    print(f"\n  要約: {len(pdoc.summary)} 字")
    if pdoc.summary:
        print("  ── 要約冒頭 ──")
        print(pdoc.summary[:400])
        print("  ...")

    print(f"\n  エンティティ: {len(pdoc.entities)} 件")
    for e in pdoc.entities[:10]:
        print(f"    {e.name} [{e.type}] conf={e.confidence:.2f}")

    print(f"\n  トリプル: {len(pdoc.triples)} 件")
    for t in pdoc.triples[:10]:
        print(f"    ({t.subject}) --[{t.predicate}]--> ({t.object})")

    # ── Step 3: Neo4j 書き込み ────────────────────────────────────────────────
    if args.no_graph:
        print("\n[Step 3] Neo4j 書き込みをスキップ (--no-graph)")
    else:
        print("\n[Step 3] Neo4j 書き込み")
        try:
            from factfull.graph.neo4j import Neo4jClient
            with Neo4jClient() as g:
                g.setup_schema()
                g.write_processed_doc(pdoc)
                stats = g.get_statistics()
                print(f"  グラフ統計: {stats}")
        except Exception as e:
            print(f"  ⚠️  Neo4j 書き込み失敗（スキップ）: {e}")

    print("\n" + "=" * 60)
    print("✅ E2E 完了")
    print("=" * 60)


if __name__ == "__main__":
    main()
