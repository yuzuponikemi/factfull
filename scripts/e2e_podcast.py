#!/usr/bin/env python3
"""
E2E テスト: Podcast → SourceDoc → ProcessedDoc → Neo4j

既存エピソードを regen=True で再利用し、write_graph=True で Neo4j に書き込む。

使い方:
    uv run scripts/e2e_podcast.py
    uv run scripts/e2e_podcast.py --url https://www.youtube.com/watch?v=XXXX
    uv run scripts/e2e_podcast.py --no-graph
"""
import argparse
import os

DEFAULT_URL = "https://www.youtube.com/watch?v=Q8Fkpi18QXU"

MODEL = "gemma4:e4b"


def main() -> None:
    parser = argparse.ArgumentParser(description="E2E: Podcast → ProcessedDoc → Neo4j")
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--no-graph", action="store_true", help="Neo4j 書き込みをスキップ")
    args = parser.parse_args()

    os.environ["FACTFULL_OLLAMA_MODEL"] = MODEL
    os.environ["FACTFULL_LLM_BACKEND"] = "ollama"
    os.environ.setdefault("NEO4J_URI", "bolt://localhost:7687")
    os.environ.setdefault("NEO4J_USER", "neo4j")
    os.environ.setdefault("NEO4J_PASSWORD", "factfull123")

    print("=" * 60)
    print("E2E テスト: Podcast → ProcessedDoc → Neo4j")
    print("=" * 60)
    print(f"\n対象 URL: {args.url}")

    from factfull.podcast.pipeline import PipelineConfig, run_pipeline

    config = PipelineConfig(
        translate_model=MODEL,
        analyze_model=MODEL,
        factcheck_model=MODEL,
        write_graph=not args.no_graph,
        critique=False,   # E2E確認なので高速化のため省略
        editorial=False,
    )

    print("\n[pipeline] regen=True で既存データを再利用しながら実行...")
    result = run_pipeline(config, args.url, regen=True)

    print("\n" + "=" * 60)
    print("✅ Pipeline 完了")
    print("=" * 60)
    print(f"  タイトル  : {result.title}")
    print(f"  チャンネル: {result.channel}")
    print(f"  スコア    : {result.score:.0f}/100")
    print(f"  出力先    : {result.summary_path}")

    if args.no_graph:
        print("\n[Step 3] Neo4j 書き込みをスキップ (--no-graph)")
    else:
        print("\n[Step 3] Neo4j 書き込み済み (write_graph=True)")


if __name__ == "__main__":
    main()
