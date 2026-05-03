#!/usr/bin/env python3
"""
E2E テスト: Cross-source Synthesis (Phase 4)

Neo4j に蓄積された複数ソースのナレッジグラフから
横断的な統合記事を生成する。

使い方:
    uv run scripts/e2e_synthesis.py
    uv run scripts/e2e_synthesis.py --topic "AIと数学的思考"
    uv run scripts/e2e_synthesis.py --min-sources 2
    uv run scripts/e2e_synthesis.py --out /tmp/synthesis.md
"""
import argparse
import os
from pathlib import Path

MODEL = "gemma4:e4b"
DEFAULT_TOPIC = "AIと数学的思考"
DEFAULT_OUT = Path.home() / "synthesis" / "cross_source.md"


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 4: Cross-source synthesis")
    parser.add_argument("--topic", default=DEFAULT_TOPIC, help="統合記事のテーマ")
    parser.add_argument("--min-sources", type=int, default=2, help="最低ソース数（デフォルト: 2）")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="出力先 Markdown パス")
    args = parser.parse_args()

    os.environ["FACTFULL_OLLAMA_MODEL"] = MODEL
    os.environ["FACTFULL_LLM_BACKEND"] = "ollama"
    os.environ.setdefault("NEO4J_URI", "bolt://localhost:7687")
    os.environ.setdefault("NEO4J_USER", "neo4j")
    os.environ.setdefault("NEO4J_PASSWORD", "factfull123")

    print("=" * 60)
    print("Phase 4: Cross-source Synthesis")
    print("=" * 60)

    from factfull.graph.neo4j import Neo4jClient
    from factfull.synthesis.cross_source import find_shared_entities, synthesize

    with Neo4jClient() as client:
        stats = client.get_statistics()
        print(f"\n📊 グラフ統計: {stats}")

        # ── Step 1: 共通エンティティ確認 ────────────────────────────────────
        print(f"\n🔍 Step 1: 共通エンティティ（≥{args.min_sources}ソース）を探索...")
        shared = find_shared_entities(client, min_sources=args.min_sources)

        if not shared:
            print("  共通エンティティなし → 全エンティティモードで統合します")
        else:
            print(f"  共通エンティティ: {len(shared)} 件")
        for e in shared[:10]:
            srcs = ", ".join(s["title"][:30] for s in e["sources"])
            print(f"    [{e['num_sources']}] {e['name']} ({e['type']}) — {srcs}")
        if len(shared) > 10:
            print(f"    ... 他 {len(shared) - 10} 件")


        # ── Step 2: 統合記事生成 ─────────────────────────────────────────────
        print(f"\n✍️  Step 2: 統合記事を生成中... (topic: {args.topic})")
        article = synthesize(client, model=MODEL, min_sources=args.min_sources, topic=args.topic)

        print(f"\n  生成完了: {len(article)} 文字")

        # ── Step 3: 保存 ─────────────────────────────────────────────────────
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(article, encoding="utf-8")
        print(f"  💾 保存: {args.out}")

    print("\n" + "=" * 60)
    print("✅ Synthesis 完了")
    print("=" * 60)
    print("\n── 記事冒頭 ──")
    print(article[:800])
    print("...")


if __name__ == "__main__":
    main()
