#!/usr/bin/env python3
"""
Synthesis パイプライン — 複数ソース横断記事生成
================================================
Neo4j に蓄積されたナレッジグラフから話者別 claim を抽出し、
論文スタイルの横断論考を生成して homupe ブログへ投稿する。

使い方:
  # 記事生成のみ（stdout + ファイル保存）
  python pipelines/synthesis.py --topic "AGIと人工知能の未来"

  # homupe ブログへ投稿
  python pipelines/synthesis.py --topic "AGIと人工知能の未来" --publish

  # 出力先を指定
  python pipelines/synthesis.py --topic "AIと数学的思考" --out /tmp/article.md

環境変数:
  OLLAMA_URL / FACTFULL_OLLAMA_URL  Ollama エンドポイント
  NEO4J_URI                         Neo4j 接続先（デフォルト: bolt://localhost:7687）
  NEO4J_PASSWORD                    Neo4j パスワード
  HOMUPE_ROOT                       homupe リポジトリルート（--publish 時）
"""
import argparse
import os
from datetime import date
from pathlib import Path

MODEL = "gemma4:26b"
DEFAULT_TOPIC = "AIと人工知能の未来"
DEFAULT_OUT = Path.home() / "synthesis" / "cross_source.md"


def _setup_env() -> None:
    os.environ.setdefault("FACTFULL_OLLAMA_MODEL", MODEL)
    os.environ.setdefault("FACTFULL_LLM_BACKEND", "ollama")
    os.environ.setdefault("NEO4J_URI", "bolt://localhost:7687")
    os.environ.setdefault("NEO4J_USER", "neo4j")
    os.environ.setdefault("NEO4J_PASSWORD", "factfull123")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="複数ソース横断 synthesis 記事を生成して homupe へ投稿",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--topic", default=DEFAULT_TOPIC, help="論考テーマ（日本語）")
    parser.add_argument("--min-sources", type=int, default=2, help="最低ソース数（デフォルト: 2）")
    parser.add_argument("--model", default=MODEL, help=f"LLM モデル（デフォルト: {MODEL}）")
    parser.add_argument("--out", type=Path, default=None, help="Markdown 出力先（省略時は ~/synthesis/cross_source.md）")
    parser.add_argument("--publish", action="store_true", help="homupe ブログへ記事を投稿する")
    args = parser.parse_args()

    _setup_env()

    from factfull.graph.neo4j import Neo4jClient
    from factfull.synthesis.cross_source import synthesize

    print(f"[synthesis] topic='{args.topic}'  model={args.model}  min_sources={args.min_sources}")

    with Neo4jClient() as client:
        stats = client.get_statistics()
        print(f"[neo4j] sources={stats['sources']}  entities={stats['entities']}  mentions={stats['mentions']}")

        article = synthesize(
            client,
            model=args.model,
            min_sources=args.min_sources,
            topic=args.topic,
        )

    print(f"[synthesis] 生成完了: {len(article)} 文字")

    # ── 保存 ─────────────────────────────────────────────────────────────────
    out_path = args.out or DEFAULT_OUT
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(article, encoding="utf-8")
    print(f"[synthesis] 保存: {out_path}")

    # ── homupe 投稿 ───────────────────────────────────────────────────────────
    if args.publish:
        from factfull.publishers.homupe import default_blog_dir

        today = date.today()
        slug = args.topic.replace(" ", "-").replace("　", "-")[:40]
        filename = f"{today.isoformat()}-synthesis-{slug}.md"
        blog_dir = default_blog_dir()
        post_path = blog_dir / filename

        # frontmatter を付けて書き出す
        tags = "AI, 合成記事"
        frontmatter = f"""---
date: {today.isoformat()}
categories:
  - Podcast
tags:
  - {tags.replace(", ", "\n  - ")}
---

# {args.topic}

複数のポッドキャストエピソードの知識グラフから自動合成した横断論考。

<!-- more -->

"""
        post_path.write_text(frontmatter + article, encoding="utf-8")
        print(f"[publish] 投稿: {post_path}")

        # git add & commit
        import subprocess
        subprocess.run(["git", "-C", str(blog_dir.parent.parent), "add", str(post_path)], check=True)
        subprocess.run(
            ["git", "-C", str(blog_dir.parent.parent), "commit", "-m",
             f"feat: synthesis article — {args.topic}"],
            check=True,
        )
        print("[publish] git commit 完了")

    print("\n── 記事冒頭 ──")
    print(article[:600])
    print("...")


if __name__ == "__main__":
    main()
