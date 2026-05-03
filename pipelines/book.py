#!/usr/bin/env python3
"""
Book Guide パイプライン
=======================
book-research の run_dir を受け取り、book_guide.md をファクトチェックして
homupe に Book Guide 記事として投稿する。

使い方:
  # ファクトチェックのみ
  python pipelines/book.py /path/to/book-research/data/run_20260430_135649

  # ファクトチェック + homupe ブログ投稿
  python pipelines/book.py /path/to/book-research/data/run_20260430_135649 --publish

  # book-research の data ディレクトリと book_id で指定（最新 run を自動検出）
  python pipelines/book.py --data-dir /path/to/book-research/data --book plato_republic --publish

環境変数:
  HOMUPE_ROOT  homupe リポジトリルート（デフォルト: ~/source/personal/homupe）
"""
import argparse
import sys
from pathlib import Path

from factfull.book.pipeline import BookPipelineConfig, run_book_pipeline

config = BookPipelineConfig(
    factcheck_model="gemma4:e4b",
    threshold=95.0,
    max_iter=5,
    max_claims=50,
    top_k=5,
    critique=False,
    editorial=False,
)

META_MODEL = "gemma4:e4b"


def find_latest_run(data_dir: Path, book_id: str) -> Path:
    """book_id に一致する最新の run ディレクトリを返す。"""
    candidates = []
    for run_dir in sorted(data_dir.glob("run_*"), reverse=True):
        events = run_dir / "events.log"
        if events.exists() and f"book={book_id}" in events.read_text(encoding="utf-8"):
            candidates.append(run_dir)
    if not candidates:
        raise FileNotFoundError(
            f"book_id={book_id} に一致する run ディレクトリが見つかりません: {data_dir}"
        )
    return candidates[0]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Book Guide ファクトチェック → homupe 投稿パイプライン",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "run_dir",
        nargs="?",
        help="book-research の run ディレクトリ（絶対パスまたは相対パス）",
    )
    group.add_argument(
        "--book",
        metavar="BOOK_ID",
        help="book_id（--data-dir と併用して最新 run を自動検出）",
    )
    parser.add_argument(
        "--data-dir",
        default=str(Path.home() / "source" / "personal" / "book-research" / "data"),
        help="book-research data ディレクトリ（--book 使用時に指定）",
    )
    parser.add_argument(
        "--publish",
        action="store_true",
        help="homupe ブログへ記事を投稿する",
    )
    args = parser.parse_args()

    if args.run_dir:
        run_dir = Path(args.run_dir).resolve()
    else:
        data_dir = Path(args.data_dir)
        run_dir = find_latest_run(data_dir, args.book)
        print(f"📂 最新 run を検出: {run_dir.name}")

    result = run_book_pipeline(config, run_dir)
    print(f"\n✅ ファクトチェック完了: {result.book_guide_path}")
    print(f"   スコア: {result.score:.0f}/100  /  {result.author} 『{result.book_title}』")

    if not args.publish:
        return

    from factfull.publishers.homupe import (
        generate_book_metadata, create_book_guide_post, default_blog_dir,
    )
    print("\n🏷️  ブログメタデータ生成中...")
    meta = generate_book_metadata(result, model=META_MODEL)
    print(f"   タイトル: {meta.title_ja}")
    print(f"   スラッグ: {meta.slug}")

    post_path = create_book_guide_post(result, meta, blog_dir=default_blog_dir())
    print(f"\n✅ ブログ記事作成: {post_path}")


if __name__ == "__main__":
    sys.exit(main())
