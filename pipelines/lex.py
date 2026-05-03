#!/usr/bin/env python3
"""
Podcast パイプライン（Lex Fridman / Dwarkesh Patel / その他）
=============================================================
使い方:
  # 記事生成のみ
  python pipelines/lex.py https://www.youtube.com/watch?v=VIDEO_ID

  # 記事生成 + homupe ブログ投稿
  python pipelines/lex.py https://www.youtube.com/watch?v=VIDEO_ID --publish

  # X (Twitter) にも投稿
  python pipelines/lex.py https://www.youtube.com/watch?v=VIDEO_ID --publish --tweet

  # Pass 2 から再実行（section_summaries.json 再利用）
  python pipelines/lex.py https://www.youtube.com/watch?v=VIDEO_ID --regen

環境変数:
  OLLAMA_URL / FACTFULL_OLLAMA_URL  Ollama エンドポイント
  PODCAST_OUTPUT_DIR                エピソード保存先（デフォルト: ~/podcasts）
  HOMUPE_ROOT                       homupe リポジトリルート
  TAVILY_API_KEY                    ゲスト検索（省略可）
  FIREFOX_PROFILE_PATH              X 投稿用 Firefox プロファイル
"""
import argparse
from pathlib import Path

from factfull.podcast.pipeline import PipelineConfig, run_pipeline

config = PipelineConfig(
    translate_model="translategemma:12b",
    analyze_model="gemma4:26b",
    extract_model="gemma4:26b",
    factcheck_model="gemma4:e4b",
    editorial_model=None,
    translate_chunk_size=6000,
    summary_chunk_size=5000,
    threshold=95.0,
    max_iter=5,
    max_claims=50,
    top_k=5,
    critique=True,
    editorial=True,
    fetch_comments=False,
    write_graph=True,
    output_base=Path.home() / "podcasts",
    blog_name="SoryuNews",
    reader_persona="英語圏情報にアクセスしたい日本語話者のエンジニア・研究者",
    n_questions=4,
)

META_MODEL = "gemma4:e4b"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Podcast → 日本語翻訳記事 パイプライン",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("url", help="YouTube URL")
    parser.add_argument("--regen", action="store_true",
                        help="section_summaries.json を再利用して Pass 2 以降のみ実行")
    parser.add_argument("--publish", action="store_true",
                        help="homupe ブログへ記事を投稿する")
    parser.add_argument("--tweet", action="store_true",
                        help="X (Twitter) にも投稿（--publish と併用）")
    args = parser.parse_args()

    result = run_pipeline(config, args.url, regen=args.regen)
    print(f"\n✅ 記事生成完了: {result.summary_path}")
    print(f"   スコア: {result.score:.0f}/100  /  {result.title}")

    if not args.publish:
        return

    from factfull.publishers.homupe import (
        generate_blog_metadata, create_blog_post, post_tweet, default_blog_dir,
    )
    print("\n🏷️  ブログメタデータ生成中...")
    meta = generate_blog_metadata(result, model=META_MODEL)
    print(f"   タイトル: {meta.title_ja}")
    print(f"   スラッグ: {meta.slug}")

    post_path = create_blog_post(result, meta, blog_dir=default_blog_dir())
    print(f"\n✅ ブログ記事作成: {post_path}")

    if args.tweet:
        post_tweet(result, meta)


if __name__ == "__main__":
    main()
