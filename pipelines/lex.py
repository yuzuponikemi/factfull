#!/usr/bin/env python3
"""
Lex Fridman Podcast パイプライン
=================================
使い方:
  # フルパイプライン（初回）
  python pipelines/lex.py https://www.youtube.com/watch?v=VIDEO_ID

  # 既存 section_summaries.json から Pass 2 以降を再実行
  python pipelines/lex.py https://www.youtube.com/watch?v=VIDEO_ID --regen

環境変数:
  OLLAMA_URL / FACTFULL_OLLAMA_URL  Ollama エンドポイント（デフォルト: http://localhost:11435/api/generate）
  PODCAST_OUTPUT_DIR                エピソード保存先（デフォルト: ~/podcasts）
"""
import argparse
import sys
from pathlib import Path

from factfull.podcast.pipeline import PipelineConfig, run_pipeline

config = PipelineConfig(
    # モデル
    translate_model="translategemma:12b",
    analyze_model="gemma4:26b",
    factcheck_model="gemma4:e4b",
    editorial_model=None,           # None → factcheck_model を使用

    # チャンクサイズ
    translate_chunk_size=6000,
    summary_chunk_size=5000,        # Lex は長尺（2〜4h）が多いので大きめ

    # ファクトチェックループ
    threshold=95.0,
    max_iter=5,
    max_claims=50,
    top_k=5,

    # 機能
    editorial=True,
    fetch_comments=False,

    # 出力先
    output_base=Path.home() / "podcasts",

    # コンテンツ
    blog_name="SoryuNews",
    reader_persona="英語圏情報にアクセスしたい日本語話者のエンジニア・研究者",
    n_questions=4,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Lex Fridman Podcast → 日本語翻訳記事 パイプライン",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("url", help="YouTube URL")
    parser.add_argument(
        "--regen",
        action="store_true",
        help="既存の section_summaries.json を再利用して Pass 2 以降のみ実行",
    )
    args = parser.parse_args()

    result = run_pipeline(config, args.url, regen=args.regen)
    print(f"\n✅ 完成: {result.summary_path}")
    print(f"   スコア: {result.score:.0f}/100  /  {result.title}")


if __name__ == "__main__":
    main()
