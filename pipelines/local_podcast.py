#!/usr/bin/env python3
"""
ローカル音声ポッドキャストパイプライン
========================================
使い方:
  # MP3 を処理して記事生成
  python pipelines/local_podcast.py ~/podcasts/off_topic/audio/2026-05-06_ep347_317_最終回.mp3 \\
      --source-id off_topic_ep347 --channel "Off Topic // オフトピック"

  # 記事生成 + homupe ブログ投稿
  python pipelines/local_podcast.py /path/to/episode.mp3 \\
      --source-id off_topic_ep347 --channel "Off Topic // オフトピック" --publish

  # 既存 transcript を再利用（Whisper スキップ）
  python pipelines/local_podcast.py /path/to/episode.mp3 \\
      --source-id off_topic_ep347 --regen

  # Whisper モデルサイズ指定
  python pipelines/local_podcast.py /path/to/episode.mp3 \\
      --source-id off_topic_ep347 --whisper-model large-v3-turbo

環境変数:
  OLLAMA_URL / FACTFULL_OLLAMA_URL  Ollama エンドポイント
  PODCAST_OUTPUT_DIR                エピソード保存先（デフォルト: ~/podcasts）
  HOMUPE_ROOT                       homupe リポジトリルート
  TAVILY_API_KEY                    ゲスト検索（省略可）
"""
import argparse
import os
from pathlib import Path

from factfull.podcast.local_pipeline import LocalPipelineConfig, run_local_pipeline
from factfull.publishers.homupe import (
    BlogMetadata,
    create_local_podcast_post,
    default_blog_dir,
    generate_blog_metadata,
)

config = LocalPipelineConfig(
    source_id="",  # CLI で上書き
    channel="",
    language="ja",
    whisper_model="large-v3",
    analyze_model="gemma4:26b",
    factcheck_model="gemma4:e4b",
    write_graph=True,
    output_base=Path(
        os.environ.get("PODCAST_OUTPUT_DIR", str(Path.home() / "podcasts"))
    ),
)


def main() -> None:
    parser = argparse.ArgumentParser(description="ローカル音声ポッドキャストパイプライン")
    parser.add_argument("mp3_path", help="音声ファイルパス（MP3/M4A/WAV）")
    parser.add_argument("--source-id", required=True, help="エピソード識別子（例: off_topic_ep347）")
    parser.add_argument("--channel", default="", help="チャンネル名")
    parser.add_argument("--language", default="ja", help="音声言語（デフォルト: ja）")
    parser.add_argument("--whisper-model", default="large-v3",
                        choices=["tiny", "base", "small", "medium", "large-v3", "large-v3-turbo"],
                        help="Whisper モデルサイズ")
    parser.add_argument("--regen", action="store_true", help="既存 transcript を再利用")
    parser.add_argument("--publish", action="store_true", help="homupe ブログに記事を投稿")
    parser.add_argument("--no-graph", action="store_true", help="Neo4j KG 書き込みをスキップ")
    parser.add_argument("--model", default="gemma4:26b", help="要約・分析モデル（デフォルト: gemma4:26b）")
    parser.add_argument("--factcheck-model", default="gemma4:e4b", help="ファクトチェックモデル")
    args = parser.parse_args()

    config.source_id = args.source_id
    config.channel = args.channel
    config.language = args.language
    config.whisper_model = args.whisper_model
    config.analyze_model = args.model
    config.factcheck_model = args.factcheck_model
    config.write_graph = not args.no_graph

    mp3_path = Path(args.mp3_path).expanduser()
    if not mp3_path.exists():
        print(f"ERROR: ファイルが見つかりません: {mp3_path}")
        raise SystemExit(1)

    result = run_local_pipeline(config, mp3_path, regen=args.regen)
    print(f"\n✅ パイプライン完了  スコア: {result.score:.0f}/100")
    print(f"   出力: {result.episode_dir}")

    if args.publish:
        print("\n📝 ブログメタデータ生成中...")
        meta = generate_blog_metadata(result, model=config.factcheck_model)
        blog_dir = default_blog_dir()
        post_path = create_local_podcast_post(result, meta, blog_dir)
        print(f"\n✅ 記事作成: {post_path}")


if __name__ == "__main__":
    main()
