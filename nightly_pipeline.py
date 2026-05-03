"""
nightly_pipeline.py
====================
深夜バッチ: 新着 YouTube エピソードを検出→処理→KG更新→インサイト生成。

フロー:
  Phase 1: yt-dlp で各チャンネルの新着エピソードを検出
  Phase 2: 新着エピソードを直列処理（translate→KG書き込み→homupe投稿）
  Phase 3: 新着があった日のみ KG インサイトを再生成して homupe 投稿

使い方:
    uv run python nightly_pipeline.py
    uv run python nightly_pipeline.py --dry-run       # 検出のみ（処理しない）
    uv run python nightly_pipeline.py --force-phase3  # 新着なしでも Phase 3 を実行

crontab:
    0 2 * * * cd /Users/ikmx/source/personal/factfull && uv run python nightly_pipeline.py >> /tmp/factfull_nightly.log 2>&1
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any


# ── チャンネル設定 ──────────────────────────────────────────────────────────────

@dataclass
class ChannelConfig:
    name: str
    playlist_url: str            # YouTube playlist / channel URL
    speakers: list[str] = field(default_factory=list)   # KG フィルタ用（任意）
    max_new: int = 3             # 1回の実行で処理する上限


CHANNELS: list[ChannelConfig] = [
    ChannelConfig(
        name="Lex Fridman Podcast",
        playlist_url="https://www.youtube.com/@lexfridman/podcasts",
        max_new=2,
    ),
    ChannelConfig(
        name="Dwarkesh Podcast",
        playlist_url="https://www.youtube.com/@DwarkeshPatel/videos",
        max_new=2,
    ),
    ChannelConfig(
        name="Y Combinator",
        playlist_url="https://www.youtube.com/@ycombinator/videos",
        max_new=2,
    ),
]


# ── Phase 1: 新着検出 ─────────────────────────────────────────────────────────

def _fetch_playlist_ids(playlist_url: str, max_entries: int = 30) -> list[dict[str, str]]:
    """yt-dlp --flat-playlist で最新エントリのメタデータを返す。"""
    cmd = [
        "yt-dlp",
        "--flat-playlist",
        "--playlist-end", str(max_entries),
        "--print", "%(id)s\t%(title)s\t%(upload_date)s",
        "--no-warnings",
        "--quiet",
        playlist_url,
    ]
    try:
        out = subprocess.check_output(cmd, text=True, timeout=60)
    except subprocess.TimeoutExpired:
        print(f"  [WARN] yt-dlp timeout: {playlist_url}", flush=True)
        return []
    except subprocess.CalledProcessError as e:
        print(f"  [WARN] yt-dlp error ({e.returncode}): {playlist_url}", flush=True)
        return []

    entries = []
    for line in out.strip().splitlines():
        parts = line.split("\t", 2)
        if len(parts) >= 2:
            entries.append({
                "video_id": parts[0].strip(),
                "title": parts[1].strip() if len(parts) > 1 else "",
                "upload_date": parts[2].strip() if len(parts) > 2 else "",
            })
    return entries


def _get_processed_ids(client) -> set[str]:
    """Neo4j に登録済みの source_id を返す。"""
    rows = client.run_cypher(
        "MATCH (s:Source {source_type: 'podcast'}) RETURN s.source_id AS source_id"
    )
    return {r["source_id"] for r in rows}


def detect_new_episodes(
    channels: list[ChannelConfig],
    client,
) -> list[tuple[ChannelConfig, dict]]:
    """各チャンネルの新着エピソード（未処理）をリストで返す。"""
    processed = _get_processed_ids(client)
    print(f"[Phase 1] 既処理エピソード: {len(processed)} 件", flush=True)

    new_episodes: list[tuple[ChannelConfig, dict]] = []
    for ch in channels:
        print(f"  チャンネル確認中: {ch.name}", flush=True)
        entries = _fetch_playlist_ids(ch.playlist_url)
        count = 0
        for entry in entries:
            vid = entry["video_id"]
            if vid not in processed:
                new_episodes.append((ch, entry))
                count += 1
                if count >= ch.max_new:
                    break
        print(f"  → 新着 {count} 件", flush=True)

    return new_episodes


# ── Phase 2: エピソード処理 ───────────────────────────────────────────────────

def _process_episode(
    ch: ChannelConfig,
    entry: dict,
    config,
    blog_dir: Path,
    dry_run: bool = False,
) -> bool:
    """1エピソードを処理して homupe に投稿。成功したら True を返す。"""
    from factfull.podcast.pipeline import run_pipeline
    from factfull.publishers.homupe import generate_blog_metadata, create_blog_post

    vid = entry["video_id"]
    url = f"https://www.youtube.com/watch?v={vid}"
    title = entry.get("title", vid)

    print(f"\n  [Episode] {title}", flush=True)
    print(f"  URL: {url}", flush=True)

    if dry_run:
        print("  [DRY-RUN] スキップ", flush=True)
        return False

    try:
        result = run_pipeline(config, url)
        print(f"  → score={result.score:.1f}  KG書き込み完了", flush=True)

        meta = generate_blog_metadata(result, model=config.analyze_model)
        post_path = create_blog_post(result, meta, blog_dir=blog_dir)
        print(f"  → 投稿: {post_path}", flush=True)
        return True

    except Exception as e:
        print(f"  [ERROR] {type(e).__name__}: {e}", flush=True)
        return False


def process_episodes(
    new_episodes: list[tuple[ChannelConfig, dict]],
    blog_dir: Path,
    dry_run: bool = False,
) -> int:
    """新着エピソードを直列で処理。成功件数を返す。"""
    from factfull.podcast.pipeline import PipelineConfig

    config = PipelineConfig(
        write_graph=True,
        analyze_model="gemma4:26b",
        translate_model="translategemma:12b",
        factcheck_model="gemma4:e4b",
    )

    ok = 0
    for ch, entry in new_episodes:
        success = _process_episode(ch, entry, config, blog_dir, dry_run=dry_run)
        if success:
            ok += 1
    return ok


# ── Phase 3: KG インサイト生成 ────────────────────────────────────────────────

def run_kg_insights(
    client,
    blog_dir: Path,
    model: str = "gemma4:26b",
    dry_run: bool = False,
) -> None:
    """KG インサイトを生成して homupe に投稿する。"""
    from factfull.synthesis.kg_analysis import analyze_and_generate

    print("\n[Phase 3] KG インサイト生成中...", flush=True)

    if dry_run:
        print("  [DRY-RUN] スキップ", flush=True)
        return

    insight_text = analyze_and_generate(client, model=model, min_bridge_speakers=2)
    print(f"  → 生成完了: {len(insight_text)} 文字", flush=True)

    today = date.today()
    slug = f"{today.isoformat()}-kg-insights"
    filename = f"{slug}.md"
    post_path = blog_dir / filename

    frontmatter = f"""---
date: {today.isoformat()}
categories:
  - Synthesis
tags:
  - AI
  - ナレッジグラフ
  - 合成記事
  - KG解析
---

# KG インサイト — {today.isoformat()}

<!-- more -->

"""
    post_path.write_text(frontmatter + insight_text, encoding="utf-8")
    print(f"  → 保存: {post_path}", flush=True)


# ── メイン ────────────────────────────────────────────────────────────────────

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="factfull 夜間パイプライン")
    parser.add_argument("--dry-run", action="store_true",
                        help="新着検出のみ（処理・投稿しない）")
    parser.add_argument("--force-phase3", action="store_true",
                        help="新着なしでも Phase 3 を実行")
    parser.add_argument("--model", default="gemma4:26b",
                        help="Phase 3 で使用するモデル")
    parser.add_argument("--channels", default=None,
                        help="処理するチャンネル名（カンマ区切り、デフォルト:全チャンネル）")
    args = parser.parse_args()

    # 環境変数デフォルト
    os.environ.setdefault("NEO4J_URI", "bolt://localhost:7687")
    os.environ.setdefault("NEO4J_USER", "neo4j")
    os.environ.setdefault("NEO4J_PASSWORD", "factfull123")
    os.environ.setdefault("FACTFULL_LLM_BACKEND", "ollama")

    started_at = datetime.now()
    print(f"\n{'='*60}", flush=True)
    print(f"[nightly_pipeline] 開始: {started_at:%Y-%m-%d %H:%M:%S}", flush=True)
    print(f"{'='*60}", flush=True)

    channels = CHANNELS
    if args.channels:
        names = {n.strip() for n in args.channels.split(",")}
        channels = [c for c in CHANNELS if c.name in names]
        print(f"チャンネル絞り込み: {[c.name for c in channels]}", flush=True)

    from factfull.graph.neo4j import Neo4jClient
    from factfull.publishers.homupe import default_blog_dir

    blog_dir = default_blog_dir()
    blog_dir.mkdir(parents=True, exist_ok=True)

    with Neo4jClient() as client:
        # Phase 1
        print("\n[Phase 1] 新着エピソード検出", flush=True)
        new_episodes = detect_new_episodes(channels, client)
        print(f"\n  合計新着: {len(new_episodes)} 件", flush=True)

        if not new_episodes:
            print("  新着なし。", flush=True)
            if args.force_phase3:
                run_kg_insights(client, blog_dir, model=args.model, dry_run=args.dry_run)
            print("\n[nightly_pipeline] 完了（新着なし）", flush=True)
            return

        # Phase 2
        print(f"\n[Phase 2] エピソード処理（直列、{len(new_episodes)} 件）", flush=True)
        ok = process_episodes(new_episodes, blog_dir, dry_run=args.dry_run)
        print(f"\n[Phase 2] 完了: 成功 {ok}/{len(new_episodes)} 件", flush=True)

        # Phase 3: 新着があった日のみ
        if ok > 0 or args.force_phase3:
            run_kg_insights(client, blog_dir, model=args.model, dry_run=args.dry_run)

    elapsed = (datetime.now() - started_at).total_seconds()
    print(f"\n[nightly_pipeline] 完了: {elapsed:.0f}秒", flush=True)
    print(f"{'='*60}\n", flush=True)


if __name__ == "__main__":
    main()
