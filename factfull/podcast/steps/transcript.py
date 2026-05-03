"""
factfull/podcast/steps/transcript.py
--------------------------------------
Step 1-4: メタデータ取得 / トランスクリプト取得 / 翻訳 / サマリー生成
"""
from __future__ import annotations

import json
import os
from pathlib import Path


def fetch_episode(config, youtube_url: str, regen: bool = False):
    """
    Steps 1-4 を実行して PipelineResult を返す。
    regen=True の場合、既存ディレクトリを再利用して Pass 1 をスキップできる。
    """
    from factfull.podcast.archiver import PodcastArchiver
    from factfull.podcast.pipeline import PipelineResult

    os.environ["FACTFULL_LLM_BACKEND"] = "ollama"
    os.environ["FACTFULL_OLLAMA_MODEL"] = config.factcheck_model

    arch = PodcastArchiver(youtube_url=youtube_url, config=config)

    skip_pass1 = False
    if regen:
        existing = _find_existing_dir(config.output_base, arch.video_id)
        if existing:
            print(f"  [regen] 既存ディレクトリを使用: {existing.name}")
            arch.out_dir = existing
            _load_existing_data(arch, existing)
            skip_pass1 = (existing / "section_summaries.json").exists()
            if not skip_pass1:
                print("  [regen] section_summaries.json なし → Pass 1 から実行")
        else:
            print("  [regen] 既存ディレクトリが見つかりません → フルで実行")
            regen = False

    if not regen or not _has_transcript(arch):
        arch.fetch_metadata()
        arch.fetch_transcript()
        arch.translate_to_japanese()

    arch.generate_summary(skip_pass1=skip_pass1)

    if config.fetch_comments:
        arch.fetch_comments()
        arch.summarize_comments()

    return PipelineResult(
        video_id=arch.video_id,
        title=arch.metadata.get("title", ""),
        channel=arch.metadata.get("channel", ""),
        summary_path=arch.out_dir / "summary_ja.md",
        episode_dir=arch.out_dir,
        score=0.0,
        metadata=arch.metadata,
    )


def _find_existing_dir(output_base: Path, video_id: str) -> Path | None:
    candidates = sorted(output_base.glob(f"{video_id}_*"), reverse=True)
    for d in candidates:
        if (d / "transcript_en.txt").exists():
            return d
    return None


def _has_transcript(arch) -> bool:
    return bool(arch.transcript_en) or (arch.out_dir / "transcript_en.txt").exists()


def _load_existing_data(arch, ep_dir: Path) -> None:
    meta_path = ep_dir / "metadata.json"
    if meta_path.exists():
        arch.metadata = json.loads(meta_path.read_text(encoding="utf-8"))

    for attr, filename in [
        ("transcript_en", "transcript_en.txt"),
        ("transcript_ja", "transcript_ja.txt"),
    ]:
        p = ep_dir / filename
        if p.exists():
            setattr(arch, attr, p.read_text(encoding="utf-8"))

    ts_path = ep_dir / "transcript_en_timestamped.json"
    if ts_path.exists():
        arch.transcript_raw = json.loads(ts_path.read_text(encoding="utf-8"))
