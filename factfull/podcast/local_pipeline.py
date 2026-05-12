"""
factfull/podcast/local_pipeline.py
=====================================
ローカル MP3 ファイルを factfull ポッドキャストパイプラインで処理する。

YouTube 版との違い:
  - Step 1: faster-whisper で文字起こし（YouTube トランスクリプト API の代替）
  - Step 2: 翻訳スキップ（日本語音声なら Whisper が直接 transcript_ja.txt を生成）
  - Step 3 以降: 既存パイプライン（generate_summary → factcheck → KG）をそのまま再利用

使い方:
    from factfull.podcast.local_pipeline import LocalPipelineConfig, run_local_pipeline

    config = LocalPipelineConfig(
        source_id   = "off_topic_ep347",
        channel     = "Off Topic // オフトピック",
        whisper_model = "large-v3",
        write_graph = True,
    )
    result = run_local_pipeline(config, mp3_path, episode_meta)
    # result は PipelineResult と同一構造
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


@dataclass
class LocalPipelineConfig:
    # エピソード識別
    source_id: str              # "off_topic_ep347" — KG / ファイル名に使用
    channel: str = ""           # チャンネル名
    language: str = "ja"        # 音声言語
    speakers: list[str] = field(default_factory=list)  # 既知の出演者名（例: ["草野美希", "宮武徹郎"]）

    # Whisper 設定
    whisper_model: str = "large-v3"
    whisper_device: str = "cpu"
    whisper_compute: str = "int8"

    # 話者分離設定
    diarize: bool = False
    hf_token: str | None = None   # 省略時は HF_TOKEN 環境変数を参照

    # 要約・ファクトチェックモデル
    analyze_model: str = "gemma4:26b"
    factcheck_model: str = "gemma4:e4b"
    translate_chunk_size: int = 6000
    summary_chunk_size: int = 5000
    blog_name: str = "SoryuNews"
    reader_persona: str = "英語圏情報にアクセスしたい日本語話者のエンジニア・研究者"
    n_questions: int = 4

    # ファクトチェック設定
    threshold: float = 95.0
    max_iter: int = 5
    max_claims: int = 50
    top_k: int = 5
    critique: bool = True
    editorial: bool = True

    # KG
    write_graph: bool = True

    # 出力先
    output_base: Path = field(default_factory=lambda: Path.home() / "podcasts")


def run_local_pipeline(
    config: LocalPipelineConfig,
    mp3_path: str | Path,
    episode_meta: dict | None = None,
    regen: bool = False,
) -> "PipelineResult":
    """
    ローカル音声ファイルを処理して PipelineResult を返す。

    Args:
        config:       LocalPipelineConfig
        mp3_path:     MP3 ファイルパス
        episode_meta: タイトル・日付など（rss_downloader の .json から）
                      なければ mp3 のファイル名から推測
        regen:        既存ファイルを再利用してステップをスキップ
                        - transcript_{lang}.txt があれば Whisper をスキップ
                        - summary_ja.md があれば要約生成をスキップ

    Returns:
        PipelineResult（YouTube 版と同一）
    """
    from factfull.podcast.pipeline import PipelineResult
    from factfull.podcast.steps.factcheck import run_factcheck_on_file
    from factfull.podcast.steps.graph import write_to_graph
    from factfull.ingest.audio_transcriber import transcribe

    mp3_path = Path(mp3_path)
    meta = _resolve_meta(mp3_path, episode_meta, config)

    # 出力ディレクトリ
    date_str = datetime.now().strftime("%Y%m%d")
    out_dir = config.output_base / f"{config.source_id}_{date_str}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: metadata.json
    _write_metadata(out_dir, meta, config)

    # Step 2: Whisper 文字起こし（スキップ可）
    transcript_path = out_dir / f"transcript_{config.language}.txt"
    if regen and transcript_path.exists():
        print(f"  [regen] 既存 transcript を使用: {transcript_path.name}")
        transcript_text = transcript_path.read_text(encoding="utf-8")
    else:
        print(f"\n📝 Step 1: Whisper 文字起こし...")
        tr = transcribe(
            mp3_path,
            output_dir=out_dir,
            language=config.language,
            model_size=config.whisper_model,
            device=config.whisper_device,
            compute_type=config.whisper_compute,
            diarize=config.diarize,
            hf_token=config.hf_token,
            speakers=config.speakers or None,
        )
        transcript_text = tr.text
        # transcript_ja.txt として保存済み（transcriber が保存）

    # transcript_en.txt として symlink or copy を作っておく
    # (PodcastArchiver の generate_summary が transcript_en.txt を参照するケースがある)
    en_path = out_dir / "transcript_en.txt"
    if not en_path.exists():
        en_path.write_text(transcript_text, encoding="utf-8")

    # Step 3: 要約生成（regen かつ summary_ja.md が既存ならスキップ）
    summary_path = out_dir / "summary_ja.md"
    if regen and summary_path.exists():
        print(f"  [regen] 既存 summary を使用: {summary_path.name}")
    else:
        print(f"\n📊 Step 2: 要約生成...")
        arch = _build_archiver(config, meta, out_dir, transcript_text)
        arch.generate_summary()

    # Step 4: ファクトチェック
    print(f"\n{'='*60}")
    print(f"  ファクトチェック開始")
    print(f"{'='*60}")

    # PipelineConfig 互換オブジェクトを生成
    podcast_config = _make_podcast_config(config)

    result = PipelineResult(
        video_id=config.source_id,
        title=meta.get("title", config.source_id),
        channel=meta.get("channel", config.channel),
        summary_path=out_dir / "summary_ja.md",
        episode_dir=out_dir,
        score=0.0,
        metadata=meta,
    )

    result.score = run_factcheck(result, podcast_config)

    if config.write_graph:
        write_to_graph(result, podcast_config)

    return result


# ── 内部ヘルパー ──────────────────────────────────────────────────────────────

def _resolve_meta(mp3_path: Path, episode_meta: dict | None, config: LocalPipelineConfig) -> dict:
    """エピソードメタデータを解決する。JSON から読むか、ファイル名から推測。"""
    if episode_meta:
        meta = dict(episode_meta)
    else:
        # 隣の .json があれば読む
        json_path = mp3_path.with_suffix(".json")
        if json_path.exists():
            meta = json.loads(json_path.read_text(encoding="utf-8"))
        else:
            meta = {"title": mp3_path.stem, "pub_date": ""}

    meta.setdefault("channel", config.channel)
    meta.setdefault("source_id", config.source_id)
    meta.setdefault("local_file", str(mp3_path))
    return meta


def _write_metadata(out_dir: Path, meta: dict, config: LocalPipelineConfig) -> None:
    m = {
        "title":      meta.get("title", ""),
        "channel":    meta.get("channel", config.channel),
        "pub_date":   meta.get("pub_date", ""),
        "duration":   meta.get("duration", ""),
        "source_id":  config.source_id,
        "local_file": meta.get("local_file", ""),
    }
    (out_dir / "metadata.json").write_text(
        json.dumps(m, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"📋 Step 0: メタデータ保存")
    print(f"  Title  : {m['title']}")
    print(f"  Channel: {m['channel']}")
    print(f"  Duration: {m['duration']}")


def _build_archiver(config: LocalPipelineConfig, meta: dict, out_dir: Path, transcript_text: str):
    """PodcastArchiver のインスタンスを生成し、文字起こし結果を注入する。"""
    from factfull.podcast.archiver import PodcastArchiver
    from factfull.podcast.pipeline import PipelineConfig

    pc = _make_podcast_config(config)

    # 最低限必要な YouTube URL（_extract_video_id で失敗しないよう dummy を与える）
    dummy_url = f"https://www.youtube.com/watch?v={config.source_id[:11].ljust(11, '_')}"
    arch = PodcastArchiver.__new__(PodcastArchiver)

    # __init__ を呼ばずに必要な属性だけ設定
    arch.youtube_url = dummy_url
    arch.video_id = config.source_id
    arch.translate_model = config.analyze_model   # 翻訳ステップはスキップするので使わない
    arch.analyze_model = config.analyze_model
    arch.factcheck_model = config.factcheck_model
    arch.CHUNK_SIZE = config.translate_chunk_size
    arch.SUMMARY_CHUNK_SIZE = config.summary_chunk_size
    arch.blog_name = config.blog_name
    arch.reader_persona = config.reader_persona
    arch.n_questions = config.n_questions
    arch.model = config.analyze_model
    arch.out_dir = out_dir
    arch.metadata = meta
    # 既知の出演者名を transcript の先頭に注入して LLM の人名認識を補助する
    if config.speakers:
        speakers_note = (
            "【出演者情報】この音声の出演者: "
            + "、".join(config.speakers)
            + "。以下の文字起こしで人名を特定する際はこの情報を最優先で使用すること。\n\n"
        )
        enriched_transcript = speakers_note + transcript_text
    else:
        enriched_transcript = transcript_text

    arch.transcript_raw = []
    arch.transcript_en = enriched_transcript
    arch.transcript_ja = enriched_transcript
    arch.summary_ja = ""
    arch.comments_raw = []
    arch.comments_summary_ja = ""
    arch.OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11435/api/generate")

    return arch


def _make_podcast_config(lc: LocalPipelineConfig):
    """LocalPipelineConfig → PipelineConfig に変換（factcheck/graph ステップで使用）。"""
    from factfull.podcast.pipeline import PipelineConfig

    return PipelineConfig(
        translate_model=lc.analyze_model,
        analyze_model=lc.analyze_model,
        extract_model=lc.analyze_model,
        factcheck_model=lc.factcheck_model,
        translate_chunk_size=lc.translate_chunk_size,
        summary_chunk_size=lc.summary_chunk_size,
        threshold=lc.threshold,
        max_iter=lc.max_iter,
        max_claims=lc.max_claims,
        top_k=lc.top_k,
        critique=lc.critique,
        editorial=lc.editorial,
        write_graph=lc.write_graph,
        output_base=lc.output_base,
        blog_name=lc.blog_name,
        reader_persona=lc.reader_persona,
        n_questions=lc.n_questions,
    )


def run_factcheck(result, config) -> float:
    """PipelineResult に対してファクトチェックを実行してスコアを返す。"""
    from factfull.podcast.steps.factcheck import run_factcheck_on_file

    summary_path = result.summary_path
    truth_path = result.episode_dir / f"transcript_{result.metadata.get('language', 'ja')}.txt"
    if not truth_path.exists():
        truth_path = result.episode_dir / "transcript_en.txt"

    return run_factcheck_on_file(config, summary_path, truth_path, result.episode_dir)
