"""
factfull/ingest/audio_transcriber.py
======================================
faster-whisper を使ってローカル音声ファイルを文字起こしする。

対応: MP3 / M4A / WAV / OGG（ffmpeg 経由）
日本語音声の場合 language="ja" を指定すると精度が上がる。

使い方:
    from factfull.ingest.audio_transcriber import transcribe

    text, segments = transcribe(
        "episode.mp3",
        output_dir=Path("./run"),
        language="ja",
        model_size="large-v3",
    )
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class TranscribeResult:
    text: str               # 結合した全文テキスト
    language: str           # 検出した言語コード ("ja", "en", ...)
    segments: list[dict]    # [{"start": float, "end": float, "text": str}, ...]
    duration: float         # 音声の長さ（秒）


def transcribe(
    audio_path: str | Path,
    output_dir: Path | None = None,
    language: str | None = "ja",
    model_size: str = "large-v3",
    device: str = "cpu",
    compute_type: str = "int8",
) -> TranscribeResult:
    """
    音声ファイルを文字起こしして TranscribeResult を返す。

    output_dir を指定すると以下を保存:
      transcript_ja.txt  (language="ja" の場合) or transcript_en.txt
      transcript_timestamped.json

    Args:
        audio_path:    音声ファイルパス
        output_dir:    出力ディレクトリ（None の場合は保存しない）
        language:      言語コード。None で自動検出
        model_size:    Whisper モデルサイズ (tiny/base/small/medium/large-v3/large-v3-turbo)
        device:        "cpu" or "cuda"
        compute_type:  "int8" (CPU) / "float16" (GPU)
    """
    from faster_whisper import WhisperModel

    audio_path = Path(audio_path)
    print(f"  [whisper] モデル読み込み: {model_size} ({device}/{compute_type})")
    model = WhisperModel(model_size, device=device, compute_type=compute_type)

    print(f"  [whisper] 文字起こし開始: {audio_path.name}")
    segments_iter, info = model.transcribe(
        str(audio_path),
        language=language,
        beam_size=5,
        vad_filter=True,          # 無音区間をスキップ
        vad_parameters={"min_silence_duration_ms": 500},
    )

    detected_lang = info.language
    duration = info.duration
    print(f"  [whisper] 言語: {detected_lang}  長さ: {duration/60:.1f}分")

    segments_list = []
    text_parts = []
    for seg in segments_iter:
        segments_list.append({
            "start": round(seg.start, 2),
            "end":   round(seg.end, 2),
            "text":  seg.text.strip(),
        })
        text_parts.append(seg.text.strip())
        # 進捗表示（1分ごと）
        if seg.start % 60 < 2 and seg.start > 1:
            print(f"  [whisper] {seg.start/60:.0f}分 経過...", flush=True)

    full_text = "\n".join(text_parts)
    print(f"  [whisper] 完了: {len(full_text):,} 文字 / {len(segments_list)} セグメント")

    result = TranscribeResult(
        text=full_text,
        language=detected_lang,
        segments=segments_list,
        duration=duration,
    )

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        _save(result, output_dir)

    return result


def _save(result: TranscribeResult, output_dir: Path) -> None:
    lang = result.language or "xx"
    # テキストファイル
    txt_name = f"transcript_{lang}.txt"
    (output_dir / txt_name).write_text(result.text, encoding="utf-8")

    # タイムスタンプ付き JSON
    ts_data = {
        "language": result.language,
        "duration": result.duration,
        "segments": result.segments,
    }
    (output_dir / "transcript_timestamped.json").write_text(
        json.dumps(ts_data, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"  [whisper] 保存: {txt_name}, transcript_timestamped.json")
