"""
factfull/ingest/audio_transcriber.py
======================================
faster-whisper で文字起こし + pyannote.audio で話者分離（オプション）。

対応: MP3 / M4A / WAV / OGG（ffmpeg 経由）

使い方（話者分離なし）:
    result = transcribe("episode.mp3", output_dir=Path("./run"), language="ja")

使い方（話者分離あり）:
    result = transcribe(
        "episode.mp3",
        output_dir=Path("./run"),
        language="ja",
        diarize=True,
        hf_token="hf_xxx",           # HuggingFace トークン
        speakers=["草野美希", "宮武徹郎"],  # 話者名（順序は自動検出）
    )

出力テキスト形式（diarize=True）:
    [草野美希] こんにちは、オフトピックです。
    [宮武徹郎] 今日は最終回ということで...
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class TranscribeResult:
    text: str                    # 結合した全文テキスト（話者ラベル付き or なし）
    language: str                # 検出した言語コード ("ja", "en", ...)
    segments: list[dict]         # [{"start", "end", "text", "speaker"(optional)}, ...]
    duration: float              # 音声の長さ（秒）
    diarized: bool = False       # 話者分離が実行されたか


def transcribe(
    audio_path: str | Path,
    output_dir: Path | None = None,
    language: str | None = "ja",
    model_size: str = "large-v3",
    device: str = "cpu",
    compute_type: str = "int8",
    diarize: bool = False,
    hf_token: str | None = None,
    speakers: list[str] | None = None,
    num_speakers: int | None = None,
) -> TranscribeResult:
    """
    音声ファイルを文字起こしして TranscribeResult を返す。

    Args:
        audio_path:    音声ファイルパス
        output_dir:    出力ディレクトリ（None の場合は保存しない）
        language:      言語コード。None で自動検出
        model_size:    Whisper モデルサイズ
        device:        "cpu" or "cuda"
        compute_type:  "int8" (CPU) / "float16" (GPU)
        diarize:       True で pyannote 話者分離を実行
        hf_token:      HuggingFace トークン（diarize=True 時に必要）
                       省略時は HF_TOKEN 環境変数を参照
        speakers:      既知の話者名リスト（例: ["草野美希", "宮武徹郎"]）
                       指定すると SPEAKER_00/01 をこの名前にマッピングする
        num_speakers:  話者数のヒント（省略時は pyannote が自動推定）
    """
    audio_path = Path(audio_path)

    # Step 1: Whisper 文字起こし
    segments_list, detected_lang, duration = _run_whisper(
        audio_path, language, model_size, device, compute_type
    )

    # Whisper 完了直後にチェックポイント保存（diarize 失敗時に --regen で再利用可能にする）
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        _save_checkpoint(segments_list, detected_lang, duration, output_dir)

    # Step 2: 話者分離（オプション）
    if diarize:
        token = hf_token or os.environ.get("HF_TOKEN", "")
        if not token:
            print("  [diarize] ⚠️  HF_TOKEN 未設定 → 話者分離をスキップ")
        else:
            segments_list = _run_diarization(
                audio_path, segments_list, token, speakers, num_speakers
            )

    # テキスト組み立て
    diarized = diarize and any("speaker" in s for s in segments_list)
    if diarized:
        text_parts = [
            f"[{s['speaker']}] {s['text']}" if s.get("speaker") else s["text"]
            for s in segments_list
        ]
    else:
        text_parts = [s["text"] for s in segments_list]

    full_text = "\n".join(text_parts)
    print(f"  [whisper] 完了: {len(full_text):,} 文字 / {len(segments_list)} セグメント"
          + (" (話者分離済み)" if diarized else ""))

    result = TranscribeResult(
        text=full_text,
        language=detected_lang,
        segments=segments_list,
        duration=duration,
        diarized=diarized,
    )

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        _save(result, output_dir)

    return result


# ── 内部実装 ──────────────────────────────────────────────────────────────────

def _run_whisper(
    audio_path: Path,
    language: str | None,
    model_size: str,
    device: str,
    compute_type: str,
) -> tuple[list[dict], str, float]:
    from faster_whisper import WhisperModel

    print(f"  [whisper] モデル読み込み: {model_size} ({device}/{compute_type})")
    model = WhisperModel(model_size, device=device, compute_type=compute_type)

    print(f"  [whisper] 文字起こし開始: {audio_path.name}")
    segments_iter, info = model.transcribe(
        str(audio_path),
        language=language,
        beam_size=5,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500},
    )

    detected_lang = info.language
    duration = info.duration
    print(f"  [whisper] 言語: {detected_lang}  長さ: {duration/60:.1f}分")

    segments_list: list[dict] = []
    for seg in segments_iter:
        segments_list.append({
            "start": round(seg.start, 2),
            "end":   round(seg.end, 2),
            "text":  seg.text.strip(),
        })
        if seg.start % 60 < 2 and seg.start > 1:
            print(f"  [whisper] {seg.start/60:.0f}分 経過...", flush=True)

    return segments_list, detected_lang, duration


def _run_diarization(
    audio_path: Path,
    segments: list[dict],
    hf_token: str,
    speakers: list[str] | None,
    num_speakers: int | None,
) -> list[dict]:
    """pyannote で話者分離し、Whisper セグメントに speaker ラベルを付与して返す。"""
    try:
        from pyannote.audio import Pipeline
        import torch
    except ImportError:
        print("  [diarize] ⚠️  pyannote.audio 未インストール → スキップ")
        print("            uv pip install 'pyannote.audio>=3.1' で導入できます")
        return segments

    print("  [diarize] pyannote モデル読み込み中...")
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        token=hf_token,
    )

    # Apple Silicon では MPS を試みる、なければ CPU
    if torch.backends.mps.is_available():
        pipeline.to(torch.device("mps"))
        print("  [diarize] デバイス: MPS (Apple Silicon)")
    else:
        print("  [diarize] デバイス: CPU")

    kwargs: dict = {}
    if num_speakers is not None:
        kwargs["num_speakers"] = num_speakers
    elif speakers:
        kwargs["num_speakers"] = len(speakers)

    # MP3 は sample count が不正確なため WAV に変換してから渡す
    import subprocess, tempfile
    wav_tmp = Path(tempfile.mktemp(suffix=".wav"))
    try:
        print(f"  [diarize] WAV 変換中: {audio_path.name} → {wav_tmp.name} ...")
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(audio_path), "-ar", "16000", "-ac", "1", str(wav_tmp)],
            check=True, capture_output=True,
        )
        print(f"  [diarize] 話者分離実行中: {audio_path.name} ...")
        diarization = pipeline(str(wav_tmp), **kwargs)
    finally:
        wav_tmp.unlink(missing_ok=True)

    # pyannote の結果をフラットなリストに変換
    # [(start, end, speaker_label), ...]
    dia_segments: list[tuple[float, float, str]] = [
        (turn.start, turn.end, speaker)
        for turn, _, speaker in diarization.itertracks(yield_label=True)
    ]
    print(f"  [diarize] 話者数: {len({s for _, _, s in dia_segments})}人 / "
          f"セグメント数: {len(dia_segments)}")

    # 話者ラベル → 実名マッピングを構築
    speaker_map = _build_speaker_map(dia_segments, speakers)
    if speaker_map:
        print(f"  [diarize] マッピング: {speaker_map}")

    # Whisper セグメントに話者を割り当て（最大重複時間の話者を採用）
    labeled = []
    for seg in segments:
        speaker_label = _assign_speaker(seg["start"], seg["end"], dia_segments)
        labeled.append({
            **seg,
            "speaker": speaker_map.get(speaker_label, speaker_label) if speaker_label else None,
        })

    return labeled


def _build_speaker_map(
    dia_segments: list[tuple[float, float, str]],
    speakers: list[str] | None,
) -> dict[str, str]:
    """話者ラベル（SPEAKER_00 等）→ 実名のマッピングを構築する。

    speakers リストが指定されている場合、発話時間が多い順に名前を割り当てる。
    """
    if not speakers:
        return {}

    # 各話者の総発話時間を集計
    duration_by_label: dict[str, float] = {}
    for start, end, label in dia_segments:
        duration_by_label[label] = duration_by_label.get(label, 0.0) + (end - start)

    # 発話時間の多い順にソート
    sorted_labels = sorted(duration_by_label, key=lambda l: duration_by_label[l], reverse=True)

    return {label: name for label, name in zip(sorted_labels, speakers)}


def _assign_speaker(
    seg_start: float,
    seg_end: float,
    dia_segments: list[tuple[float, float, str]],
) -> str | None:
    """Whisper セグメントと最も重複する話者ラベルを返す。"""
    best_label: str | None = None
    best_overlap = 0.0

    for d_start, d_end, label in dia_segments:
        overlap = min(seg_end, d_end) - max(seg_start, d_start)
        if overlap > best_overlap:
            best_overlap = overlap
            best_label = label

    return best_label if best_overlap > 0 else None


def _save_checkpoint(
    segments: list[dict], lang: str, duration: float, output_dir: Path
) -> None:
    """Whisper 完了直後のチェックポイント。diarize 失敗時も transcript が残る。"""
    txt_name = f"transcript_{lang or 'xx'}.txt"
    text = "\n".join(s["text"] for s in segments)
    (output_dir / txt_name).write_text(text, encoding="utf-8")

    ts_data = {"language": lang, "duration": duration, "diarized": False, "segments": segments}
    (output_dir / "transcript_timestamped.json").write_text(
        json.dumps(ts_data, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"  [whisper] チェックポイント保存: {txt_name}, transcript_timestamped.json")


def _save(result: TranscribeResult, output_dir: Path) -> None:
    lang = result.language or "xx"
    txt_name = f"transcript_{lang}.txt"
    (output_dir / txt_name).write_text(result.text, encoding="utf-8")

    ts_data = {
        "language": result.language,
        "duration": result.duration,
        "diarized": result.diarized,
        "segments": result.segments,
    }
    (output_dir / "transcript_timestamped.json").write_text(
        json.dumps(ts_data, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"  [whisper] 保存: {txt_name}, transcript_timestamped.json")
