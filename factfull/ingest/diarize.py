"""
factfull/ingest/diarize.py
===========================
pyannote.audio を使った話者分離（Speaker Diarization）。

フロー:
  1. yt-dlp で YouTube から音声ダウンロード（WAV）
  2. pyannote.audio で話者ターン分割 → [{start, end, speaker}]
  3. 既存の YouTube トランスクリプト（タイムスタンプ付き）と突合
  4. 話者ラベルを名前にマッピング（ゲスト名はタイトルから、ホストはチャンネル名）
  5. 話者付きテキストとして保存

前提:
  - HuggingFace トークン（HF_TOKEN 環境変数）
    https://hf.co/pyannote/speaker-diarization-3.1 でアクセス承認が必要
  - ffmpeg インストール済み
  - uv sync --extra diarize 実行済み
"""
from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any


# ── 音声ダウンロード ──────────────────────────────────────────────────────────

def download_audio(video_id: str, out_path: Path) -> Path:
    """YouTube から音声を WAV でダウンロードする。既存ファイルはスキップ。"""
    if out_path.exists():
        print(f"  [diarize] 音声ファイル再利用: {out_path.name}", flush=True)
        return out_path

    url = f"https://www.youtube.com/watch?v={video_id}"
    # yt-dlp は拡張子を自動付与するので stem を渡す
    stem = str(out_path.with_suffix(""))
    cmd = [
        "yt-dlp", url,
        "-x", "--audio-format", "wav",
        "--audio-quality", "0",
        "-o", f"{stem}.%(ext)s",
        "--no-playlist",
    ]
    print(f"  [diarize] 音声ダウンロード中: {video_id}", flush=True)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"yt-dlp failed: {result.stderr[:300]}")

    # yt-dlp が出力したファイルを探す
    candidates = list(out_path.parent.glob(f"{out_path.stem}.*"))
    if not candidates:
        raise FileNotFoundError(f"yt-dlp output not found under {out_path.parent}")
    actual = candidates[0]
    if actual != out_path:
        actual.rename(out_path)
    return out_path


# ── 話者分離 ─────────────────────────────────────────────────────────────────

def run_diarization(
    audio_path: Path,
    hf_token: str,
    num_speakers: int = 2,
) -> list[dict[str, Any]]:
    """pyannote.audio で話者ターンを検出する。

    Returns:
        [{start: float, end: float, speaker: str}, ...]
    """
    from pyannote.audio import Pipeline
    import torch

    print(f"  [diarize] 話者分離開始: {audio_path.name}", flush=True)
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        token=hf_token,
    )

    if torch.backends.mps.is_available():
        pipeline.to(torch.device("mps"))
    elif torch.cuda.is_available():
        pipeline.to(torch.device("cuda"))

    output = pipeline(str(audio_path), num_speakers=num_speakers)

    # pyannote v4: returns DiarizeOutput; use exclusive_speaker_diarization
    # (no overlapping turns) for cleaner alignment
    annotation = output.exclusive_speaker_diarization
    segments = [
        {"start": round(turn.start, 2), "end": round(turn.end, 2), "speaker": speaker}
        for turn, _, speaker in annotation.itertracks(yield_label=True)
    ]
    print(f"  [diarize] ターン検出: {len(segments)} 件", flush=True)
    return segments


# ── トランスクリプトとの突合 ──────────────────────────────────────────────────

def align_with_transcript(
    diarization: list[dict[str, Any]],
    transcript_raw: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """YouTube トランスクリプトの各セグメントに話者ラベルを付与する。

    Args:
        diarization: [{start, end, speaker}, ...]
        transcript_raw: [{text, start, duration}, ...]  ← YouTube Caption API 形式

    Returns:
        [{text, start, duration, speaker}, ...]
    """
    result = []
    for seg in transcript_raw:
        t = float(seg["start"])
        speaker = "UNKNOWN"
        # 最も重なりが大きいターンを選ぶ
        best_overlap = 0.0
        seg_end = t + float(seg.get("duration", 0))
        for d in diarization:
            overlap = min(seg_end, d["end"]) - max(t, d["start"])
            if overlap > best_overlap:
                best_overlap = overlap
                speaker = d["speaker"]
        result.append({**seg, "speaker": speaker})
    return result


# ── 話者名マッピング ──────────────────────────────────────────────────────────

def infer_speaker_names(
    labeled_segments: list[dict[str, Any]],
    title: str,
    channel: str,
) -> dict[str, str]:
    """SPEAKER_00 / SPEAKER_01 ラベルを実際の名前にマッピングする。

    ヒューリスティック:
      - 総発話時間が長い方 → ゲスト（インタビューされる側）
      - 短い方 → ホスト（チャンネル名）
      - ゲスト名はタイトルから抽出（"Name – topic" / "Name: topic" 形式）
    """
    # ゲスト名をタイトルから抽出
    guest_name = _parse_guest_from_title(title)

    # 話者ごとの総発話時間を集計
    time_by_speaker: dict[str, float] = {}
    for seg in labeled_segments:
        sp = seg.get("speaker", "UNKNOWN")
        time_by_speaker[sp] = time_by_speaker.get(sp, 0.0) + float(seg.get("duration", 0))

    if len(time_by_speaker) < 2:
        # 話者が1人しか検出されなかった場合
        only = next(iter(time_by_speaker), "SPEAKER_00")
        return {only: guest_name or channel or "Speaker"}

    sorted_speakers = sorted(time_by_speaker, key=lambda s: time_by_speaker[s], reverse=True)
    guest_label = sorted_speakers[0]
    host_label  = sorted_speakers[1]

    name_map = {
        guest_label: guest_name or "Guest",
        host_label:  channel or "Host",
    }
    print(
        f"  [diarize] 話者マッピング: {host_label}→{name_map[host_label]}, "
        f"{guest_label}→{name_map[guest_label]}",
        flush=True,
    )
    return name_map


def _parse_guest_from_title(title: str) -> str:
    """タイトルからゲスト名を抽出する。

    対応パターン:
      "Adam Marblestone – AI is missing..."   → "Adam Marblestone"
      "Sam Altman: The Future of AI"          → "Sam Altman"
      "#123 – Adam Marblestone"               → "Adam Marblestone"
    """
    # エピソード番号プレフィックスを除去（"#123 –", "Ep 12:"）
    cleaned = re.sub(r"^(#?\d+\s*[–—|:]\s*)", "", title).strip()
    # "Name – topic" or "Name: topic" の Name 部分を取る
    m = re.match(r"^([A-Z][^–—|:\n]{2,40}?)\s*[–—|:]", cleaned)
    if m:
        candidate = m.group(1).strip()
        # 全部大文字でない、2語以上、長すぎないことを確認
        if len(candidate.split()) >= 2 and len(candidate) <= 40:
            return candidate
    return ""


# ── テキスト出力 ──────────────────────────────────────────────────────────────

def to_plain_text(
    labeled_segments: list[dict[str, Any]],
    name_map: dict[str, str],
) -> str:
    """話者ラベル付きプレーンテキストに変換する。

    同一話者が連続する場合はまとめて1ブロックにする。
    出力例:
        [Lex Fridman] Welcome back everyone...
        [Adam Marblestone] Thanks for having me...
    """
    lines: list[str] = []
    current_speaker: str | None = None
    buffer: list[str] = []

    def _flush() -> None:
        if buffer and current_speaker is not None:
            lines.append(f"[{current_speaker}] " + " ".join(buffer))

    for seg in labeled_segments:
        raw_sp = seg.get("speaker", "UNKNOWN")
        sp = name_map.get(raw_sp, raw_sp)
        text = seg.get("text", "").strip()
        if not text:
            continue
        if sp != current_speaker:
            _flush()
            current_speaker = sp
            buffer = [text]
        else:
            buffer.append(text)

    _flush()
    return "\n".join(lines)


# ── ワンショット API ──────────────────────────────────────────────────────────

def diarize_episode(
    video_id: str,
    episode_dir: Path,
    title: str = "",
    channel: str = "",
    num_speakers: int = 2,
    hf_token: str | None = None,
) -> str:
    """1エピソードを話者分離して diarized テキストを返す。

    副作用:
      episode_dir/audio.wav                  ← ダウンロード済み音声
      episode_dir/diarization.json           ← 生のターンリスト
      episode_dir/transcript_en_diarized.json← 話者ラベル付きセグメント
      episode_dir/transcript_en_diarized.txt ← 人間可読テキスト

    Returns:
        話者ラベル付きプレーンテキスト
    """
    token = hf_token or os.environ.get("HF_TOKEN", "")
    if not token:
        raise ValueError("HF_TOKEN が未設定です。環境変数か引数で渡してください。")

    audio_path = episode_dir / "audio.wav"
    diarization_path = episode_dir / "diarization.json"

    # 既存の分離結果があれば再利用
    if diarization_path.exists():
        print(f"  [diarize] 分離結果再利用: {diarization_path.name}", flush=True)
        diarization = json.loads(diarization_path.read_text(encoding="utf-8"))
    else:
        download_audio(video_id, audio_path)
        diarization = run_diarization(audio_path, token, num_speakers=num_speakers)
        diarization_path.write_text(
            json.dumps(diarization, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    # 既存のタイムスタンプ付きトランスクリプトと突合
    ts_path = episode_dir / "transcript_en_timestamped.json"
    if not ts_path.exists():
        raise FileNotFoundError(
            f"タイムスタンプ付きトランスクリプトが見つかりません: {ts_path}"
        )
    transcript_raw = json.loads(ts_path.read_text(encoding="utf-8"))

    labeled = align_with_transcript(diarization, transcript_raw)
    name_map = infer_speaker_names(labeled, title=title, channel=channel)

    # 結果を保存
    labeled_path = episode_dir / "transcript_en_diarized.json"
    labeled_path.write_text(
        json.dumps(labeled, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    text = to_plain_text(labeled, name_map)
    text_path = episode_dir / "transcript_en_diarized.txt"
    text_path.write_text(text, encoding="utf-8")

    print(f"  [diarize] 完了: {text_path}", flush=True)
    return text
