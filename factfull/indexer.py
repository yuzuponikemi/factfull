"""
Truth ソースをチャンク化して BM25 インデックスを構築する。
"""
from __future__ import annotations
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List

from rank_bm25 import BM25Okapi  # type: ignore


@dataclass
class Chunk:
    text: str
    source: str   # ファイル名
    offset: int   # テキスト内の先頭位置


def _tokenize(text: str) -> list[str]:
    """日本語・英語混在対応の簡易トークナイザ。"""
    # 英数字はそのまま、日本語は1文字ずつ
    tokens: list[str] = []
    for m in re.finditer(r"[A-Za-z0-9]+|[^\s]", text):
        tokens.append(m.group().lower())
    return tokens


def build_index(
    truth_paths: list[Path],
    chunk_size: int = 400,
    overlap: int = 80,
) -> tuple[BM25Okapi, list[Chunk]]:
    """
    truth_paths のテキストをチャンク化し BM25 インデックスを返す。
    戻り値: (bm25, chunks)  ※ chunks[i] が bm25 の i 番目の文書に対応
    """
    chunks: list[Chunk] = []

    for path in truth_paths:
        try:
            text = path.read_text(encoding="utf-8")
        except Exception:
            continue
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunks.append(Chunk(text=text[start:end], source=path.name, offset=start))
            if end == len(text):
                break
            start += chunk_size - overlap

    if not chunks:
        raise ValueError("Truth ソースにテキストが見つかりません")

    tokenized = [_tokenize(c.text) for c in chunks]
    bm25 = BM25Okapi(tokenized)
    return bm25, chunks
