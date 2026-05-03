"""
Truth ソースをチャンク化して BM25 インデックスを構築する。

チャンキングの実装は factfull.ingest.chunker に委譲。
Chunk / _tokenize は後方互換のため再エクスポートする。
"""
from __future__ import annotations

from pathlib import Path

from rank_bm25 import BM25Okapi  # type: ignore

from factfull.ingest.chunker import Chunk, chunk_by_chars, tokenize as _tokenize_fn

# 後方互換エイリアス
def _tokenize(text: str) -> list[str]:
    return _tokenize_fn(text)


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
        chunks.extend(chunk_by_chars(text, source=path.name, chunk_size=chunk_size, overlap=overlap))

    if not chunks:
        raise ValueError("Truth ソースにテキストが見つかりません")

    tokenized = [_tokenize_fn(c.text) for c in chunks]
    bm25 = BM25Okapi(tokenized)
    return bm25, chunks
