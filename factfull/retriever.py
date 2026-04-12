"""
BM25 インデックスからクレームに関連する証拠パッセージを取得する。
"""
from __future__ import annotations
from .indexer import BM25Okapi, Chunk, _tokenize


def retrieve(
    claim: str,
    bm25: BM25Okapi,
    chunks: list[Chunk],
    top_k: int = 5,
) -> list[Chunk]:
    """claim に最も関連するチャンクを top_k 件返す。"""
    tokens = _tokenize(claim)
    scores = bm25.get_scores(tokens)

    ranked = sorted(
        range(len(chunks)),
        key=lambda i: scores[i],
        reverse=True,
    )[:top_k]

    # スコアが 0 のものは除外（完全に無関係）
    return [chunks[i] for i in ranked if scores[i] > 0]
