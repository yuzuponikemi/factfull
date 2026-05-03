"""
factfull/ingest/chunker.py
===========================
テキスト分割の共通ロジック。

すべての Ingestion モジュール（podcast / paper / book / web）が使う。
BM25 インデックス構築（indexer.py）も内部でこれを呼ぶ。

チャンク戦略:
  - character: 文字数ベース（デフォルト）。日英混在テキストに安定。
  - sentence: 文末（。.!?）で区切る。要約向き。
"""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class Chunk:
    text: str
    source: str   # ファイル名またはソース識別子
    offset: int   # 元テキスト内の先頭位置


def tokenize(text: str) -> list[str]:
    """日本語・英語混在対応の簡易トークナイザ。BM25 用。"""
    tokens: list[str] = []
    for m in re.finditer(r"[A-Za-z0-9]+|[^\s]", text):
        tokens.append(m.group().lower())
    return tokens


def chunk_by_chars(
    text: str,
    source: str = "",
    chunk_size: int = 400,
    overlap: int = 80,
) -> list[Chunk]:
    """文字数ベースでスライディングウィンドウ分割する。

    BM25 インデックス構築・ファクトチェック用。
    chunk_size=400, overlap=80 が factfull のデフォルト。
    """
    chunks: list[Chunk] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(Chunk(text=text[start:end], source=source, offset=start))
        if end == len(text):
            break
        start += chunk_size - overlap
    return chunks


def chunk_by_sentences(
    text: str,
    source: str = "",
    chunk_size: int = 1000,
    overlap: int = 200,
) -> list[Chunk]:
    """文末で区切ってからチャンクに詰める。

    要約パイプライン（archiver の summary_chunk_size）向き。
    """
    sentence_ends = re.compile(r"(?<=[。．.!?！？])\s*")
    sentences = sentence_ends.split(text)

    chunks: list[Chunk] = []
    buf = ""
    offset = 0
    buf_start = 0

    for sent in sentences:
        if not sent:
            continue
        if buf and len(buf) + len(sent) > chunk_size:
            chunks.append(Chunk(text=buf.strip(), source=source, offset=buf_start))
            # オーバーラップ: バッファ末尾 overlap 文字を引き継ぐ
            buf = buf[-overlap:] + sent
            buf_start = offset - overlap
        else:
            if not buf:
                buf_start = offset
            buf += sent
        offset += len(sent)

    if buf.strip():
        chunks.append(Chunk(text=buf.strip(), source=source, offset=buf_start))

    return chunks


def chunk_text(
    text: str,
    source: str = "",
    strategy: str = "character",
    chunk_size: int = 400,
    overlap: int = 80,
) -> list[Chunk]:
    """統合エントリポイント。strategy で分割方式を選択する。

    Args:
        strategy: "character"（デフォルト）または "sentence"
    """
    if strategy == "sentence":
        return chunk_by_sentences(text, source=source, chunk_size=chunk_size, overlap=overlap)
    return chunk_by_chars(text, source=source, chunk_size=chunk_size, overlap=overlap)
