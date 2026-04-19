"""
factfull/ingest/book.py
========================
書籍テキスト（Gutenberg / ローカルファイル / URL） → SourceDoc

移植元: cogito/services/ingestor/adapters/book.py

使い方:
    from factfull.ingest.book import ingest_book

    # Project Gutenberg URL から
    doc = ingest_book(
        source="https://www.gutenberg.org/cache/epub/5827/pg5827.txt",
        title="Discourse on the Method",
        author="René Descartes",
        source_type="gutenberg",
    )

    # ローカルテキストファイルから
    doc = ingest_book(Path("book.txt"), title="My Book")
"""
from __future__ import annotations

import re
import urllib.request
from pathlib import Path
from datetime import datetime, timezone
from typing import Literal

from factfull.core.types import SourceDoc
from factfull.ingest.chunker import chunk_by_chars

BookSourceType = Literal["gutenberg", "url", "local_file"]


# ── テキスト取得 ──────────────────────────────────────────────────────────────

def _download_text(url: str, cache_path: Path | None = None) -> str:
    if cache_path and cache_path.exists():
        return cache_path.read_text(encoding="utf-8")
    req = urllib.request.Request(url, headers={"User-Agent": "factfull/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        text = resp.read().decode("utf-8", errors="replace")
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(text, encoding="utf-8")
    return text


def _load_local(path: Path) -> str:
    return path.read_text(encoding="utf-8")


# ── テキスト整形 ──────────────────────────────────────────────────────────────

def _strip_gutenberg(text: str) -> str:
    """Project Gutenberg のヘッダー・フッターを除去する。"""
    start = re.search(r"\*\*\* START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK .+? \*\*\*", text)
    end = re.search(r"\*\*\* END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK .+? \*\*\*", text)
    if start:
        text = text[start.end():]
    if end:
        text = text[:end.start()]
    return text.strip()


# ── 公開 API ──────────────────────────────────────────────────────────────────

def ingest_book(
    source: str | Path,
    title: str = "",
    author: str = "",
    source_type: BookSourceType = "local_file",
    cache_dir: Path | None = None,
    chunk_size: int = 2000,
    overlap: int = 200,
) -> SourceDoc:
    """書籍テキストを SourceDoc に変換する。

    Args:
        source: URL（gutenberg/url）またはローカルファイルパス（local_file）
        title: 書籍タイトル（省略時はファイル名またはURL末尾）
        author: 著者名
        source_type: "gutenberg" | "url" | "local_file"
        cache_dir: ダウンロードキャッシュ保存先
        chunk_size: チャンク文字数
        overlap: オーバーラップ文字数

    Returns:
        SourceDoc (source_type="book")
    """
    source_str = str(source)

    if source_type in ("gutenberg", "url"):
        cache_path = None
        if cache_dir:
            safe_name = re.sub(r"[^\w]", "_", source_str[-40:]) + ".txt"
            cache_path = cache_dir / safe_name
        print(f"  [book] ダウンロード中: {source_str[:80]}", flush=True)
        text = _download_text(source_str, cache_path)
        if source_type == "gutenberg":
            text = _strip_gutenberg(text)
        source_id = source_str
        display_title = title or source_str.split("/")[-1]
    else:
        path = Path(source)
        text = _load_local(path)
        source_id = str(path)
        display_title = title or path.stem

    chunks = chunk_by_chars(text, source=display_title, chunk_size=chunk_size, overlap=overlap)
    print(f"  [book] チャンク数: {len(chunks)}", flush=True)

    return SourceDoc(
        source_type="book",
        source_id=source_id,
        title=display_title,
        text=text,
        chunks=[c.text for c in chunks],
        metadata={
            "author": author,
            "book_source_type": source_type,
            "source_url": source_str if source_type in ("gutenberg", "url") else None,
            "char_count": len(text),
        },
        created_at=datetime.now(timezone.utc).isoformat(),
    )
