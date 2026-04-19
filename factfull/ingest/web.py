"""
factfull/ingest/web.py
=======================
Web URL → SourceDoc

HTML を取得してメインテキストを抽出する。
trafilatura が使える場合は本文抽出の精度が上がる（任意依存）。
なければ標準ライブラリの html.parser で最低限の抽出を行う。

使い方:
    from factfull.ingest.web import ingest_url

    doc = ingest_url("https://example.com/article")
    doc = ingest_url("https://example.com/article", title="カスタムタイトル")
"""
from __future__ import annotations

import html
import re
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from factfull.core.types import SourceDoc
from factfull.ingest.chunker import chunk_by_chars


# ── HTML テキスト抽出 ─────────────────────────────────────────────────────────

def _fetch_html(url: str) -> str:
    req = urllib.request.Request(url, headers={
        "User-Agent": "Mozilla/5.0 (compatible; factfull/1.0)"
    })
    with urllib.request.urlopen(req, timeout=30) as resp:
        raw = resp.read()
        charset = resp.headers.get_content_charset() or "utf-8"
        return raw.decode(charset, errors="replace")


def _extract_title(html_text: str) -> str:
    m = re.search(r"<title[^>]*>([^<]+)</title>", html_text, re.IGNORECASE)
    return html.unescape(m.group(1).strip()) if m else ""


def _strip_html(html_text: str) -> str:
    """trafilatura なしの簡易 HTML → テキスト変換。"""
    # script / style / head を除去
    for tag in ("script", "style", "head", "nav", "footer", "header"):
        html_text = re.sub(rf"<{tag}[^>]*>.*?</{tag}>", " ", html_text, flags=re.DOTALL | re.IGNORECASE)
    # タグ除去
    text = re.sub(r"<[^>]+>", " ", html_text)
    # HTML エンティティ
    text = html.unescape(text)
    # 連続空白
    text = re.sub(r"\s{3,}", "\n\n", text)
    return text.strip()


def _extract_text(html_text: str) -> str:
    """trafilatura 優先、なければフォールバック。"""
    try:
        import trafilatura  # type: ignore
        result = trafilatura.extract(html_text, include_comments=False, include_tables=False)
        if result:
            return result
    except ImportError:
        pass
    return _strip_html(html_text)


# ── 公開 API ──────────────────────────────────────────────────────────────────

def ingest_url(
    url: str,
    title: str = "",
    chunk_size: int = 1500,
    overlap: int = 150,
    cache_path: Path | None = None,
) -> SourceDoc:
    """Web URL を取得して SourceDoc に変換する。

    Args:
        url: 取得する URL
        title: タイトル（省略時は <title> タグから取得）
        chunk_size: チャンク文字数
        overlap: オーバーラップ文字数
        cache_path: HTML キャッシュ保存先（省略時はキャッシュしない）

    Returns:
        SourceDoc (source_type="web")
    """
    if cache_path and cache_path.exists():
        print(f"  [web] キャッシュ使用: {cache_path.name}", flush=True)
        html_text = cache_path.read_text(encoding="utf-8")
    else:
        print(f"  [web] 取得中: {url}", flush=True)
        html_text = _fetch_html(url)
        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(html_text, encoding="utf-8")

    extracted_title = title or _extract_title(html_text)
    text = _extract_text(html_text)
    chunks = chunk_by_chars(text, source=url, chunk_size=chunk_size, overlap=overlap)
    print(f"  [web] 抽出: {len(text)}字 / {len(chunks)} チャンク", flush=True)

    return SourceDoc(
        source_type="web",
        source_id=url,
        title=extracted_title,
        text=text,
        chunks=[c.text for c in chunks],
        metadata={
            "url": url,
            "char_count": len(text),
        },
        created_at=datetime.now(timezone.utc).isoformat(),
    )
