"""
factfull/ingest/paper.py
=========================
arXiv 論文 / ローカル PDF → SourceDoc

移植元:
  kg-builder/src/kg_builder/processor/pdf_extractor.py  (PDF 抽出)
  kg-builder/scripts/download_arxiv_paper.py             (arXiv ダウンロード)
  infoseeker/src/core.py                                 (arXiv 検索)

使い方:
    from factfull.ingest.paper import ingest_arxiv, ingest_pdf

    # arXiv ID から直接取り込み
    doc = ingest_arxiv("2403.11996")

    # ローカル PDF から取り込み
    doc = ingest_pdf(Path("paper.pdf"))

    # キーワード検索して上位 N 件を取り込み
    docs = search_arxiv("knowledge graph LLM", max_results=5)
"""
from __future__ import annotations

import re
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from factfull.core.types import SourceDoc
from factfull.ingest.chunker import chunk_by_chars


# ── PDF テキスト抽出 ──────────────────────────────────────────────────────────

def _extract_pdf_text(pdf_path: Path) -> str:
    """pymupdf で PDF 全文を抽出する。"""
    try:
        import pymupdf  # type: ignore
    except ImportError as e:
        raise ImportError("PDF 取り込みには pymupdf が必要です: pip install pymupdf") from e

    parts: list[str] = []
    with pymupdf.open(str(pdf_path)) as doc:
        for page in doc:
            text = page.get_text()
            if text.strip():
                parts.append(text)
    return "\n\n".join(parts)


def _extract_pdf_metadata(pdf_path: Path) -> dict[str, Any]:
    """PDF メタデータと推定タイトルを返す。"""
    try:
        import pymupdf  # type: ignore
    except ImportError:
        return {"title": pdf_path.stem, "num_pages": 0}

    with pymupdf.open(str(pdf_path)) as doc:
        meta = doc.metadata or {}
        title = meta.get("title", "")
        if not title:
            first = doc[0].get_text() if doc.page_count > 0 else ""
            lines = [l.strip() for l in first.split("\n") if l.strip()]
            title = lines[0] if lines else pdf_path.stem
        return {
            "title": title,
            "author": meta.get("author", ""),
            "num_pages": doc.page_count,
            "file_size": pdf_path.stat().st_size,
        }


# ── arXiv ダウンロード ────────────────────────────────────────────────────────

def _clean_arxiv_id(arxiv_id: str) -> str:
    """'2403.11996v3' → '2403.11996'。URL から ID を取り出す処理も行う。"""
    # URL形式 (https://arxiv.org/abs/2403.11996)
    m = re.search(r"arxiv\.org/(?:abs|pdf)/([0-9]+\.[0-9]+)", arxiv_id)
    if m:
        arxiv_id = m.group(1)
    return arxiv_id.split("v")[0]


def _download_arxiv_pdf(arxiv_id: str, output_dir: Path) -> Path:
    """arXiv PDF をダウンロードしてパスを返す。"""
    clean_id = _clean_arxiv_id(arxiv_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / f"{clean_id.replace('.', '_')}.pdf"

    if pdf_path.exists():
        print(f"  [paper] キャッシュ使用: {pdf_path.name}", flush=True)
        return pdf_path

    url = f"https://arxiv.org/pdf/{clean_id}.pdf"
    print(f"  [paper] ダウンロード中: {url}", flush=True)
    req = urllib.request.Request(url, headers={
        "User-Agent": "factfull/1.0 (research pipeline; https://github.com/ikmx)"
    })
    with urllib.request.urlopen(req, timeout=60) as resp:
        pdf_path.write_bytes(resp.read())

    size_mb = pdf_path.stat().st_size / 1024 / 1024
    print(f"  [paper] 保存: {pdf_path.name} ({size_mb:.1f} MB)", flush=True)
    return pdf_path


# ── 公開 API ──────────────────────────────────────────────────────────────────

def ingest_pdf(
    pdf_path: Path,
    source_id: str | None = None,
    chunk_size: int = 2000,
    overlap: int = 200,
) -> SourceDoc:
    """ローカル PDF ファイルを SourceDoc に変換する。

    Args:
        pdf_path: PDF ファイルのパス
        source_id: ソース識別子（省略時はファイル名）
        chunk_size: チャンク文字数
        overlap: オーバーラップ文字数

    Returns:
        SourceDoc (source_type="paper")
    """
    pdf_path = Path(pdf_path)
    meta = _extract_pdf_metadata(pdf_path)
    text = _extract_pdf_text(pdf_path)
    chunks = chunk_by_chars(text, source=pdf_path.name, chunk_size=chunk_size, overlap=overlap)

    return SourceDoc(
        source_type="paper",
        source_id=source_id or pdf_path.stem,
        title=meta.get("title", pdf_path.stem),
        text=text,
        chunks=[c.text for c in chunks],
        metadata={
            **meta,
            "pdf_path": str(pdf_path),
        },
        created_at=datetime.now(timezone.utc).isoformat(),
    )


def ingest_arxiv(
    arxiv_id: str,
    output_dir: Path | None = None,
    chunk_size: int = 2000,
    overlap: int = 200,
) -> SourceDoc:
    """arXiv ID から論文をダウンロードして SourceDoc に変換する。

    Args:
        arxiv_id: arXiv ID（例: "2403.11996"）または arXiv URL
        output_dir: PDF 保存先（省略時: ~/papers/arxiv）
        chunk_size: チャンク文字数

    Returns:
        SourceDoc (source_type="paper")
    """
    clean_id = _clean_arxiv_id(arxiv_id)
    save_dir = output_dir or (Path.home() / "papers" / "arxiv")
    pdf_path = _download_arxiv_pdf(clean_id, save_dir)

    doc = ingest_pdf(pdf_path, source_id=clean_id, chunk_size=chunk_size, overlap=overlap)
    doc.metadata["arxiv_id"] = clean_id
    doc.metadata["arxiv_url"] = f"https://arxiv.org/abs/{clean_id}"

    # arxiv ライブラリが使えればメタデータを補完
    try:
        import arxiv  # type: ignore
        results = list(arxiv.Client().results(arxiv.Search(id_list=[clean_id], max_results=1)))
        if results:
            r = results[0]
            doc.title = r.title
            doc.metadata.update({
                "authors": [a.name for a in r.authors],
                "abstract": r.summary,
                "published": r.published.isoformat(),
                "categories": r.categories,
            })
    except Exception:
        pass  # arxiv ライブラリなし or ネットワークエラーは無視

    return doc


def search_arxiv(
    query: str,
    max_results: int = 5,
    output_dir: Path | None = None,
    download: bool = False,
) -> list[SourceDoc]:
    """arXiv キーワード検索して SourceDoc リストを返す。

    download=True のとき PDF をダウンロードしてテキスト抽出まで行う。
    download=False（デフォルト）のときはメタデータのみの SourceDoc を返す。

    Args:
        query: 検索クエリ
        max_results: 最大取得件数
        output_dir: PDF 保存先（download=True 時）
        download: True のとき PDF を取得してフルテキストを含める

    Returns:
        SourceDoc のリスト
    """
    try:
        import arxiv  # type: ignore
    except ImportError as e:
        raise ImportError("arXiv 検索には arxiv パッケージが必要です: pip install arxiv") from e

    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
    )

    docs: list[SourceDoc] = []
    for result in client.results(search):
        clean_id = result.entry_id.split("/")[-1].split("v")[0]

        if download:
            try:
                doc = ingest_arxiv(clean_id, output_dir=output_dir)
                docs.append(doc)
            except Exception as e:
                print(f"  [paper] ダウンロード失敗 {clean_id}: {e}", flush=True)
        else:
            # メタデータのみ（テキスト・チャンクは空）
            docs.append(SourceDoc(
                source_type="paper",
                source_id=clean_id,
                title=result.title,
                text=result.summary,   # abstract をテキストとして使用
                metadata={
                    "arxiv_id": clean_id,
                    "arxiv_url": f"https://arxiv.org/abs/{clean_id}",
                    "authors": [a.name for a in result.authors],
                    "abstract": result.summary,
                    "published": result.published.isoformat(),
                    "categories": result.categories,
                    "pdf_url": result.pdf_url,
                },
                created_at=datetime.now(timezone.utc).isoformat(),
            ))

    print(f"  [paper] 検索完了: {len(docs)} 件 (query={query!r})", flush=True)
    return docs
