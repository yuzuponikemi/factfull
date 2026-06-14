"""
factfull/bilingual/pipeline.py
==============================
論文 → 英日対訳 JSON パイプラインのオーケストレーター。

固定ステップで LLM を N 回呼ぶ（ReAct ループではない）ため、CLAUDE.md の
方針どおり factfull 内に置く。

  1. dispatch     : arXiv ID/URL か ローカル PDF を判別して SourceDoc + pdf_path を得る
  2. extract      : pymupdf 構造抽出（テキスト＋図表）
  3. segment      : 見出し/段落/図表/参考文献に整形（en のみ）
  4. translate    : 段落バッチ翻訳（ja 充填）＋タイトル/アブストラクト翻訳
  5. write        : BilingualDoc を JSON に書き出し

使い方:
    from factfull.bilingual.pipeline import BilingualConfig, run_bilingual
    result = run_bilingual(BilingualConfig(), "2403.11996")
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from factfull.bilingual.extract import extract_structured_blocks
from factfull.bilingual.segment import segment_blocks
from factfull.bilingual.translate import _translate_one, translate_blocks
from factfull.bilingual.types import Block, BilingualDoc
from factfull.core.types import SourceDoc
from factfull.ingest.paper import ingest_arxiv, ingest_pdf


@dataclass
class BilingualConfig:
    model: str = "translategemma:12b"
    batch_chars: int = 3000
    num_ctx: int = 8192
    num_predict: int = 8192
    skip_references: bool = True
    skip_captions: bool = False
    heading_size_ratio: float = 1.08
    min_paragraph_chars: int = 3
    output_base: Path = field(default_factory=lambda: Path.home() / "papers" / "bilingual")
    dump_raw: bool = False


@dataclass
class BilingualResult:
    source_id: str
    title_en: str
    title_ja: str
    json_path: Path
    out_dir: Path
    n_blocks: int
    model: str


def _dispatch(source: str) -> tuple[SourceDoc, Path, str]:
    """source を arXiv とローカル PDF に振り分ける。"""
    p = Path(source).expanduser()
    if p.exists() and p.suffix.lower() == ".pdf":
        doc = ingest_pdf(p)
        return doc, p, "pdf"
    doc = ingest_arxiv(source)
    return doc, Path(doc.metadata["pdf_path"]), "arxiv"


def run_bilingual(config: BilingualConfig, source: str) -> BilingualResult:
    """論文 1 本を英日対訳 JSON に変換する。"""
    os.environ.setdefault("FACTFULL_LLM_BACKEND", "ollama")

    print(f"📥 取り込み: {source}", flush=True)
    doc, pdf_path, source_type = _dispatch(source)
    out_dir = config.output_base / doc.source_id
    assets_dir = out_dir / "assets"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"📄 構造抽出: {pdf_path.name}", flush=True)
    raw = extract_structured_blocks(pdf_path)
    if config.dump_raw:
        _dump_raw(raw, out_dir)

    blocks = segment_blocks(
        raw,
        skip_references=config.skip_references,
        skip_captions=config.skip_captions,
        heading_size_ratio=config.heading_size_ratio,
        min_paragraph_chars=config.min_paragraph_chars,
        assets_dir=assets_dir,
    )
    title_en = doc.title or pdf_path.stem
    abstract_en = str(doc.metadata.get("abstract", "")).strip()

    # 先頭の見出しが論文タイトルと一致するなら type="title" に格上げ
    # （homupe でタイトルブロックとして扱えるようにする）
    _norm = lambda s: " ".join(s.lower().split())
    if blocks and blocks[0].type == "heading" and _norm(blocks[0].en) == _norm(title_en):
        blocks[0].type = "title"
        blocks[0].level = None

    n_text = sum(1 for b in blocks if b.en)
    n_fig = sum(1 for b in blocks if b.type in ("figure", "table"))
    print(f"   ブロック: {len(blocks)}（テキスト {n_text} / 図表 {n_fig}）", flush=True)

    print(f"🌐 翻訳（{config.model}）...", flush=True)
    translate_blocks(
        blocks, title_en,
        model=config.model, batch_chars=config.batch_chars,
        num_ctx=config.num_ctx, num_predict=config.num_predict,
    )
    title_ja = _translate_title(title_en, config)
    abstract_ja = ""
    if abstract_en:
        abstract_ja = _translate_one(
            Block(id="", type="abstract", en=abstract_en), title_en,
            model=config.model, num_ctx=config.num_ctx, num_predict=config.num_predict,
        )

    bdoc = BilingualDoc(
        title_en=title_en,
        title_ja=title_ja,
        authors=list(doc.metadata.get("authors", [])),
        source_id=doc.source_id,
        arxiv_url=str(doc.metadata.get("arxiv_url", "")),
        source_type=source_type,
        model=config.model,
        translated_at=datetime.now(timezone.utc).isoformat(),
        abstract_en=abstract_en,
        abstract_ja=abstract_ja,
        metadata={
            k: doc.metadata.get(k)
            for k in ("num_pages", "published", "categories", "author")
            if k in doc.metadata
        },
        blocks=blocks,
    )

    json_path = out_dir / "bilingual.json"
    json_path.write_text(
        json.dumps(bdoc.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"✅ 対訳 JSON: {json_path}", flush=True)

    return BilingualResult(
        source_id=doc.source_id,
        title_en=title_en,
        title_ja=title_ja,
        json_path=json_path,
        out_dir=out_dir,
        n_blocks=len(blocks),
        model=config.model,
    )


def _translate_title(title_en: str, config: BilingualConfig) -> str:
    """タイトルを 1 行で日本語訳する。"""
    from factfull.llm import call
    prompt = (
        "次の英語の論文タイトルを、自然で正確な日本語に翻訳してください。"
        "訳のみを 1 行で出力してください（前置き・引用符不要）。\n\n"
        f"{title_en}"
    )
    return call(
        prompt, model=config.model, num_ctx=config.num_ctx, num_predict=512
    ).strip().strip("「」\"'")


def _dump_raw(raw, out_dir: Path) -> None:
    """生テキストブロックを extract_raw.json に書き出す（デバッグ用、画像バイトは除外）。"""
    data = [
        {
            "kind": b.kind, "page": b.page, "bbox": list(b.bbox),
            "text": b.text, "font_size": b.font_size, "bold": b.bold,
        }
        for b in raw
    ]
    (out_dir / "extract_raw.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )
