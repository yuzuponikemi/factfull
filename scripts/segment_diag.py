#!/usr/bin/env python3
"""
scripts/segment_diag.py
=======================
bilingual パイプラインの抽出/セグメンテーション層ロバストネス診断。

翻訳は一切回さず（遅い・本筋でない）、`extract_structured_blocks` →
`segment_blocks` だけを多様な論文に流し、以下の故障モードを定量化する:

  A. 著者誤分類       : page1 の heading に所属/メールらしき行が混ざる
  B. 読み順崩れリスク  : 二段組ページの途中に全幅図表が割り込む構成
  C. キャプション誤判定: 本文中の "Figure 1 shows ..." を caption 化
  D. 図ラベル付与漏れ  : figure/table に label が付かない / 孤立 caption
  E. フロー分断        : 終端なし段落の直後が heading/figure（読み順断絶の兆候）

出力:
  logs/segment_diag-<ts>.json   機械可読の全結果
  標準出力                       人間可読のサマリ表＋実例サンプル

使い方:
  .venv/bin/python scripts/segment_diag.py            # 既定の10本
  .venv/bin/python scripts/segment_diag.py 1706.03762 1810.04805
"""
from __future__ import annotations

import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

from factfull.bilingual.extract import RawBlock, extract_structured_blocks
from factfull.bilingual.segment import segment_blocks
from factfull.ingest.paper import ingest_arxiv

# 多様な版面: 2段組(NIPS/CVPR/ACL) と 単段(ICLR/arXiv preprint)、図/表/数式重め
DEFAULT_PAPERS = [
    "1706.03762",  # Attention      2col NIPS   （比較基準）
    "1810.04805",  # BERT           2col ACL
    "1512.03385",  # ResNet         2col CVPR    全幅図多い
    "1406.2661",   # GAN            2col NIPS
    "1505.04597",  # U-Net          2col         大図
    "1301.3781",   # word2vec       2col
    "1409.1556",   # VGG            単段 ICLR    表重め
    "1412.6980",   # Adam           単段 ICLR    数式重め
    "1503.02531",  # Distilling KD  単段
    "2010.11929",  # ViT            単段
]

_TERMINAL = ".?!:;。．！？”)」】"
# 正規キャプション: "Figure 1:" / "Table 2." のように番号直後が区切り
PROPER_CAPTION = re.compile(
    r"^(figure|fig\.?|table|algorithm|alg\.?)\s*\d+\s*[:.—\-]", re.IGNORECASE
)
AFFIL_HINT = re.compile(
    r"(@|university|institut|google|microsoft|deepmind|research|laborator|"
    r"\bllc\b|\binc\b|college|academ|\.edu|\.com)",
    re.IGNORECASE,
)


def _two_column_pages(raw: list[RawBlock], page_widths: dict[int, float]) -> dict:
    """ページごとに「二段組か」「全幅図表が途中に割り込むか（読み順リスク）」を判定。

    extract._sort_reading_order の二段組ヒューリスティックを再現し、
    全幅(>65%)の図表が、その上下に両カラムのテキストを持つ場合を at-risk とする。
    """
    by_page: dict[int, list[RawBlock]] = {}
    for b in raw:
        by_page.setdefault(b.page, []).append(b)

    two_col = 0
    at_risk_pages: list[int] = []
    for pg, blocks in by_page.items():
        pw = page_widths.get(pg, 612.0)
        mid = pw / 2.0
        left = [b for b in blocks if (b.bbox[0] + b.bbox[2]) / 2 < mid]
        right = [b for b in blocks if (b.bbox[0] + b.bbox[2]) / 2 >= mid]
        full = [b for b in blocks if (b.bbox[2] - b.bbox[0]) > pw * 0.65]
        is_two = len(left) >= 2 and len(right) >= 2 and len(full) <= len(blocks) * 0.3
        if not is_two:
            continue
        two_col += 1
        # 全幅図表が途中(上下に本文)に割り込むか
        figs = [b for b in full if b.kind in ("image", "table")]
        for f in figs:
            y0, y1 = f.bbox[1], f.bbox[3]
            txt_above = any(b.kind == "text" and b.bbox[3] < y0 for b in blocks)
            txt_below = any(b.kind == "text" and b.bbox[1] > y1 for b in blocks)
            if txt_above and txt_below:
                at_risk_pages.append(pg)
                break
    return {"two_column_pages": two_col, "reading_order_at_risk_pages": sorted(set(at_risk_pages))}


def diagnose(arxiv_id: str) -> dict:
    doc = ingest_arxiv(arxiv_id)
    pdf_path = Path(doc.metadata["pdf_path"])

    # ページ幅を取りたいので extract と別に開く
    import pymupdf  # type: ignore
    page_widths: dict[int, float] = {}
    with pymupdf.open(str(pdf_path)) as d:
        n_pages = d.page_count
        for pno, page in enumerate(d):
            page_widths[pno + 1] = float(page.rect.width)

    raw = extract_structured_blocks(pdf_path)
    blocks = segment_blocks(raw, assets_dir=None)

    types = Counter(b.type for b in blocks)

    # A. 著者誤分類: page1 の heading で所属/メールらしき行
    author_headings = [
        b.en for b in blocks
        if b.type == "heading" and b.page == 1 and AFFIL_HINT.search(b.en or "")
    ]

    # C. キャプション誤判定: caption だが「番号直後が区切り」でない＝本文中参照疑い
    intext_captions = [
        b.en[:120] for b in blocks
        if b.type == "caption" and not PROPER_CAPTION.match((b.en or "").strip())
    ]

    # D. 図ラベル付与漏れ / 孤立キャプション
    figs = [b for b in blocks if b.type in ("figure", "table")]
    unlabeled = [b.image_path or f"{b.type}@p{b.page}" for b in figs if not b.label]
    cap_labels = {b.label for b in blocks if b.type == "caption" and b.label}
    fig_labels = {b.label for b in figs if b.label}
    orphan_caps = sorted(cap_labels - fig_labels)

    # E. フロー分断: 終端なし段落の直後が heading/figure/table/caption
    dangling = 0
    dangling_samples: list[str] = []
    for i, b in enumerate(blocks[:-1]):
        if b.type != "paragraph" or not (b.en or "").strip():
            continue
        if b.en.rstrip()[-1] in _TERMINAL:
            continue
        nxt = blocks[i + 1].type
        if nxt in ("heading", "figure", "table", "caption", "title"):
            dangling += 1
            if len(dangling_samples) < 3:
                dangling_samples.append(b.en.rstrip()[-80:])

    ro = _two_column_pages(raw, page_widths)

    return {
        "id": arxiv_id,
        "title": (doc.title or "")[:70],
        "n_pages": n_pages,
        "n_blocks": len(blocks),
        "types": dict(types),
        "A_author_headings": author_headings,
        "B_reading_order": ro,
        "C_intext_captions": intext_captions,
        "D_unlabeled_figs": unlabeled,
        "D_orphan_captions": orphan_caps,
        "E_dangling_count": dangling,
        "E_dangling_samples": dangling_samples,
    }


def main() -> None:
    ids = sys.argv[1:] or DEFAULT_PAPERS
    results = []
    for i, pid in enumerate(ids):
        print(f"[{i+1}/{len(ids)}] 診断: {pid}", flush=True)
        try:
            results.append(diagnose(pid))
        except Exception as e:  # noqa: BLE001
            print(f"  ❌ {pid}: {type(e).__name__}: {e}", flush=True)
            results.append({"id": pid, "error": f"{type(e).__name__}: {e}"})
        if i < len(ids) - 1:
            time.sleep(3)  # arXiv へのレート配慮

    out = Path("logs") / f"segment_diag-{int(time.time())}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    # ── サマリ表 ──
    print("\n" + "=" * 96)
    print(f"{'id':<11}{'pg':>3}{'blk':>5} {'2col':>5}{'risk':>5} "
          f"{'authH':>6}{'icCap':>6}{'unlbl':>6}{'orph':>5}{'dangl':>6}")
    print("-" * 96)
    for r in results:
        if "error" in r:
            print(f"{r['id']:<11} ERROR {r['error'][:60]}")
            continue
        ro = r["B_reading_order"]
        print(f"{r['id']:<11}{r['n_pages']:>3}{r['n_blocks']:>5} "
              f"{ro['two_column_pages']:>5}{len(ro['reading_order_at_risk_pages']):>5}"
              f"{len(r['A_author_headings']):>6}{len(r['C_intext_captions']):>6}"
              f"{len(r['D_unlabeled_figs']):>6}{len(r['D_orphan_captions']):>5}"
              f"{r['E_dangling_count']:>6}")
    print("=" * 96)
    print("authH=著者誤分類 icCap=本文中参照のcaption化 unlbl=ラベル無し図表 "
          "orph=孤立caption dangl=フロー分断")
    print(f"\n詳細: {out}")


if __name__ == "__main__":
    main()
