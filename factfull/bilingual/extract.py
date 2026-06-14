"""
factfull/bilingual/extract.py
=============================
構造を保った PDF 抽出。pymupdf の 'dict' 抽出を使い、テキストブロックを
フォントサイズ・太字つきで取り出す。さらに図（埋め込みラスタ画像）と表
（find_tables）を矩形位置つきで取り出し、読み順に並べる。

平文の SourceDoc.text ではなく PDF パスから直接抽出するのは、見出し判定に
必要なフォントサイズ手がかりが平文化で失われているため。

公開関数:
    extract_structured_blocks(pdf_path, assets_dir=None) -> list[RawBlock]
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# pymupdf span flags: bit 4 (=16) が太字（superscript=1, italic=2, serifed=4, ...）
_BOLD_FLAG = 1 << 4


@dataclass
class RawBlock:
    """セグメンテーション前の生ブロック。"""
    kind: str                      # "text" | "image" | "table"
    page: int                      # 1 始まり
    bbox: tuple[float, float, float, float]
    text: str = ""                 # kind == "text"
    font_size: float = 0.0         # 支配的スパンのサイズ
    bold: bool = False             # 太字スパンを含むか
    image_bytes: bytes | None = None  # kind in ("image", "table")
    image_ext: str = "png"

    @property
    def y(self) -> float:
        return self.bbox[1]

    @property
    def x0(self) -> float:
        return self.bbox[0]


# ── フォント解析 ───────────────────────────────────────────────────────────────

def _dominant_font(text_block: dict) -> tuple[float, bool, str]:
    """テキストブロックの (支配的フォントサイズ, 太字有無, 連結テキスト) を返す。

    サイズは「文字数で重み付けした最頻サイズ」。太字は span flags の太字ビット、
    またはフォント名に bold/black を含むかで判定する。
    """
    size_weight: dict[float, int] = {}
    bold = False
    parts: list[str] = []
    for line in text_block.get("lines", []):
        for span in line.get("spans", []):
            t = span.get("text", "")
            parts.append(t)
            size = round(float(span.get("size", 0.0)), 1)
            size_weight[size] = size_weight.get(size, 0) + max(len(t), 1)
            flags = int(span.get("flags", 0))
            fname = str(span.get("font", "")).lower()
            if flags & _BOLD_FLAG or "bold" in fname or "black" in fname:
                bold = True
        parts.append("\n")
    text = "".join(parts).strip("\n")
    dom_size = max(size_weight, key=size_weight.get) if size_weight else 0.0
    return dom_size, bold, text


# ── 読み順ソート（2 カラム対応） ────────────────────────────────────────────────

def _sort_reading_order(blocks: list[RawBlock], page_width: float) -> list[RawBlock]:
    """1 ページ分のブロックを読み順に並べる。

    明確なガター（中央付近に縦の空白）があれば 2 カラムとみなし、
    左カラムを上→下、続いて右カラムを上→下に並べる。
    そうでなければ y 昇順（同 y は x 昇順）の単一カラム扱い。
    """
    if not blocks:
        return []

    mid = page_width / 2.0
    left = [b for b in blocks if (b.bbox[0] + b.bbox[2]) / 2.0 < mid]
    right = [b for b in blocks if (b.bbox[0] + b.bbox[2]) / 2.0 >= mid]

    # 2 カラムと判定する条件: 両カラムに十分なブロックがあり、
    # 横幅いっぱい（ガターをまたぐ）ブロックが少ない。
    full_width = [b for b in blocks if (b.bbox[2] - b.bbox[0]) > page_width * 0.65]
    two_column = (
        len(left) >= 2 and len(right) >= 2 and len(full_width) <= len(blocks) * 0.3
    )

    def y_then_x(bs: list[RawBlock]) -> list[RawBlock]:
        return sorted(bs, key=lambda b: (round(b.bbox[1], 1), b.bbox[0]))

    if two_column:
        return y_then_x(left) + y_then_x(right)
    return y_then_x(blocks)


# ── 図表抽出 ───────────────────────────────────────────────────────────────────

def _rect_overlaps(a, b, tol: float = 2.0) -> bool:
    """2 矩形が重なるか（表領域内のテキスト除外などに使う）。"""
    return not (a[2] <= b[0] + tol or a[0] >= b[2] - tol or
                a[3] <= b[1] + tol or a[1] >= b[3] - tol)


def _extract_tables(page, page_index: int, pymupdf) -> list[RawBlock]:
    """find_tables で表を検出し、各表領域を PNG に描画した RawBlock を返す。

    検出に失敗しても黙って空リストを返す（堅牢性優先）。
    """
    out: list[RawBlock] = []
    try:
        finder = page.find_tables()
    except Exception:
        return out
    for tbl in getattr(finder, "tables", []):
        try:
            rect = pymupdf.Rect(tbl.bbox)
            if rect.is_empty or rect.width < 20 or rect.height < 20:
                continue
            pix = page.get_pixmap(clip=rect, dpi=150)
            out.append(RawBlock(
                kind="table",
                page=page_index + 1,
                bbox=tuple(round(v, 1) for v in tbl.bbox),
                image_bytes=pix.tobytes("png"),
                image_ext="png",
            ))
        except Exception:
            continue
    return out


# ── 公開 API ───────────────────────────────────────────────────────────────────

def extract_structured_blocks(
    pdf_path: Path,
    assets_dir: Path | None = None,
) -> list[RawBlock]:
    """PDF を読み順の RawBlock 列に変換する。

    Args:
        pdf_path: PDF ファイルパス
        assets_dir: 図表画像を保存する場合の保存先（省略時は image_bytes を保持のみ）

    Returns:
        RawBlock のリスト（全ページを読み順に連結）
    """
    import pymupdf  # type: ignore

    pdf_path = Path(pdf_path)
    all_blocks: list[RawBlock] = []

    with pymupdf.open(str(pdf_path)) as doc:
        for pno, page in enumerate(doc):
            page_width = float(page.rect.width)
            d = page.get_text("dict")

            tables = _extract_tables(page, pno, pymupdf)
            table_rects = [t.bbox for t in tables]

            page_blocks: list[RawBlock] = list(tables)

            for blk in d.get("blocks", []):
                bbox = tuple(round(float(v), 1) for v in blk.get("bbox", (0, 0, 0, 0)))
                btype = blk.get("type", 0)

                if btype == 1:  # 画像ブロック
                    img = blk.get("image")
                    if not img:
                        continue
                    page_blocks.append(RawBlock(
                        kind="image",
                        page=pno + 1,
                        bbox=bbox,
                        image_bytes=img,
                        image_ext=str(blk.get("ext", "png")),
                    ))
                    continue

                # テキストブロック: 表領域に含まれるものは表に取り込み済みなので除外
                if any(_rect_overlaps(bbox, tr) for tr in table_rects):
                    continue
                size, bold, text = _dominant_font(blk)
                if not text.strip():
                    continue
                page_blocks.append(RawBlock(
                    kind="text",
                    page=pno + 1,
                    bbox=bbox,
                    text=text,
                    font_size=size,
                    bold=bold,
                ))

            all_blocks.extend(_sort_reading_order(page_blocks, page_width))

    if assets_dir is not None:
        _dump_images(all_blocks, Path(assets_dir))

    return all_blocks


def _dump_images(blocks: list[RawBlock], assets_dir: Path) -> None:
    """image / table ブロックの画像バイト列をファイルに書き出し、
    image_bytes をそのまま保持しつつ、後段が参照できるようにする。

    （実際の相対パス割り当ては segment 側で行う。ここではバイト列の確定のみ）。
    """
    assets_dir.mkdir(parents=True, exist_ok=True)
