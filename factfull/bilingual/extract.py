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

import re
from collections import Counter
from dataclasses import dataclass, field
from hashlib import md5
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


# ── 画像ノイズ判定 ─────────────────────────────────────────────────────────────

def _keep_image(
    w: float,
    h: float,
    page_area: float,
    *,
    min_dim: float = 40.0,
    min_area_frac: float = 0.01,
    max_area_frac: float = 0.92,
    max_aspect: float = 12.0,
) -> bool:
    """埋め込みラスタ画像を「本物の図」として残すか判定する。

    pymupdf の画像ブロックには記号・装飾・背景・罫線など本文でないラスタが
    大量に混じる（VGG 論文で偽 figure が 134 個出た）。サイズ・面積比・
    アスペクト比で足切りし、過抽出を防ぐ。反復画像（ロゴ等）の除去は
    extract_structured_blocks 側でバイトハッシュにより別途行う。
    """
    if w < min_dim or h < min_dim:
        return False                       # 小さすぎ（記号・インラインアイコン）
    area = w * h
    if page_area > 0 and not (page_area * min_area_frac <= area <= page_area * max_area_frac):
        return False                       # ページ比 1% 未満（装飾） / 92% 超（全面背景）
    short, long = sorted((w, h))
    if short > 0 and long / short > max_aspect:
        return False                       # 細長い罫線・区切り
    return True


# ── 読み順ソート（2 カラム対応） ────────────────────────────────────────────────

def _sort_reading_order(blocks: list[RawBlock], page_width: float) -> list[RawBlock]:
    """1 ページ分のブロックを読み順に並べる。

    明確なガター（中央付近に縦の空白）があれば 2 カラムとみなし、
    左カラムを上→下、続いて右カラムを上→下に並べる。
    全幅（図表や全幅段落）が段の途中に割り込む場合は、その全幅ブロックを
    境界として上→下に「帯」へ分割し、各帯内で 左カラム→右カラム を並べる
    （全幅図の前後で左右カラムが誤連結するのを防ぐ）。
    2 カラムでなければ y 昇順（同 y は x 昇順）の単一カラム扱い。
    """
    if not blocks:
        return []

    mid = page_width / 2.0

    def cx(b: RawBlock) -> float:
        return (b.bbox[0] + b.bbox[2]) / 2.0

    left = [b for b in blocks if cx(b) < mid]
    right = [b for b in blocks if cx(b) >= mid]

    # 2 カラムと判定する条件: 両カラムに十分なブロックがあり、
    # 横幅いっぱい（ガターをまたぐ）ブロックが少ない。
    full_width = [b for b in blocks if (b.bbox[2] - b.bbox[0]) > page_width * 0.65]
    two_column = (
        len(left) >= 2 and len(right) >= 2 and len(full_width) <= len(blocks) * 0.3
    )

    def y_then_x(bs: list[RawBlock]) -> list[RawBlock]:
        return sorted(bs, key=lambda b: (round(b.bbox[1], 1), b.bbox[0]))

    if not two_column:
        return y_then_x(blocks)
    if not full_width:
        return y_then_x(left) + y_then_x(right)

    # 全幅ブロックを y 昇順の境界とし、間の帯ごとに 左→右 を出力する。
    fulls = sorted(full_width, key=lambda b: b.bbox[1])
    full_ids = {id(b) for b in fulls}
    cols = [b for b in blocks if id(b) not in full_ids]

    def emit_band(band: list[RawBlock]) -> list[RawBlock]:
        return y_then_x([b for b in band if cx(b) < mid]) + \
               y_then_x([b for b in band if cx(b) >= mid])

    out: list[RawBlock] = []
    prev_y = float("-inf")
    for f in fulls:
        out += emit_band([b for b in cols if prev_y <= b.bbox[1] < f.bbox[1]])
        out.append(f)
        prev_y = f.bbox[1]
    out += emit_band([b for b in cols if b.bbox[1] >= prev_y])
    return out


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


def _cluster_rects(rects: list, gap: float):
    """y 昇順に矩形を貪欲マージし、(union_rect, n_paths) のクラスタ列を返す。"""
    clusters: list[list] = []
    for r in sorted(rects, key=lambda r: r.y0):
        if clusters and r.y0 <= clusters[-1][0].y1 + gap:
            clusters[-1][0] |= r
            clusters[-1][1] += 1
        else:
            clusters.append([r.__class__(r), 1])
    return clusters


_FIG_CAPTION = re.compile(r"^(figure|fig\.?)\s*\d+\s*[:.]", re.IGNORECASE)
_TABLE_CAPTION = re.compile(r"^table\s*\d+\s*[:.]", re.IGNORECASE)


def _block_text(blk: dict) -> str:
    """テキストブロックの連結文字列。"""
    return "".join(
        s.get("text", "")
        for ln in blk.get("lines", []) for s in ln.get("spans", [])
    ).strip()


def _figure_caption_boxes(page_dict: dict) -> list[tuple]:
    """ページ内の図キャプション（'Figure N:' / 'Figure N.'）の bbox を返す。"""
    out: list[tuple] = []
    for blk in page_dict.get("blocks", []):
        if blk.get("type", 0) == 0 and _FIG_CAPTION.match(_block_text(blk)):
            out.append(tuple(round(float(v), 1) for v in blk.get("bbox", (0, 0, 0, 0))))
    return out


def _extract_text_tables(
    page, page_dict: dict, page_index: int, pymupdf, exclude_rects: list,
) -> list[RawBlock]:
    """罫線の無いテキスト整列表（find_tables が検出できない）を捕捉する。

    'Table N' キャプション直下に並ぶ「短いセル（1 行・少数文字）が複数列に整列
    した密集領域」を表とみなし、その領域をレンダリングして表画像にする。これに
    より、バラバラのセルが段落として文字化け翻訳されるのを防ぎ、孤立しがちな
    Table キャプションに表本体を与える。VGG（罫線ゼロのテキスト表）が典型例。
    """
    text_blocks = [b for b in page_dict.get("blocks", []) if b.get("type", 0) == 0]
    cap_blocks = [b for b in text_blocks if _TABLE_CAPTION.match(_block_text(b))]
    if not cap_blocks:
        return []

    pw = float(page.rect.width)

    def is_prose(b: dict) -> bool:
        t = _block_text(b)
        w = b["bbox"][2] - b["bbox"][0]
        return len(t) > 120 and w > pw * 0.35

    out: list[RawBlock] = []
    for cap in cap_blocks:
        cap_box = cap["bbox"]
        # キャプション直下のセル候補を y 昇順に収集
        below = sorted(
            (b for b in text_blocks
             if b is not cap and b["bbox"][1] >= cap_box[3] - 2),
            key=lambda b: b["bbox"][1],
        )
        cells: list[dict] = []
        prev_y = cap_box[3]
        for b in below:
            if _TABLE_CAPTION.match(_block_text(b)) or _FIG_CAPTION.match(_block_text(b)):
                break
            if is_prose(b):
                break
            if cells and b["bbox"][1] - prev_y > 34:   # 大きな縦ギャップ＝表の外
                break
            if len(_block_text(b)) > 60:               # 長い行＝セルでない
                break
            cells.append(b)
            prev_y = b["bbox"][3]

        # 表らしさ: 十分なセル数 ＋ 複数列（distinct x が 3 以上）
        if len(cells) < 8:
            continue
        col_buckets = {round(b["bbox"][0] / 20) for b in cells}
        if len(col_buckets) < 3:
            continue

        x0 = min(b["bbox"][0] for b in cells)
        y0 = min(b["bbox"][1] for b in cells)
        x1 = max(b["bbox"][2] for b in cells)
        y1 = max(b["bbox"][3] for b in cells)
        box = tuple(round(v, 1) for v in (x0, y0, x1, y1))
        if any(_rect_overlaps(box, ex, tol=6.0) for ex in exclude_rects):
            continue
        try:
            pix = page.get_pixmap(clip=pymupdf.Rect(box), dpi=150)
        except Exception:
            continue
        out.append(RawBlock(
            kind="table", page=page_index + 1, bbox=box,
            image_bytes=pix.tobytes("png"), image_ext="png",
        ))
    return out


def _anchored_by_caption(box: tuple, caption_boxes: list[tuple]) -> bool:
    """図領域 box に隣接（直上 or 直下、横方向に重なる）する図キャプションがあるか。"""
    for cb in caption_boxes:
        h_overlap = not (cb[2] < box[0] - 10 or cb[0] > box[2] + 10)
        if not h_overlap:
            continue
        below = -6.0 <= cb[1] - box[3] <= 48.0     # キャプションが図の直下
        above = -6.0 <= box[1] - cb[3] <= 48.0     # キャプションが図の直上
        if below or above:
            return True
    return False


def _extract_vector_figures(
    page, page_index: int, pymupdf, exclude_rects: list, caption_boxes: list[tuple],
) -> list[RawBlock]:
    """ベクター描画（get_drawings）の密集領域のうち、図キャプションが隣接する
    ものだけを図として描画・抽出する。

    ResNet 等の図は埋め込みラスタでなくベクターグラフィックスで pymupdf の
    画像ブロックに現れない（figure=0）。描画パスをカラム別に縦クラスタ化し、
    「直上/直下に Figure N キャプションがある」クラスタのみ採用することで、
    数式・装飾・罫線などのベクターを図と誤認しない（精度優先）。
    既存の表/画像領域と重なるクラスタは二重取得を避けて除外する。
    """
    if not caption_boxes:
        return []
    try:
        drawings = page.get_drawings()
    except Exception:
        return []
    rects = [
        d["rect"] for d in drawings
        if d.get("rect") is not None and d["rect"].width > 1 and d["rect"].height > 1
    ]
    if len(rects) < 4:
        return []

    pw = float(page.rect.width)
    ph = float(page.rect.height)
    page_area = pw * ph
    mid = pw / 2.0

    # カラム別に分離（カラム跨ぎの誤マージを防ぐ）
    full = [r for r in rects if r.width > pw * 0.55]
    left = [r for r in rects if r.width <= pw * 0.55 and (r.x0 + r.x1) / 2 < mid]
    right = [r for r in rects if r.width <= pw * 0.55 and (r.x0 + r.x1) / 2 >= mid]

    boxes: list = []
    for group in (full, left, right):
        for rect, n_paths in _cluster_rects(group, gap=16.0):
            if n_paths < 4:
                continue
            if rect.width < pw * 0.16 or rect.height < 36:
                continue
            if rect.height > ph * 0.85:               # 全面背景は除外
                continue
            if rect.width * rect.height < page_area * 0.012:
                continue
            box = tuple(round(v, 1) for v in (rect.x0, rect.y0, rect.x1, rect.y1))
            if not _anchored_by_caption(box, caption_boxes):
                continue                              # キャプション無し＝図でない可能性大
            if any(_rect_overlaps(box, ex, tol=6.0) for ex in exclude_rects):
                continue                              # 既存の表/画像と重複
            boxes.append(box)

    # 1 つの図が内部の空白で複数クラスタに割れることがあるので、近接領域を結合する
    boxes = _merge_boxes(boxes, gap=22.0)

    out: list[RawBlock] = []
    for box in boxes:
        try:
            pix = page.get_pixmap(clip=pymupdf.Rect(box), dpi=150)
        except Exception:
            continue
        out.append(RawBlock(
            kind="image", page=page_index + 1, bbox=box,
            image_bytes=pix.tobytes("png"), image_ext="png",
        ))
    return out


def _merge_boxes(boxes: list[tuple], gap: float) -> list[tuple]:
    """重なる / gap 以内で近接する矩形を反復的に結合する。"""
    boxes = list(boxes)
    merged = True
    while merged:
        merged = False
        out: list[tuple] = []
        for b in boxes:
            for i, o in enumerate(out):
                near = not (b[2] < o[0] - gap or b[0] > o[2] + gap or
                            b[3] < o[1] - gap or b[1] > o[3] + gap)
                if near:
                    out[i] = (min(b[0], o[0]), min(b[1], o[1]),
                              max(b[2], o[2]), max(b[3], o[3]))
                    merged = True
                    break
            else:
                out.append(b)
        boxes = out
    return boxes


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
            page_area = page_width * float(page.rect.height)
            d = page.get_text("dict")

            # 残す埋め込みラスタ画像の領域（ベクター図/テキスト表がこれと重複
            # しないよう先に集める）。ノイズ画像は除外済みの基準で判定する。
            raster_rects = []
            for blk in d.get("blocks", []):
                if blk.get("type", 0) != 1 or not blk.get("image"):
                    continue
                bb = tuple(round(float(v), 1) for v in blk.get("bbox", (0, 0, 0, 0)))
                if _keep_image(bb[2] - bb[0], bb[3] - bb[1], page_area):
                    raster_rects.append(bb)
            caption_boxes = _figure_caption_boxes(d)
            vec_figs = _extract_vector_figures(
                page, pno, pymupdf, raster_rects, caption_boxes
            )
            vec_rects = [f.bbox for f in vec_figs]

            # ベクター図と重なる find_tables 検出（図中の格子の誤検出）は捨てる
            tables = [
                t for t in _extract_tables(page, pno, pymupdf)
                if not any(_rect_overlaps(t.bbox, vr, tol=6.0) for vr in vec_rects)
            ]
            # 罫線無しテキスト表（find_tables が拾えない）をキャプション基準で捕捉
            text_tables = _extract_text_tables(
                page, d, pno, pymupdf,
                raster_rects + vec_rects + [t.bbox for t in tables],
            )
            tables += text_tables
            table_rects = [t.bbox for t in tables]

            page_blocks: list[RawBlock] = list(tables) + list(vec_figs)

            for blk in d.get("blocks", []):
                bbox = tuple(round(float(v), 1) for v in blk.get("bbox", (0, 0, 0, 0)))
                btype = blk.get("type", 0)

                if btype == 1:  # 画像ブロック
                    img = blk.get("image")
                    if not img:
                        continue
                    if not _keep_image(bbox[2] - bbox[0], bbox[3] - bbox[1], page_area):
                        continue            # 装飾・記号・罫線などのノイズ画像を除外
                    page_blocks.append(RawBlock(
                        kind="image",
                        page=pno + 1,
                        bbox=bbox,
                        image_bytes=img,
                        image_ext=str(blk.get("ext", "png")),
                    ))
                    continue

                # テキストブロック: 表/ベクター図 領域内のものは図表に取り込み済みなので除外
                if any(_rect_overlaps(bbox, tr) for tr in table_rects + vec_rects):
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

    all_blocks = _drop_repeated_images(all_blocks)

    if assets_dir is not None:
        _dump_images(all_blocks, Path(assets_dir))

    return all_blocks


def _drop_repeated_images(blocks: list[RawBlock], min_repeat: int = 3) -> list[RawBlock]:
    """文書全体で同一バイト列が反復する画像（ロゴ・透かし等）を除去する。"""
    counts: Counter[str] = Counter(
        md5(b.image_bytes).hexdigest()
        for b in blocks if b.kind == "image" and b.image_bytes
    )
    repeated = {h for h, c in counts.items() if c >= min_repeat}
    if not repeated:
        return blocks
    return [
        b for b in blocks
        if not (b.kind == "image" and b.image_bytes
                and md5(b.image_bytes).hexdigest() in repeated)
    ]


def _dump_images(blocks: list[RawBlock], assets_dir: Path) -> None:
    """image / table ブロックの画像バイト列をファイルに書き出し、
    image_bytes をそのまま保持しつつ、後段が参照できるようにする。

    （実際の相対パス割り当ては segment 側で行う。ここではバイト列の確定のみ）。
    """
    assets_dir.mkdir(parents=True, exist_ok=True)
