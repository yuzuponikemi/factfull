"""
factfull/bilingual/segment.py
=============================
RawBlock 列 → 整形済み Block 列。

責務:
  - ハイフン分割の結合・段落内改行の正規化
  - 見出し検出（フォントサイズ比 + セクション番号正規表現）とレベル付与
  - ヘッダ/フッタ（ページ番号・走り文）の除去
  - 図キャプション / 参考文献の分離
  - 図（image）・表（table）ブロックの位置保持と画像書き出し・ラベル付与
  - セクション階層（section_path）と安定 ID の付与

各ルールは純粋関数として切り出し、PDF/pymupdf なしで単体テスト可能にしている。
"""
from __future__ import annotations

import re
from pathlib import Path

from factfull.bilingual.extract import RawBlock
from factfull.bilingual.types import Block

# "3" / "3.1" / "4.2.1" + 見出しテキスト
SECTION_NUM = re.compile(r"^(\d+(?:\.\d+)*)\.?\s+(.+)$")
# 図表キャプション
CAPTION = re.compile(r"^(Figure|Fig\.?|Table|Algorithm|Alg\.?)\s*(\d+)", re.IGNORECASE)
# 参考文献セクション見出し
REFERENCES_HEAD = re.compile(r"^(references|bibliography|参考文献)\s*$", re.IGNORECASE)
ABSTRACT_HEAD = re.compile(r"^abstract\s*$", re.IGNORECASE)
# 著者 / 所属行（タイトルページで見出し誤検出されやすい）。メール・所属語を手がかりに分離。
AFFILIATION = re.compile(
    r"(@|\.edu|\.com|\buniversit|\binstitut|\bgoogle\b|\bmicrosoft\b|\bdeepmind\b|"
    r"\bresearch\b|\blaborator|\bcollege\b|\bacadem|\binc\b|\bllc\b)",
    re.IGNORECASE,
)

# 段落の終端とみなす文字
_TERMINAL = ".?!:;。．！？”)」】"


# ── テキスト整形 ───────────────────────────────────────────────────────────────

def _join_hyphenation(text: str) -> str:
    """行末ハイフン分割を結合し、段落内の単一改行を空白に変換する。

    'inter-\\nnational' -> 'international'
    段落内の単一 '\\n' -> ' '、二重以上の '\\n'（段落区切り）は保持。
    """
    # 行末ハイフン + 改行 + 次行頭文字 を結合
    text = re.sub(r"(\w)[­-]\n(\w)", r"\1\2", text)
    # 二重改行を一時マーカーに退避
    text = text.replace("\n\n", "\x00")
    # 残る単一改行は空白へ
    text = re.sub(r"\s*\n\s*", " ", text)
    text = text.replace("\x00", "\n\n")
    return re.sub(r"[ \t]{2,}", " ", text).strip()


def _estimate_body_size(raw: list[RawBlock]) -> float:
    """本文フォントサイズ（テキストブロックのサイズ中央値）を推定する。"""
    sizes = sorted(b.font_size for b in raw if b.kind == "text" and b.font_size > 0)
    if not sizes:
        return 10.0
    return sizes[len(sizes) // 2]


# ── 見出し / キャプション判定 ──────────────────────────────────────────────────

def _is_heading(rb: RawBlock, body_size: float, ratio: float) -> tuple[bool, int | None]:
    """テキスト RawBlock が見出しか判定し、(bool, level) を返す。

    判定:
      1. セクション番号で始まる（"3.1 Method"）→ True、level = 番号の深さ
      2. フォントサイズが本文 * ratio より大きい → True
      3. 太字かつ短行（<= 12 語）かつ終端句読点なし → True
    """
    text = rb.text.strip()
    if not text or "\n\n" in text:
        return False, None

    m = SECTION_NUM.match(text)
    if m and len(text) <= 120:
        level = m.group(1).count(".") + 1
        return True, level

    n_words = len(text.split())
    short_no_period = n_words <= 12 and text[-1] not in _TERMINAL

    if rb.font_size >= body_size * ratio and short_no_period:
        # フォントサイズで level を 1/2 に振り分け
        level = 1 if rb.font_size >= body_size * 1.5 else 2
        return True, level
    if rb.bold and short_no_period and n_words <= 10:
        return True, 2
    return False, None


def _caption_label(text: str) -> str:
    """'Figure 1: ...' → 'Figure 1'。キャプションでなければ空文字。

    番号の直後に区切り（: . — -）があれば即キャプション。区切りが無い場合は
    短い見出し的な行のみキャプションとし、'Table 2 summarizes our results …'
    のような本文中参照（長い平文）を誤ってキャプション化しない。
    """
    t = text.strip()
    m = CAPTION.match(t)
    if not m:
        return ""
    rest = t[m.end():]
    has_sep = bool(re.match(r"\s*[:.—\-]", rest))
    if not has_sep and len(t.split()) > 8:
        return ""
    kind = "Table" if m.group(1).lower().startswith("table") else "Figure"
    return f"{kind} {m.group(2)}"


def _looks_like_header_footer(text: str) -> bool:
    """ページ番号・arXiv ID・走り文など、本文でない短行か。"""
    t = text.strip()
    if not t:
        return True
    if re.fullmatch(r"\d{1,4}", t):                       # ページ番号
        return True
    if re.fullmatch(r"(page\s*)?\d+\s*(/|of)\s*\d+", t, re.IGNORECASE):
        return True
    if re.match(r"^arxiv:\d{4}\.\d{4,5}", t, re.IGNORECASE):
        return True
    return False


def _strip_headers_footers(raw: list[RawBlock]) -> list[RawBlock]:
    """複数ページで同一テキストが反復するブロック（走り文）と、
    ページ番号類を除去する。"""
    # 反復テキストの集計（小文字・空白正規化）
    counts: dict[str, int] = {}
    for b in raw:
        if b.kind != "text":
            continue
        key = re.sub(r"\s+", " ", b.text.strip().lower())
        if len(key) <= 80:
            counts[key] = counts.get(key, 0) + 1
    pages = len({b.page for b in raw}) or 1
    repeated = {k for k, c in counts.items() if c >= max(3, pages // 2) and c >= 3}

    out: list[RawBlock] = []
    for b in raw:
        if b.kind == "text":
            if _looks_like_header_footer(b.text):
                continue
            key = re.sub(r"\s+", " ", b.text.strip().lower())
            if key in repeated:
                continue
        out.append(b)
    return out


# ── メイン ─────────────────────────────────────────────────────────────────────

def segment_blocks(
    raw: list[RawBlock],
    *,
    skip_references: bool = True,
    skip_captions: bool = False,
    heading_size_ratio: float = 1.08,
    min_paragraph_chars: int = 3,
    assets_dir: Path | None = None,
) -> list[Block]:
    """RawBlock 列を整形済み Block 列に変換する（en のみ充填、ja は空）。"""
    raw = _strip_headers_footers(raw)
    body_size = _estimate_body_size(raw)

    blocks: list[Block] = []
    in_references = False
    in_abstract = False
    img_seq = 0

    for rb in raw:
        # ── 図 / 表 ────────────────────────────────────────────────────────
        if rb.kind in ("image", "table"):
            if rb.kind == "image" and (rb.bbox[2] - rb.bbox[0]) < 24:
                continue  # 装飾的な極小画像は無視
            img_seq += 1
            image_path = _save_image(rb, img_seq, assets_dir)
            blocks.append(Block(
                id="", type=("table" if rb.kind == "table" else "figure"),
                page=rb.page, bbox=list(rb.bbox), image_path=image_path,
            ))
            continue

        text = _join_hyphenation(rb.text)
        if not text:
            continue

        # ── 参考文献セクション ───────────────────────────────────────────────
        if REFERENCES_HEAD.match(text):
            in_references = True
            in_abstract = False
            blocks.append(Block(id="", type="heading", en=text, level=1, page=rb.page))
            continue
        if in_references:
            if skip_references:
                continue
            blocks.append(Block(
                id="", type="reference", en=text, page=rb.page, skip_translate=True,
            ))
            continue

        # ── 見出し ──────────────────────────────────────────────────────────
        is_head, level = _is_heading(rb, body_size, heading_size_ratio)

        # 著者 / 所属行: タイトルページで見出し誤検出されがちなので分離し、
        # 翻訳対象外の段落として残す（section_path の汚染も防ぐ）。
        if (
            rb.page == 1
            and (is_head or "@" in text)
            and AFFILIATION.search(text)
            and not SECTION_NUM.match(text)
            and not ABSTRACT_HEAD.match(text)
        ):
            blocks.append(Block(
                id="", type="paragraph", en=text, page=rb.page, skip_translate=True,
            ))
            continue

        if is_head:
            in_abstract = bool(ABSTRACT_HEAD.match(text))
            blocks.append(Block(
                id="", type="heading", en=text, level=level, page=rb.page,
            ))
            continue

        # ── キャプション ────────────────────────────────────────────────────
        label = _caption_label(text)
        if label:
            if skip_captions:
                continue
            blocks.append(Block(
                id="", type="caption", en=text, label=label, page=rb.page,
            ))
            continue

        # ── 本文 / アブストラクト ────────────────────────────────────────────
        if len(text) < min_paragraph_chars:
            continue
        btype = "abstract" if in_abstract else "paragraph"
        blocks.append(Block(id="", type=btype, en=text, page=rb.page))

    blocks = _merge_paragraphs(blocks)
    _attach_figure_labels(blocks)
    _build_section_path(blocks)
    _assign_ids(blocks)
    return blocks


def _merge_paragraphs(blocks: list[Block]) -> list[Block]:
    """終端句読点で終わらない連続段落ブロックを 1 段落に結合する
    （pymupdf によるカラム/ブロック過分割を修復）。"""
    out: list[Block] = []
    for b in blocks:
        if (
            b.type == "paragraph" and out and out[-1].type == "paragraph"
            and out[-1].en and out[-1].en.rstrip()[-1] not in _TERMINAL
        ):
            out[-1].en = out[-1].en.rstrip() + " " + b.en.lstrip()
        else:
            out.append(b)
    return out


def _attach_figure_labels(blocks: list[Block]) -> None:
    """figure/table ブロックに、近傍のキャプションからラベルを付与する。

    キャプションは図の直後（表は直前/直後）に置かれることが多いので、
    同種ラベルを持つ最近傍のキャプションを探して図表ブロックへコピーする。
    """
    used: set[int] = set()  # 1 キャプションが複数図表に多重付与されるのを防ぐ
    for i, b in enumerate(blocks):
        if b.type not in ("figure", "table") or b.label:
            continue
        want = "Table" if b.type == "table" else "Figure"
        # 近い順（±1, ±2, …, ±5）に同一ページの同種キャプションを探す
        for off in range(1, 6):
            hit = False
            for j in (i + off, i - off):
                if not (0 <= j < len(blocks)) or j in used:
                    continue
                cap = blocks[j]
                if (cap.type == "caption" and cap.label.startswith(want)
                        and cap.page == b.page):
                    b.label = cap.label
                    used.add(j)
                    hit = True
                    break
            if hit:
                break


def _build_section_path(blocks: list[Block]) -> None:
    """見出しスタックを辿り、各ブロックに section_path（祖先見出し EN）を付与する。"""
    stack: list[tuple[int, str]] = []  # (level, heading_en)
    for b in blocks:
        if b.type == "heading":
            lvl = b.level or 1
            while stack and stack[-1][0] >= lvl:
                stack.pop()
            b.section_path = [h for _, h in stack]
            stack.append((lvl, b.en))
        else:
            b.section_path = [h for _, h in stack]


def _assign_ids(blocks: list[Block]) -> None:
    """読み順に b0001… のゼロ詰め連番 ID を付与する。"""
    for i, b in enumerate(blocks, 1):
        b.id = f"b{i:04d}"


def _save_image(rb: RawBlock, seq: int, assets_dir: Path | None) -> str:
    """図表画像を assets_dir に書き出し、out_dir からの相対パスを返す。

    assets_dir が None の場合は書き出さず空文字を返す（テスト用）。
    """
    if assets_dir is None or not rb.image_bytes:
        return ""
    assets_dir.mkdir(parents=True, exist_ok=True)
    ext = rb.image_ext or "png"
    fname = f"{rb.kind}_p{rb.page}_{seq:03d}.{ext}"
    (assets_dir / fname).write_bytes(rb.image_bytes)
    return f"assets/{fname}"
