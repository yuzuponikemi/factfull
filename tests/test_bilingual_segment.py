"""factfull/bilingual/segment.py のテスト（PDF/pymupdf/LLM 不要）。"""
from factfull.bilingual.extract import RawBlock
from factfull.bilingual.segment import (
    SECTION_NUM,
    _caption_label,
    _is_heading,
    _join_hyphenation,
    _looks_like_header_footer,
    _strip_headers_footers,
    segment_blocks,
)


def _text(text, *, size=10.0, bold=False, page=1, y=100.0):
    return RawBlock(kind="text", page=page, bbox=(50.0, y, 300.0, y + 12), text=text,
                    font_size=size, bold=bold)


class TestJoinHyphenation:
    def test_hyphen_linebreak_joined(self):
        assert _join_hyphenation("inter-\nnational") == "international"

    def test_single_newline_to_space(self):
        assert _join_hyphenation("hello\nworld") == "hello world"

    def test_paragraph_break_preserved(self):
        assert _join_hyphenation("a\n\nb") == "a\n\nb"


class TestHeadingDetection:
    def test_section_number_level(self):
        assert SECTION_NUM.match("3 Introduction").group(1) == "3"
        assert SECTION_NUM.match("4.2.1 Setup").group(1) == "4.2.1"

    def test_numbered_heading(self):
        ok, level = _is_heading(_text("3.1 Method"), body_size=10.0, ratio=1.08)
        assert ok and level == 2

    def test_large_font_heading(self):
        ok, level = _is_heading(_text("Introduction", size=16.0), body_size=10.0, ratio=1.08)
        assert ok and level == 1

    def test_bold_short_line_heading(self):
        ok, _ = _is_heading(_text("Related Work", bold=True), body_size=10.0, ratio=1.08)
        assert ok

    def test_normal_sentence_is_not_heading(self):
        rb = _text("This is a normal body sentence that ends with a period.")
        ok, _ = _is_heading(rb, body_size=10.0, ratio=1.08)
        assert not ok


class TestCaption:
    def test_figure_caption(self):
        assert _caption_label("Figure 1: A diagram.") == "Figure 1"

    def test_table_caption(self):
        assert _caption_label("Table 2 Results") == "Table 2"

    def test_not_caption(self):
        assert _caption_label("Figures are useful.") == ""


class TestHeaderFooter:
    def test_page_number_is_header_footer(self):
        assert _looks_like_header_footer("12")

    def test_arxiv_id_is_header_footer(self):
        assert _looks_like_header_footer("arXiv:2403.11996v1")

    def test_repeated_running_head_stripped(self):
        # 走り文は全ページで同一、本文はページごとに異なる（実際の PDF と同様）
        raw = [_text("A Running Head", page=p, y=20.0) for p in range(1, 5)]
        raw += [_text(f"Body unique to page {p}.", page=p, y=200.0) for p in range(1, 5)]
        kept = _strip_headers_footers(raw)
        texts = {b.text for b in kept}
        assert "A Running Head" not in texts
        assert "Body unique to page 1." in texts


class TestSegmentBlocks:
    def test_section_path_and_ids(self):
        raw = [
            _text("1 Introduction", size=16.0, y=50),
            _text("Body of intro section.", y=70),
            _text("1.1 Motivation", size=13.0, y=90),
            _text("Body of motivation.", y=110),
        ]
        blocks = segment_blocks(raw)
        assert [b.id for b in blocks] == ["b0001", "b0002", "b0003", "b0004"]
        motivation_body = blocks[3]
        assert motivation_body.section_path == ["1 Introduction", "1.1 Motivation"]

    def test_references_dropped_by_default(self):
        raw = [
            _text("Body paragraph.", y=50),
            _text("References", size=14.0, y=70),
            _text("[1] Some cited work, 2020.", y=90),
        ]
        blocks = segment_blocks(raw, skip_references=True)
        types = [b.type for b in blocks]
        assert "reference" not in types
        assert any("[1]" in b.en for b in blocks) is False

    def test_references_kept_untranslated(self):
        raw = [
            _text("References", size=14.0, y=70),
            _text("[1] Some cited work, 2020.", y=90),
        ]
        blocks = segment_blocks(raw, skip_references=False)
        refs = [b for b in blocks if b.type == "reference"]
        assert len(refs) == 1 and refs[0].skip_translate is True

    def test_references_heading_and_trailing_figures_dropped(self):
        # skip_references 時は参考文献見出しも、参考文献以降の図表も出さない
        raw = [
            _text("Body paragraph.", y=50),
            _text("References", size=14.0, y=70),
            _text("[1] Some cited work, 2020.", y=90),
            RawBlock(kind="image", page=2, bbox=(50, 100, 200, 200), image_bytes=b"x"),
        ]
        blocks = segment_blocks(raw, skip_references=True)
        assert not any(b.type == "figure" for b in blocks)
        assert not any(b.type == "heading" and "References" in b.en for b in blocks)

    def test_figure_block_and_caption_label(self):
        raw = [
            RawBlock(kind="image", page=1, bbox=(50, 100, 200, 200), image_bytes=b"x"),
            _text("Figure 1: A red box.", y=210),
        ]
        blocks = segment_blocks(raw)
        fig = [b for b in blocks if b.type == "figure"][0]
        assert fig.label == "Figure 1"
        assert fig.page == 1 and fig.bbox == [50, 100, 200, 200]

    def test_paragraph_merge(self):
        raw = [
            _text("This sentence is split across", y=50),
            _text("two blocks without a period", y=64),
            _text("and finally ends here.", y=78),
        ]
        blocks = segment_blocks(raw)
        paras = [b for b in blocks if b.type == "paragraph"]
        assert len(paras) == 1
        assert paras[0].en.endswith("ends here.")
