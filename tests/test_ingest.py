"""factfull/ingest/{book,web,paper}.py のテスト（ネットワーク・LLM 不要）。"""
import json
import tempfile
from pathlib import Path

import pytest
from factfull.core.types import SourceDoc


# ── ingest/book ──────────────────────────────────────────────────────────────

class TestIngestBook:
    def test_local_file(self, tmp_path):
        from factfull.ingest.book import ingest_book
        f = tmp_path / "book.txt"
        f.write_text("Chapter 1\n" + "word " * 300 + "\nChapter 2\n" + "word " * 300, encoding="utf-8")

        doc = ingest_book(f, title="Test Book", author="Test Author")

        assert doc.source_type == "book"
        assert doc.title == "Test Book"
        assert doc.metadata["author"] == "Test Author"
        assert len(doc.chunks) > 0
        assert len(doc.text) > 0

    def test_title_fallback_to_filename(self, tmp_path):
        from factfull.ingest.book import ingest_book
        f = tmp_path / "my_novel.txt"
        f.write_text("Some content.", encoding="utf-8")

        doc = ingest_book(f)  # title 省略
        assert doc.title == "my_novel"

    def test_source_doc_is_valid(self, tmp_path):
        from factfull.ingest.book import ingest_book
        f = tmp_path / "test.txt"
        f.write_text("Hello world " * 100, encoding="utf-8")
        doc = ingest_book(f, title="Hello")

        # SourceDoc として JSON 往復できること
        restored = SourceDoc.from_dict(doc.to_dict())
        assert restored.title == "Hello"
        assert restored.source_type == "book"

    def test_gutenberg_strip(self):
        from factfull.ingest.book import _strip_gutenberg
        raw = (
            "Some header text\n"
            "*** START OF THE PROJECT GUTENBERG EBOOK FAUST ***\n"
            "Actual content here.\n"
            "*** END OF THE PROJECT GUTENBERG EBOOK FAUST ***\n"
            "Some footer text"
        )
        stripped = _strip_gutenberg(raw)
        assert "Actual content here." in stripped
        assert "Some header text" not in stripped
        assert "Some footer text" not in stripped


# ── ingest/web ───────────────────────────────────────────────────────────────

class TestIngestWeb:
    def test_strip_html_removes_tags(self):
        from factfull.ingest.web import _strip_html
        html = "<html><body><p>Hello <b>world</b>!</p></body></html>"
        text = _strip_html(html)
        assert "Hello" in text
        assert "world" in text
        assert "<b>" not in text
        assert "<p>" not in text

    def test_strip_html_removes_script(self):
        from factfull.ingest.web import _strip_html
        html = "<html><head><script>alert('x')</script></head><body>Content</body></html>"
        text = _strip_html(html)
        assert "alert" not in text
        assert "Content" in text

    def test_extract_title(self):
        from factfull.ingest.web import _extract_title
        html = "<html><head><title>My Article</title></head><body></body></html>"
        assert _extract_title(html) == "My Article"

    def test_extract_title_missing(self):
        from factfull.ingest.web import _extract_title
        assert _extract_title("<html><body>no title</body></html>") == ""

    def test_html_entities_decoded(self):
        from factfull.ingest.web import _strip_html
        html = "<p>AT&amp;T &lt;rock&gt;</p>"
        text = _strip_html(html)
        assert "AT&T" in text
        assert "<rock>" in text


# ── ingest/paper ─────────────────────────────────────────────────────────────

class TestIngestPaper:
    def test_clean_arxiv_id_plain(self):
        from factfull.ingest.paper import _clean_arxiv_id
        assert _clean_arxiv_id("2403.11996") == "2403.11996"

    def test_clean_arxiv_id_version(self):
        from factfull.ingest.paper import _clean_arxiv_id
        assert _clean_arxiv_id("2403.11996v3") == "2403.11996"

    def test_clean_arxiv_id_url(self):
        from factfull.ingest.paper import _clean_arxiv_id
        assert _clean_arxiv_id("https://arxiv.org/abs/2403.11996") == "2403.11996"
        assert _clean_arxiv_id("https://arxiv.org/pdf/2403.11996") == "2403.11996"

    def test_ingest_pdf_with_mock(self, tmp_path, monkeypatch):
        from factfull.ingest import paper as paper_mod

        # pdfplumber をモック
        class FakePage:
            def extract_text(self): return "Mocked PDF content. " * 50

        class FakePDF:
            pages = [FakePage()]
            metadata = {"Title": "Mock Paper"}
            def __enter__(self): return self
            def __exit__(self, *a): pass

        monkeypatch.setattr(paper_mod, "_extract_pdf_text",
                            lambda p: "Mocked PDF content. " * 50)
        monkeypatch.setattr(paper_mod, "_extract_pdf_metadata",
                            lambda p: {"title": "Mock Paper", "author": "Alice", "num_pages": 3})

        pdf = tmp_path / "mock.pdf"
        pdf.write_bytes(b"%PDF-1.4 fake")  # ダミーファイル

        doc = paper_mod.ingest_pdf(pdf, source_id="mock_001")
        assert doc.source_type == "paper"
        assert doc.title == "Mock Paper"
        assert doc.source_id == "mock_001"
        assert len(doc.chunks) > 0
