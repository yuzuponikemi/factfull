"""
論文 英日対訳パイプラインの実動作テスト（pymupdf 必須、Ollama は任意）。

extract → segment は pymupdf があれば実行（合成 PDF を使うのでネットワーク不要）。
翻訳まで含むエンドツーエンドは Ollama（port 11435）が起動している場合のみ実行する。

実行:
    uv run --with pytest pytest tests/test_bilingual_integration.py -v -s
"""
import json
from pathlib import Path

import pytest

pymupdf = pytest.importorskip("pymupdf")


def _ollama_available() -> bool:
    import urllib.request
    try:
        urllib.request.urlopen("http://localhost:11435/api/tags", timeout=3)
        return True
    except Exception:
        return False


def _make_pdf(path: Path) -> Path:
    doc = pymupdf.open()
    page = doc.new_page(width=400, height=600)
    page.insert_text((50, 50), "A Tiny Test Paper", fontsize=18)
    page.insert_text((50, 90), "1  Introduction", fontsize=14)
    page.insert_text((50, 115), "This paper studies nothing in particular.", fontsize=10)
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 80, 60))
    pix.set_rect(pix.irect, (50, 100, 200))
    page.insert_image(pymupdf.Rect(50, 150, 130, 210), pixmap=pix)
    page.insert_text((50, 225), "Figure 1: Nothing.", fontsize=9)
    doc.save(str(path))
    doc.close()
    return path


def test_extract_and_segment_real_pdf(tmp_path):
    """pymupdf で実 PDF を抽出・整形できること（翻訳なし）。"""
    from factfull.bilingual.extract import extract_structured_blocks
    from factfull.bilingual.segment import segment_blocks

    pdf = _make_pdf(tmp_path / "tiny.pdf")
    raw = extract_structured_blocks(pdf)
    assert any(b.kind == "image" for b in raw)

    blocks = segment_blocks(raw, assets_dir=tmp_path / "assets")
    types = {b.type for b in blocks}
    assert "heading" in types
    assert "paragraph" in types
    fig = [b for b in blocks if b.type == "figure"]
    assert fig and fig[0].image_path and (tmp_path / fig[0].image_path).exists()
    assert fig[0].label == "Figure 1"


@pytest.mark.skipif(not _ollama_available(), reason="Ollama (port 11435) 未起動")
def test_end_to_end_with_ollama(tmp_path):
    """Ollama が起動していればタイトルから JSON まで通すこと。"""
    from factfull.bilingual.pipeline import BilingualConfig, run_bilingual

    pdf = _make_pdf(tmp_path / "tiny.pdf")
    cfg = BilingualConfig(model="translategemma:12b", output_base=tmp_path / "out")
    result = run_bilingual(cfg, str(pdf))

    data = json.loads(result.json_path.read_text(encoding="utf-8"))
    assert data["title_ja"]
    assert any(b["ja"] for b in data["blocks"] if b["type"] == "paragraph")
