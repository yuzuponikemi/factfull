"""factfull/bilingual/types.py のテスト（JSON 往復・前方互換）。"""
import json

from factfull.bilingual.types import Block, BilingualDoc


def _doc():
    return BilingualDoc(
        title_en="A Paper",
        title_ja="ある論文",
        authors=["Alice", "Bob"],
        source_id="2403.11996",
        arxiv_url="https://arxiv.org/abs/2403.11996",
        source_type="arxiv",
        model="translategemma:12b",
        abstract_en="We study X.",
        abstract_ja="X を研究する。",
        metadata={"num_pages": 8},
        blocks=[
            Block(id="b0001", type="heading", en="1 Introduction", level=1),
            Block(id="b0002", type="paragraph", en="Body.", ja="本文。",
                  section_path=["1 Introduction"], page=1),
            Block(id="b0003", type="figure", page=2, bbox=[10.0, 20.0, 30.0, 40.0],
                  image_path="assets/image_p2_001.png", label="Figure 1"),
        ],
    )


class TestRoundTrip:
    def test_block_round_trip(self):
        b = Block(id="b0001", type="paragraph", en="x", ja="エックス", page=3)
        assert Block.from_dict(b.to_dict()) == b

    def test_doc_round_trip(self):
        doc = _doc()
        restored = BilingualDoc.from_dict(doc.to_dict())
        assert restored == doc

    def test_from_dict_ignores_unknown_keys(self):
        d = Block(id="b0001", type="paragraph", en="x").to_dict()
        d["future_field"] = "ignored"
        b = Block.from_dict(d)
        assert b.en == "x"

    def test_json_keeps_japanese_unescaped(self):
        doc = _doc()
        s = json.dumps(doc.to_dict(), ensure_ascii=False)
        assert "ある論文" in s
        assert "\\u" not in s
