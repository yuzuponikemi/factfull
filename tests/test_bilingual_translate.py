"""factfull/bilingual/translate.py のテスト（LLM をモック）。"""
import json

import pytest

from factfull.bilingual import translate as T
from factfull.bilingual.types import Block


def _para(en, **kw):
    return Block(id="", type="paragraph", en=en, **kw)


class TestParseResponse:
    def test_plain_array(self):
        raw = '[{"i": 0, "ja": "あ"}, {"i": 1, "ja": "い"}]'
        assert T._parse_response(raw, 2) == ["あ", "い"]

    def test_code_fence_stripped(self):
        raw = '```json\n[{"i": 0, "ja": "あ"}]\n```'
        assert T._parse_response(raw, 1) == ["あ"]

    def test_reordered_indices_matched(self):
        raw = '[{"i": 1, "ja": "い"}, {"i": 0, "ja": "あ"}]'
        assert T._parse_response(raw, 2) == ["あ", "い"]

    def test_wrong_count_raises(self):
        with pytest.raises(T.BatchMismatch):
            T._parse_response('[{"i": 0, "ja": "あ"}]', 2)

    def test_missing_index_raises(self):
        raw = '[{"i": 0, "ja": "あ"}, {"i": 0, "ja": "い"}]'
        with pytest.raises(T.BatchMismatch):
            T._parse_response(raw, 2)


class TestBatches:
    def test_respects_batch_chars(self):
        blocks = [_para("x" * 100) for _ in range(5)]
        batches = T._batches(blocks, batch_chars=250)
        assert all(sum(len(b.en) for b in batch) <= 250 or len(batch) == 1
                   for batch in batches)
        assert sum(len(b) for b in batches) == 5

    def test_heading_attaches_to_following(self):
        blocks = [
            Block(id="", type="heading", en="Method"),
            _para("y" * 400),
        ]
        batches = T._batches(blocks, batch_chars=100)
        # 見出しは単独で切られず後続と同じバッチに入る
        assert batches[0][0].type == "heading"
        assert len(batches[0]) >= 2


class TestTranslateBlocks:
    def test_happy_path_fills_ja(self, monkeypatch):
        def fake_call(prompt, **kw):
            payload = json.loads(prompt.split("入力:\n", 1)[1].strip())
            return json.dumps([{"i": it["i"], "ja": "訳:" + it["en"]} for it in payload])
        monkeypatch.setattr(T, "call", fake_call)

        blocks = [_para("alpha"), _para("beta")]
        T.translate_blocks(blocks, "Doc", batch_chars=1000)
        assert blocks[0].ja == "訳:alpha" and blocks[1].ja == "訳:beta"

    def test_fallback_on_mismatch(self, monkeypatch):
        calls = {"batch": 0, "single": 0}

        def fake_call(prompt, **kw):
            if "出力は次の JSON 配列" in prompt:
                calls["batch"] += 1
                return "[]"  # 件数不一致 → BatchMismatch
            calls["single"] += 1
            return "単独訳"
        monkeypatch.setattr(T, "call", fake_call)

        blocks = [_para("a"), _para("b")]
        T.translate_blocks(blocks, "Doc", batch_chars=1000)
        assert calls["single"] == 2
        assert blocks[0].ja == "単独訳" and blocks[1].ja == "単独訳"

    def test_skip_translate_and_figures_untouched(self, monkeypatch):
        monkeypatch.setattr(T, "call", lambda prompt, **kw: '[{"i": 0, "ja": "訳"}]')
        blocks = [
            _para("translate me"),
            Block(id="", type="figure", image_path="assets/x.png"),
            Block(id="", type="reference", en="[1] ref", skip_translate=True),
        ]
        T.translate_blocks(blocks, "Doc")
        assert blocks[0].ja == "訳"
        assert blocks[1].ja == "" and blocks[2].ja == ""
