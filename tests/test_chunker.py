"""factfull/ingest/chunker.py のテスト。"""
import pytest
from factfull.ingest.chunker import (
    Chunk, tokenize, chunk_by_chars, chunk_by_sentences, chunk_text,
)


class TestTokenize:
    def test_english(self):
        tokens = tokenize("Hello World")
        assert tokens == ["hello", "world"]

    def test_japanese(self):
        tokens = tokenize("日本語")
        assert tokens == ["日", "本", "語"]

    def test_mixed(self):
        tokens = tokenize("AI 研究")
        assert "ai" in tokens
        assert "研" in tokens

    def test_numbers(self):
        tokens = tokenize("GPT4 2024")
        assert "gpt4" in tokens
        assert "2024" in tokens

    def test_empty(self):
        assert tokenize("") == []


class TestChunkByChars:
    def test_basic_split(self):
        text = "a" * 1000
        chunks = chunk_by_chars(text, chunk_size=400, overlap=80)
        assert len(chunks) > 1
        assert all(len(c.text) <= 400 for c in chunks)

    def test_short_text_single_chunk(self):
        text = "short"
        chunks = chunk_by_chars(text, chunk_size=400, overlap=80)
        assert len(chunks) == 1
        assert chunks[0].text == "short"

    def test_overlap(self):
        text = "a" * 500
        chunks = chunk_by_chars(text, chunk_size=200, overlap=50)
        # オーバーラップがあるので隣接チャンクの末尾と先頭が一致するはず
        assert chunks[0].text[-50:] == chunks[1].text[:50]

    def test_source_and_offset(self):
        text = "hello world"
        chunks = chunk_by_chars(text, source="test.txt", chunk_size=5, overlap=0)
        assert chunks[0].source == "test.txt"
        assert chunks[0].offset == 0
        assert chunks[1].offset == 5

    def test_empty_text(self):
        chunks = chunk_by_chars("", chunk_size=400, overlap=80)
        assert chunks == []


class TestChunkBySentences:
    def test_japanese_sentences(self):
        text = "これは最初の文です。次の文が続きます。三番目の文です。"
        chunks = chunk_by_sentences(text, chunk_size=15, overlap=5)
        assert len(chunks) >= 1
        # すべてのチャンクがテキストを含む
        assert all(c.text for c in chunks)

    def test_english_sentences(self):
        text = "This is sentence one. This is sentence two. This is sentence three."
        chunks = chunk_by_sentences(text, chunk_size=30, overlap=10)
        assert len(chunks) >= 1

    def test_chunk_size_respected(self):
        text = ("短い文。" * 50)
        chunks = chunk_by_sentences(text, chunk_size=30, overlap=5)
        # オーバーラップ分を除いて chunk_size を大幅に超えないこと
        for c in chunks:
            assert len(c.text) <= 30 + 30  # overlap + 1文のバッファ


class TestChunkText:
    def test_character_strategy(self):
        text = "x" * 500
        chunks = chunk_text(text, strategy="character", chunk_size=200, overlap=0)
        assert len(chunks) == 3  # 200 + 200 + 100

    def test_sentence_strategy(self):
        text = "文A。文B。文C。"
        chunks = chunk_text(text, strategy="sentence", chunk_size=10, overlap=0)
        assert len(chunks) >= 1

    def test_unknown_strategy_falls_back_to_character(self):
        text = "y" * 100
        chunks = chunk_text(text, strategy="unknown", chunk_size=50, overlap=0)
        assert len(chunks) == 2
