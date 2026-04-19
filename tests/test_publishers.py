"""factfull/publishers/homupe.py のテスト（LLM・Selenium 不要）。"""
import os
from pathlib import Path
from datetime import date

import pytest
from factfull.publishers.homupe import (
    BlogMetadata, default_blog_dir,
    _extract_guest_name, _extract_keywords, _default_slug,
    _build_tweet_text, _insert_guest, _parse_json,
)


class TestDefaultBlogDir:
    def test_uses_homupe_root_env(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOMUPE_ROOT", str(tmp_path))
        d = default_blog_dir()
        today = date.today()
        assert str(today.year) in str(d)
        assert tmp_path.name in str(d)

    def test_accepts_explicit_root(self, tmp_path):
        d = default_blog_dir(homupe_root=tmp_path)
        assert "blog/posts" in str(d)


class TestExtractGuestName:
    def test_colon_separator(self):
        assert _extract_guest_name("Jensen Huang: NVIDIA and AI") == "Jensen Huang"

    def test_dash_separator(self):
        assert _extract_guest_name("Andrej Karpathy - The Future of AI") == "Andrej Karpathy"

    def test_em_dash_separator(self):
        assert _extract_guest_name("Sam Altman — OpenAI's Vision") == "Sam Altman"

    def test_no_separator_returns_none(self):
        assert _extract_guest_name("State of AI in 2026") is None

    def test_lowercase_word_returns_none(self):
        # "the AI Podcast" など固有名詞でない場合
        assert _extract_guest_name("the future: AI trends") is None


class TestExtractKeywords:
    def test_extracts_keywords(self):
        summary = "## キーワード\n`AI` / `機械学習` / `LLM`\n\n## 次のセクション"
        tags = _extract_keywords(summary)
        assert "AI" in tags
        assert "機械学習" in tags
        assert "LLM" in tags

    def test_no_keywords_section(self):
        assert _extract_keywords("## 概要\nテスト") == []

    def test_max_10_tags(self):
        kw = " / ".join([f"`tag{i}`" for i in range(20)])
        summary = f"## キーワード\n{kw}\n\n"
        tags = _extract_keywords(summary)
        assert len(tags) <= 10


class TestDefaultSlug:
    def test_basic(self):
        assert _default_slug("Jensen Huang: NVIDIA") == "jensen-huang-nvidia"

    def test_max_length(self):
        long_title = "a " * 50
        assert len(_default_slug(long_title)) <= 60

    def test_special_chars_removed(self):
        slug = _default_slug("Hello, World! (2026)")
        assert "," not in slug
        assert "!" not in slug
        assert "(" not in slug


class TestBuildTweetText:
    def setup_method(self):
        self.meta = BlogMetadata(
            title_ja="テストタイトル",
            excerpt="これはテスト用の抜粋文です。AIの未来について議論します。",
            tags=["AI", "LLM"],
            guest="テストゲスト",
            slug="2026-04-19-test",
            date="2026-04-19",
        )

    def test_contains_title(self):
        text = _build_tweet_text(self.meta, "https://example.com/blog/test/")
        assert "テストタイトル" in text

    def test_contains_url(self):
        url = "https://example.com/blog/test/"
        text = _build_tweet_text(self.meta, url)
        assert url in text

    def test_under_max_length(self):
        text = _build_tweet_text(self.meta, "https://example.com/blog/test/", max_len=270)
        assert len(text) <= 270


class TestInsertGuest:
    def test_inserts_after_front_matter(self):
        summary = "---\ndate: 2026-04-19\n---\n\n## 概要\n内容"
        result = _insert_guest(summary, "ゲスト紹介文")
        assert "## ゲスト" in result
        assert result.index("## ゲスト") < result.index("## 概要")

    def test_fallback_when_no_front_matter(self):
        summary = "## 概要\n内容"
        result = _insert_guest(summary, "ゲスト紹介文")
        assert "## ゲスト" in result


class TestParseJson:
    def test_valid_json(self):
        d = _parse_json('{"title_ja": "テスト", "slug": "test"}')
        assert d["title_ja"] == "テスト"

    def test_json_in_text(self):
        d = _parse_json('Sure! {"title_ja": "タイトル"} Done.')
        assert d["title_ja"] == "タイトル"

    def test_invalid_returns_empty(self):
        assert _parse_json("no json here") == {}
