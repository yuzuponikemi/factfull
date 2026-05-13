"""
tests/test_chapter_resolver.py
=================================
ChapterResolver の統合テスト。

各テストは実際の Web アクセス（Wikipedia API / Open Library）を行うため、
ネットワーク接続と Ollama が必要。

実行:
    uv run pytest tests/test_chapter_resolver.py -v -s

特定の本だけテスト:
    uv run pytest tests/test_chapter_resolver.py -v -s -k "baudrillard"
"""
from __future__ import annotations

import pytest
from factfull.book.chapter_resolver import ChapterResolver, ChapterStructure


# ── 既知の正解（グラウンドトゥルース） ────────────────────────────────────────

# 各本の「必須」章・部のキーワード（正規化後のタイトルに含まれていれば OK）
GROUND_TRUTH: dict[str, dict] = {
    "baudrillard_consumer_society": {
        "title": "La Société de consommation",
        "author": "Jean Baudrillard",
        # 実際の3-4部構成: liturgie formelle(abondance) / théorie de la consommation /
        #                mass-media sexe loisirs / anomie (部によってはサブ章)
        # ロシア語 Wikipedia: Баланс изобилия / Теория потребления / СМИ секс досуг
        "required_keywords": ["теория", "потребления"],  # ロシア語版に確実に存在
        "forbidden_keywords": ["sign value", "hyperreality", "rise of"],  # 幻覚タイトル
        "min_chapters": 2,
        "min_confidence": 0.4,
    },
    "wittgenstein_investigations": {
        "title": "Philosophical Investigations",
        "author": "Ludwig Wittgenstein",
        # Part I (remarks 1–693) + Part II / Philosophy of Psychology
        "required_keywords": ["part"],
        "forbidden_keywords": [],
        "min_chapters": 2,
        "min_confidence": 0.4,
    },
    "nietzsche_zarathustra": {
        "title": "Thus Spoke Zarathustra",
        "author": "Friedrich Nietzsche",
        # 4 Teile / 4 Parts
        "required_keywords": ["part", "zarathustra"],
        "forbidden_keywords": [],
        "min_chapters": 4,
        "min_confidence": 0.5,
    },
    "debord_spectacle": {
        "title": "The Society of the Spectacle",
        "author": "Guy Debord",
        # 9 chapters (theses 1-221)
        "required_keywords": ["spectacle", "commodity"],
        "forbidden_keywords": [],
        "min_chapters": 5,
        "min_confidence": 0.4,
    },
    "plato_republic": {
        "title": "The Republic",
        "author": "Plato",
        # 10 books
        "required_keywords": ["book"],
        "forbidden_keywords": [],
        "min_chapters": 8,
        "min_confidence": 0.5,
    },
}


# ── フィクスチャ ──────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def resolver() -> ChapterResolver:
    return ChapterResolver(model="gemma4:e4b")


# ── ヘルパー ──────────────────────────────────────────────────────────────────

def assert_structure(result: ChapterStructure, gt: dict) -> None:
    """ChapterStructure が期待値を満たすかアサートする。"""
    print(f"\n  Sources consulted: {len(result.sources_consulted)}")
    print(f"  Sources agreed:   {result.sources_agreed}")
    print(f"  Confidence:       {result.confidence:.2f}")
    print(f"  Chapters found:   {len(result.chapters)}")
    for ch in result.chapters:
        print(f"    {ch}")

    assert len(result.chapters) >= gt["min_chapters"], (
        f"章数が少なすぎる: {len(result.chapters)} < {gt['min_chapters']}"
    )
    assert result.confidence >= gt["min_confidence"], (
        f"confidence が低すぎる: {result.confidence:.2f} < {gt['min_confidence']}"
    )

    all_titles = " ".join(ch.normalized() for ch in result.chapters)

    for kw in gt.get("required_keywords", []):
        assert kw.lower() in all_titles, (
            f"必須キーワード '{kw}' が章タイトルに見つからない\n全タイトル: {all_titles}"
        )

    for kw in gt.get("forbidden_keywords", []):
        assert kw.lower() not in all_titles, (
            f"幻覚キーワード '{kw}' が章タイトルに含まれている（章立てが架空の可能性）\n全タイトル: {all_titles}"
        )


# ── テストケース ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize("book_id", list(GROUND_TRUTH.keys()))
def test_resolve_known_books(resolver: ChapterResolver, book_id: str) -> None:
    """既知の書籍の章立てを正しく解決できる。"""
    gt = GROUND_TRUTH[book_id]
    result = resolver.resolve(gt["title"], gt["author"])
    assert_structure(result, gt)


def test_reliability_flag(resolver: ChapterResolver) -> None:
    """is_reliable() が適切に機能する。"""
    result = resolver.resolve("La Société de consommation", "Jean Baudrillard")
    # 有名書籍は複数ソースで確認できるはず
    if result.sources_agreed >= 2:
        assert result.is_reliable() == (result.confidence >= 0.6)


def test_unknown_book_returns_low_confidence(resolver: ChapterResolver) -> None:
    """存在しない書籍は低 confidence を返す。"""
    result = resolver.resolve(
        "Xyz Nonexistent Book Title 99999", "Anonymous Author 12345"
    )
    assert result.confidence < 0.5, (
        f"存在しない本が高 confidence を返した: {result.confidence}"
    )


def test_markdown_table_output(resolver: ChapterResolver) -> None:
    """to_markdown_table() が有効な Markdown を返す。"""
    result = resolver.resolve("The Republic", "Plato")
    if result.chapters:
        table = result.to_markdown_table()
        assert "| # |" in table
        assert len(table.split("\n")) >= 3  # ヘッダー + セパレーター + 1行以上


# ── 個別ソーステスト ──────────────────────────────────────────────────────────

def test_wikidata_resolution_baudrillard(resolver: ChapterResolver) -> None:
    """Wikidata でボードリヤールの多言語記事タイトルを解決できる。"""
    titles = resolver._resolve_via_wikidata("La Société de consommation", "Jean Baudrillard")
    print(f"\n  Wikidata sitelinks: {titles}")
    assert titles, "Wikidata sitelinks が取得できなかった"
    # ロシア語版は章構成が最も詳しい
    assert "ru" in titles or "fr" in titles, "ru/fr のどちらかが取得できるはず"


def test_wikipedia_sections_baudrillard(resolver: ChapterResolver) -> None:
    """Wikipedia の sections API でボードリヤールのロシア語記事を取得できる。"""
    # Wikidata 経由で正確なタイトルを取得
    titles = resolver._resolve_via_wikidata("La Société de consommation", "Jean Baudrillard")
    article_title = titles.get("ru", "")
    url, sections = resolver._fetch_wikipedia_sections(
        "La Société de consommation", "Jean Baudrillard", "ru",
        article_title=article_title,
    )
    assert url, "Wikipedia RU: URL が取得できなかった"
    print(f"\n  URL: {url}")
    print(f"  Sections: {len(sections)}")
    for s in sections[:15]:
        print(f"    [{s.get('number')}] {s.get('line', '')}")
    assert len(sections) >= 3, "章構成が取得できていない"


def test_wikipedia_sections_nietzsche(resolver: ChapterResolver) -> None:
    """Wikipedia でニーチェの章構成を取得できる。"""
    titles = resolver._resolve_via_wikidata("Thus Spoke Zarathustra", "Friedrich Nietzsche")
    print(f"\n  Wikidata sitelinks: {titles}")
    article_title = titles.get("en", "") or titles.get("de", "")
    url, sections = resolver._fetch_wikipedia_sections(
        "Thus Spoke Zarathustra", "Friedrich Nietzsche", "en",
        article_title=article_title,
    )
    assert url, f"Wikipedia EN: URL が取得できなかった (article_title={article_title!r})"
    toc = resolver._extract_toc_from_sections(sections, "en")
    print(f"\n  Nietzsche TOC ({len(toc)} entries):")
    for ch in toc:
        print(f"    {ch}")


def test_openlibrary_baudrillard(resolver: ChapterResolver) -> None:
    """Open Library でボードリヤールの TOC を検索できる（あれば）。"""
    result = resolver._fetch_openlibrary("La Société de consommation", "Jean Baudrillard")
    if result:
        url, toc = result
        print(f"\n  OpenLibrary URL: {url}")
        print(f"  TOC entries: {len(toc)}")
        for ch in toc[:10]:
            print(f"    {ch}")
    else:
        pytest.skip("Open Library に TOC データなし（正常）")


def test_consensus_with_two_identical_sources() -> None:
    """同一の章立てが2ソースから来た場合、高 confidence を返す。"""
    from factfull.book.chapter_resolver import ChapterEntry

    resolver = ChapterResolver.__new__(ChapterResolver)
    toc = [
        ChapterEntry(title="The Formal Liturgy of the Object", num="1"),
        ChapterEntry(title="Theory of Consumption", num="2"),
        ChapterEntry(title="Mass Media, Sex and Leisure", num="3"),
        ChapterEntry(title="Anomie in the Affluent Society", num="4"),
    ]
    extractions = [("source_a", toc), ("source_b", toc)]
    chapters, confidence, agreed = resolver._consensus(extractions)

    assert len(chapters) == 4
    assert confidence >= 0.6
    assert agreed >= 2


def test_consensus_rejects_single_source_chapters() -> None:
    """1ソースにしかない章は採用されない。"""
    from factfull.book.chapter_resolver import ChapterEntry

    resolver = ChapterResolver.__new__(ChapterResolver)
    toc_a = [
        ChapterEntry(title="Real Chapter One", num="1"),
        ChapterEntry(title="Real Chapter Two", num="2"),
        ChapterEntry(title="Hallucinated Chapter Only in A", num="3"),
    ]
    toc_b = [
        ChapterEntry(title="Real Chapter One", num="1"),
        ChapterEntry(title="Real Chapter Two", num="2"),
    ]
    chapters, confidence, agreed = resolver._consensus([("a", toc_a), ("b", toc_b)])
    titles = [ch.normalized() for ch in chapters]

    assert any("real chapter one" in t for t in titles)
    assert any("real chapter two" in t for t in titles)
    assert not any("hallucinated" in t for t in titles), (
        "幻覚章が合意リストに含まれてしまった"
    )
