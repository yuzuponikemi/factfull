"""
話者名正規化テスト — Grounded Extraction の動作検証

テスト戦略:
  - 実際の LLM を使う (gemma4:e4b) が、入力を極小に抑えて高速化
  - 各ケース ~5-10 秒、全体 < 60 秒を目標
  - OLLAMA_URL が使えない環境では自動スキップ

実行:
  uv run pytest tests/test_speaker_normalization.py -v
  uv run pytest tests/test_speaker_normalization.py -v -s  # print あり
"""
from __future__ import annotations

import re
import pytest

# ── フィクスチャ ──────────────────────────────────────────────────────────────

# (summary_text, canonical_speakers, expected_speaker_in_descriptions)
FIXTURES = [
    pytest.param(
        # Case 1: 正規名あり — LLM が表記を揃えるか
        """\
―― **Dario Amodei**（Anthropic CEO）

Dario Amodeiは、現在のAI開発が指数関数的成長の終焉に近づいていると主張した。
彼によれば、スケーリング則はある限界に達しつつあり、次のフェーズは「研究の時代」だという。
また、AIのエンタープライズ導入における最大のボトルネックは法的・セキュリティ上の制約だと述べた。
""",
        ["Dario Amodei"],
        "Dario Amodei",
        id="grounded_single_speaker",
    ),
    pytest.param(
        # Case 2: 正規名なし（従来動作） — 表記揺れが発生しやすい
        """\
ゲストのDario Amodearyは、AIスケーリングの終焉について語った。
彼は「スケーリング則は限界に達している」と述べ、
次世代AIにはアルゴリズムの深化が必要だと主張した。
""",
        [],  # 正規名リストなし
        None,  # 期待値なし（揺れが起きても失敗させない）
        id="no_canonical_speakers",
    ),
    pytest.param(
        # Case 3: 複数話者 — 両方が正規名で出力されるか
        """\
―― **Demis Hassabis**（Google DeepMind CEO）

Demis Hassabisは、AGI実現にはワールドモデルが必要だと主張した。

―― **Ilya Sutskever**（SSI創設者）

Ilya Sutskeverは、スケーリングの時代が終わり、研究の時代が始まると述べた。
LLMだけではAGIは達成できないという立場を明確にした。
""",
        ["Demis Hassabis", "Ilya Sutskever"],
        None,  # 両方チェックは assertions で
        id="grounded_two_speakers",
    ),
]


# ── ヘルパー ──────────────────────────────────────────────────────────────────

def _extract_speaker_prefixes(entities) -> set[str]:
    """エンティティ description の [Speaker Name] prefix を集める。"""
    prefixes = set()
    for e in entities:
        m = re.match(r"^\[([^\]]+)\]", e.description or "")
        if m:
            prefixes.add(m.group(1))
    return prefixes


def _is_ollama_available() -> bool:
    try:
        import urllib.request
        urllib.request.urlopen("http://localhost:11434/api/tags", timeout=2)
        return True
    except Exception:
        return False


# ── テスト ────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(not _is_ollama_available(), reason="Ollama not running")
@pytest.mark.parametrize("summary_text,canonical_speakers,expected_speaker", FIXTURES)
def test_speaker_grounded_extraction(summary_text, canonical_speakers, expected_speaker):
    """正規スピーカー名を注入したとき、descriptions に正確な名前が現れるか。"""
    import os
    os.environ.setdefault("FACTFULL_LLM_BACKEND", "ollama")
    os.environ.setdefault("FACTFULL_OLLAMA_MODEL", "gemma4:e4b")

    from factfull.extract.podcast_extract import extract_from_summary

    entities, triples = extract_from_summary(
        summary_text,
        source_id="test_normalization",
        model="gemma4:e4b",
        chunk_size=3000,
        canonical_speakers=canonical_speakers or None,
    )

    print(f"\n  抽出: {len(entities)} entities, {len(triples)} triples")
    prefixes = _extract_speaker_prefixes(entities)
    print(f"  speaker prefixes in descriptions: {prefixes}")

    # speaker-attributed entities が存在するか
    speaker_attributed = [e for e in entities if (e.description or "").startswith("[")]
    assert len(speaker_attributed) > 0, "speaker-attributed entities が1件もない"

    # 正規名が指定されている場合、その名前が必ず使われているか
    if expected_speaker:
        assert expected_speaker in prefixes, (
            f"正規名 '{expected_speaker}' がどの description にも現れない。"
            f"実際の prefixes: {prefixes}"
        )

    # 正規名リストが2件の場合、両方チェック
    if len(canonical_speakers) == 2:
        for name in canonical_speakers:
            assert name in prefixes, (
                f"正規名 '{name}' が descriptions に現れない。"
                f"実際の prefixes: {prefixes}"
            )

    # 正規名が指定されているのに全く異なる名前が使われていないか
    if canonical_speakers:
        canonical_set = {s.lower() for s in canonical_speakers}
        for prefix in prefixes:
            close_enough = any(
                prefix.lower() in c or c in prefix.lower()
                for c in canonical_set
            )
            assert close_enough, (
                f"正規名リストにない話者名 '{prefix}' が使われた。"
                f"正規名: {canonical_speakers}"
            )


@pytest.mark.skipif(not _is_ollama_available(), reason="Ollama not running")
def test_speakers_block_injection():
    """_make_speakers_block() がプロンプトに正しく注入されるか（LLM 不要）。"""
    from factfull.extract.podcast_extract import _make_speakers_block

    block = _make_speakers_block(["Dario Amodei", "Demis Hassabis"])
    assert "Dario Amodei" in block
    assert "Demis Hassabis" in block
    assert "CANONICAL" in block
    assert "EXACT" in block


def test_make_speakers_block_empty():
    """空リストで空文字列を返すか。"""
    from factfull.extract.podcast_extract import _make_speakers_block
    assert _make_speakers_block([]) == ""


def test_extract_speakers_from_summary():
    """サマリーから ―― **Name** 形式で話者名を抽出できるか。"""
    from factfull.extract.podcast_extract import _extract_speakers_from_summary

    text = """\
―― **Dario Amodei**（Anthropic CEO）
本文...
―― **Demis Hassabis**（Google DeepMind CEO）
本文...
"""
    names = _extract_speakers_from_summary(text)
    assert "Dario Amodei" in names
    assert "Demis Hassabis" in names


def test_extract_speakers_filters_japanese():
    """日本語を含む偽話者名を除外するか。"""
    from factfull.extract.podcast_extract import _extract_speakers_from_summary

    text = "―― **Anthropicの元メンバー**\n本文..."
    names = _extract_speakers_from_summary(text)
    assert names == []
