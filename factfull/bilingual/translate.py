"""
factfull/bilingual/translate.py
===============================
段落単位の英日翻訳。整列（順序・件数）を保証するため、番号つき JSON 配列
プロトコルでバッチ翻訳し、不整合時はブロック単位フォールバックする。

公開関数:
    translate_blocks(blocks, doc_title_en, *, model, ...) -> None  （ja を破壊的に充填）
"""
from __future__ import annotations

import json
import re

from factfull.bilingual.types import Block
from factfull.llm import call

# 翻訳対象のブロック種別（figure/table は画像なので対象外）
_TRANSLATABLE = {"title", "heading", "abstract", "paragraph", "caption"}


class BatchMismatch(Exception):
    """バッチ翻訳の応答が件数/ID 不整合だったことを示す。"""


_PROMPT = """\
あなたは学術論文の英日翻訳者です。以下は論文「{title}」の本文ブロックです。
各ブロックを自然で正確な日本語の学術文体（である調）に翻訳してください。

規則:
- 固有名詞・人名・製品名・データセット名・手法名は原語のまま残す。
- 数式・記号・LaTeX 断片・引用 [12] や (Smith et al., 2020) はそのまま保持する。
- 専門用語は初出のみ「日本語（English）」の形で併記してよい。
- ブロックを結合・分割・並び替え・省略しない。入力と同じ件数・同じ i を返す。
{glossary}{section}
出力は次の JSON 配列のみ（説明文・コードフェンスなし）:
[{{"i": 0, "ja": "..."}}, {{"i": 1, "ja": "..."}}]

入力:
{payload}
"""


def translate_blocks(
    blocks: list[Block],
    doc_title_en: str,
    *,
    model: str = "translategemma:12b",
    batch_chars: int = 3000,
    num_ctx: int = 8192,
    num_predict: int = 8192,
) -> None:
    """blocks の翻訳対象ブロックの ja を破壊的に充填する。"""
    targets = [
        b for b in blocks
        if b.type in _TRANSLATABLE and not b.skip_translate and b.en.strip()
    ]
    glossary: dict[str, str] = {}
    done = 0
    for batch in _batches(targets, batch_chars):
        section = batch[0].section_path[0] if batch[0].section_path else ""
        try:
            translations = _translate_batch(
                batch, doc_title_en, section, glossary,
                model=model, num_ctx=num_ctx, num_predict=num_predict,
            )
        except (BatchMismatch, json.JSONDecodeError, ValueError) as e:
            print(f"  [bilingual] バッチ整列に失敗、ブロック単位にフォールバック: {e}", flush=True)
            translations = [
                _translate_one(
                    b, doc_title_en, model=model, num_ctx=num_ctx, num_predict=num_predict,
                )
                for b in batch
            ]
        for b, ja in zip(batch, translations):
            b.ja = ja
        _update_glossary(batch, translations, glossary)
        done += len(batch)
        print(f"  [bilingual] 翻訳 {done}/{len(targets)} ブロック", flush=True)


def _batches(blocks: list[Block], batch_chars: int) -> list[list[Block]]:
    """連続する翻訳対象を、累積 EN 文字数が batch_chars を超えるまでまとめる。

    見出しは単独で送らず後続段落と同じバッチに入れて文脈を与える
    （見出し単独で 1 バッチになるのを避ける）。
    """
    batches: list[list[Block]] = []
    cur: list[Block] = []
    size = 0
    for b in blocks:
        blen = len(b.en)
        # 既に上限を超えていて、かつ直近が見出しでなければ切る
        if cur and size + blen > batch_chars and cur[-1].type != "heading":
            batches.append(cur)
            cur, size = [], 0
        cur.append(b)
        size += blen
    if cur:
        batches.append(cur)
    return batches


def _translate_batch(
    batch: list[Block],
    title: str,
    section: str,
    glossary: dict[str, str],
    *,
    model: str,
    num_ctx: int,
    num_predict: int,
) -> list[str]:
    """1 バッチを JSON 配列プロトコルで翻訳し、入力順の訳文リストを返す。"""
    payload = json.dumps(
        [{"i": i, "en": b.en} for i, b in enumerate(batch)],
        ensure_ascii=False,
    )
    gloss = ""
    pairs_list = [(en, ja) for en, ja in glossary.items() if ja]
    if pairs_list:
        pairs = "、".join(f"{en}→{ja}" for en, ja in pairs_list[:30])
        gloss = f"- 既訳の用語集に従い表記を統一する: {pairs}\n"
    sec = f"\nセクション: {section}\n" if section else ""
    prompt = _PROMPT.format(title=title, glossary=gloss, section=sec, payload=payload)

    raw = call(prompt, model=model, num_ctx=num_ctx, num_predict=num_predict)
    return _parse_response(raw, len(batch))


def _parse_response(raw: str, n_expected: int) -> list[str]:
    """LLM 応答から JSON 配列を取り出し、i でインデックスした訳文リストを返す。

    件数不足・i の欠落/重複があれば BatchMismatch を送出する。
    """
    text = raw.strip()
    # コードフェンス除去
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE).strip()
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if not m:
        raise BatchMismatch("JSON 配列が見つかりません")
    data = json.loads(m.group())
    if not isinstance(data, list) or len(data) != n_expected:
        raise BatchMismatch(f"件数不一致: expected {n_expected}, got {len(data)}")

    by_index: dict[int, str] = {}
    for item in data:
        if not isinstance(item, dict) or "i" not in item or "ja" not in item:
            raise BatchMismatch("i/ja を持たない要素があります")
        by_index[int(item["i"])] = str(item["ja"])
    if set(by_index) != set(range(n_expected)):
        raise BatchMismatch(f"i の集合が不正: {sorted(by_index)}")
    return [by_index[i] for i in range(n_expected)]


def _translate_one(
    block: Block,
    title: str,
    *,
    model: str,
    num_ctx: int,
    num_predict: int,
) -> str:
    """フォールバック: 1 ブロックを単独翻訳する（整列は自明）。"""
    prompt = (
        f"次の英語テキストを、論文「{title}」の一部として自然で正確な日本語の"
        "学術文体（である調）に翻訳してください。固有名詞・数式・引用はそのまま"
        "保持し、訳文のみを出力してください（前置き不要）。\n\n"
        f"{block.en}"
    )
    return call(prompt, model=model, num_ctx=num_ctx, num_predict=num_predict).strip()


def _update_glossary(
    batch: list[Block], translations: list[str], glossary: dict[str, str]
) -> None:
    """「Full Name (ACRONYM)」形式で定義された略語を用語集に控える。

    本文中で初出定義された略語は原語のまま統一したいので、acronym→acronym を
    登録してプロンプトの用語集に供給する（表記ゆれ抑制の補助。ベストエフォート）。
    en↔ja の語単位アラインメントは行わない。
    """
    for ja in translations:
        for acro in re.findall(r"\(([A-Z][A-Za-z0-9]{1,6})\)", ja):
            glossary.setdefault(acro, acro)
