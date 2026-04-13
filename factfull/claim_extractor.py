"""
チェック対象ドキュメントから検証可能なアトミックなクレームを抽出する。
ドキュメントをチャンク分割して全文をカバーする。
"""
from __future__ import annotations
import json
import re
from . import llm


_PROMPT_TEMPLATE = """\
以下の文書から、事実として確認可能なアトミックなクレームをすべて抽出してください。

## 抽出ルール
- 各クレームは1文で完結すること
- 数値・固有名詞・人名・組織名・日時・金額を含むクレームを優先する
- 意見・推測・感情表現は除外する（「〜と思われる」「〜かもしれない」など）
- クレームは JSON 配列形式で出力すること（前置き・説明・他のテキストは一切出力しない）
- 必ず文書の内容だけを元にすること

## 出力形式（JSONのみ）
["クレーム1", "クレーム2", ...]

## 文書
{document}
"""

# 1チャンクあたりの文字数（num_ctx=8192 に収まる範囲）
_CHUNK_SIZE = 4000

# ファクトチェック対象外のセクション見出しキーワード
# （AI の考察・意見・メタ情報など、トランスクリプトで検証できないもの）
_SKIP_SECTION_KEYWORDS = frozenset({"編集後記", "キーワード", "動画"})


def _strip_non_factual_sections(document: str) -> str:
    """
    ## 編集後記 など事実検証対象外のセクションを除去してから返す。
    見出し行を検出し、該当セクションの本文ごとスキップする。
    """
    lines = document.splitlines(keepends=True)
    result: list[str] = []
    skipping = False
    for line in lines:
        m = re.match(r'^#{1,3} (.+)$', line.rstrip())
        if m:
            section_title = m.group(1)
            skipping = any(kw in section_title for kw in _SKIP_SECTION_KEYWORDS)
        if not skipping:
            result.append(line)
    return "".join(result)


def extract(document: str, max_claims: int = 30) -> list[str]:
    """
    document からアトミックなクレームのリストを返す。
    長いドキュメントはチャンク分割して全文をカバーする。
    編集後記など意見・考察セクションは事前に除外する。
    """
    document = _strip_non_factual_sections(document)
    chunks = _split_chunks(document, _CHUNK_SIZE)
    seen: set[str] = set()
    all_claims: list[str] = []

    for i, chunk in enumerate(chunks, 1):
        if len(all_claims) >= max_claims:
            break
        print(f"  [claim] チャンク {i}/{len(chunks)} を処理中...", flush=True)
        prompt = _PROMPT_TEMPLATE.format(document=chunk)
        raw = llm.call(prompt, num_ctx=8192)

        claims = _parse_json_array(raw)
        if not claims:
            claims = _parse_lines(raw)

        for c in claims:
            if c not in seen:
                seen.add(c)
                all_claims.append(c)

    return all_claims[:max_claims]


def _split_chunks(text: str, size: int) -> list[str]:
    """文字数ベースでテキストをオーバーラップなしに分割する。"""
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        # 文末（。！？\n）で区切る
        if end < len(text):
            for sep in ("。\n", "。", "\n\n", "\n"):
                pos = text.rfind(sep, start, end)
                if pos > start:
                    end = pos + len(sep)
                    break
        chunks.append(text[start:end])
        start = end
    return chunks


def _parse_json_array(text: str) -> list[str]:
    m = re.search(r"\[.*?\]", text, re.DOTALL)
    if not m:
        return []
    try:
        result = json.loads(m.group())
        if isinstance(result, list):
            return [str(x) for x in result if x]
    except json.JSONDecodeError:
        pass
    return []


_META_PATTERNS = re.compile(
    r"^(以下[はに]|これら[はの]|上記[はの]|抽出しました|クレームを抽出|ご確認|注意|なお、|※)"
)


def _parse_lines(text: str) -> list[str]:
    claims = []
    for line in text.splitlines():
        line = line.strip().lstrip("0123456789.-・•* ")
        if len(line) > 10 and not _META_PATTERNS.match(line):
            claims.append(line)
    return claims
