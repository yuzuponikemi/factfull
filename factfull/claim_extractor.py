"""
チェック対象ドキュメントから検証可能なアトミックなクレームを抽出する。
"""
from __future__ import annotations
import json
import re
from . import llm


_PROMPT_TEMPLATE = """\
以下の文書から、事実として確認可能なアトミックなクレームをすべて抽出してください。

## 抽出ルール
- 各クレームは1文で完結すること
- 数値・固有名詞・人名・組織名・日時を含むクレームを優先する
- 意見・推測・感情表現は除外する（「〜と思われる」「〜かもしれない」など）
- クレームは JSON 配列形式で出力すること（他のテキストは出力しない）

## 出力例
["Dario AmodeiはAnthropicのCEOである", "Claude 3のリリースは2024年3月だった"]

## 文書
{document}
"""


def extract(document: str, max_claims: int = 30) -> list[str]:
    """
    document からアトミックなクレームのリストを返す。
    LLM が JSON を返せなかった場合は行ベースでパースする。
    """
    prompt = _PROMPT_TEMPLATE.format(document=document[:12000])
    raw = llm.call(prompt, num_ctx=16384)

    # JSON 配列を抽出
    claims = _parse_json_array(raw)
    if not claims:
        # フォールバック: 行ベースパース
        claims = _parse_lines(raw)

    return claims[:max_claims]


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


def _parse_lines(text: str) -> list[str]:
    claims = []
    for line in text.splitlines():
        line = line.strip().lstrip("0123456789.-・•* ")
        if len(line) > 10:
            claims.append(line)
    return claims
