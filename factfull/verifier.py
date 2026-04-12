"""
クレームと証拠パッセージを照合して判定を下す。
"""
from __future__ import annotations
import re
from dataclasses import dataclass
from enum import Enum
from .indexer import Chunk
from . import llm


class Verdict(str, Enum):
    SUPPORTED = "supported"       # ✅ 支持されている
    CONTRADICTED = "contradicted" # ❌ 矛盾している
    PARTIAL = "partial"           # ⚠️ 部分的に正しい / 誇張
    UNVERIFIABLE = "unverifiable" # ❓ 証拠なし


VERDICT_EMOJI = {
    Verdict.SUPPORTED: "✅",
    Verdict.CONTRADICTED: "❌",
    Verdict.PARTIAL: "⚠️",
    Verdict.UNVERIFIABLE: "❓",
}

VERDICT_SCORE = {
    Verdict.SUPPORTED: 1.0,
    Verdict.PARTIAL: 0.5,
    Verdict.CONTRADICTED: 0.0,
    Verdict.UNVERIFIABLE: None,  # スコア計算から除外
}


@dataclass
class VerificationResult:
    claim: str
    verdict: Verdict
    reason: str
    evidence: list[str]  # 使用した証拠パッセージ


_PROMPT_TEMPLATE = """\
あなたはファクトチェッカーです。以下のクレームが、提示された証拠に基づいて正しいかどうか判定してください。

## クレーム
{claim}

## 証拠パッセージ（Truth ソースから取得）
{evidence}

## 判定ルール
証拠に基づいて以下のいずれかを選択してください：
- supported: クレームは証拠によって明確に支持されている
- contradicted: クレームは証拠と明確に矛盾している
- partial: クレームは部分的に正しいが、誇張・誤記・欠落がある
- unverifiable: 証拠が不十分でクレームの真偽を判定できない

## 出力形式（JSON のみ出力、他のテキストは不要）
{{"verdict": "supported|contradicted|partial|unverifiable", "reason": "1〜2文で判定理由"}}
"""


def verify(
    claim: str,
    evidence_chunks: list[Chunk],
) -> VerificationResult:
    """クレームを証拠と照合して VerificationResult を返す。"""
    if not evidence_chunks:
        return VerificationResult(
            claim=claim,
            verdict=Verdict.UNVERIFIABLE,
            reason="関連する証拠が見つかりませんでした。",
            evidence=[],
        )

    evidence_texts = [c.text for c in evidence_chunks]
    evidence_block = "\n\n---\n\n".join(
        f"[{c.source}]\n{c.text}" for c in evidence_chunks
    )

    prompt = _PROMPT_TEMPLATE.format(claim=claim, evidence=evidence_block)
    raw = llm.call(prompt, num_ctx=8192)

    verdict, reason = _parse_verdict(raw)
    return VerificationResult(
        claim=claim,
        verdict=verdict,
        reason=reason,
        evidence=evidence_texts,
    )


def _parse_verdict(text: str) -> tuple[Verdict, str]:
    import json as _json

    m = re.search(r"\{.*?\}", text, re.DOTALL)
    if m:
        try:
            data = _json.loads(m.group())
            verdict_str = data.get("verdict", "").lower()
            reason = data.get("reason", "")
            if verdict_str in Verdict._value2member_map_:
                return Verdict(verdict_str), reason
        except _json.JSONDecodeError:
            pass

    # フォールバック: テキストからキーワード検索
    text_lower = text.lower()
    if "contradicted" in text_lower:
        verdict = Verdict.CONTRADICTED
    elif "partial" in text_lower:
        verdict = Verdict.PARTIAL
    elif "supported" in text_lower:
        verdict = Verdict.SUPPORTED
    else:
        verdict = Verdict.UNVERIFIABLE

    # reason は最初の文を使う
    first_sentence = text.split("。")[0].strip() if "。" in text else text[:100]
    return verdict, first_sentence
