"""
検証結果からファクトチェックレポート（Markdown）を生成する。
"""
from __future__ import annotations
from datetime import datetime
from .verifier import VerificationResult, Verdict, VERDICT_EMOJI, VERDICT_SCORE


def generate_report(
    results: list[VerificationResult],
    target_name: str = "",
    truth_names: list[str] | None = None,
) -> str:
    score = _compute_score(results)
    counts = _count_verdicts(results)

    lines: list[str] = []
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    lines.append(f"# Factfull チェックレポート")
    if target_name:
        lines.append(f"\n**対象**: {target_name}")
    if truth_names:
        lines.append(f"**Truth ソース**: {', '.join(truth_names)}")
    lines.append(f"**実行日時**: {now}")
    lines.append("")

    # サマリー
    lines.append("## 📊 総合評価")
    lines.append("")
    lines.append(f"**信頼度スコア: {score:.0f} / 100**")
    lines.append("")
    lines.append("| 判定 | 件数 |")
    lines.append("|------|------|")
    lines.append(f"| ✅ 支持 (supported) | {counts[Verdict.SUPPORTED]} |")
    lines.append(f"| ❌ 矛盾 (contradicted) | {counts[Verdict.CONTRADICTED]} |")
    lines.append(f"| ⚠️ 要確認 (partial) | {counts[Verdict.PARTIAL]} |")
    lines.append(f"| ❓ 判定不能 (unverifiable) | {counts[Verdict.UNVERIFIABLE]} |")
    lines.append(f"| 合計 | {len(results)} |")
    lines.append("")

    # 矛盾・要確認を先出し
    priority = [Verdict.CONTRADICTED, Verdict.PARTIAL, Verdict.SUPPORTED, Verdict.UNVERIFIABLE]

    for verdict in priority:
        subset = [r for r in results if r.verdict == verdict]
        if not subset:
            continue
        emoji = VERDICT_EMOJI[verdict]
        lines.append(f"## {emoji} {verdict.value.upper()}")
        lines.append("")
        for r in subset:
            lines.append(f"### {r.claim}")
            lines.append(f"**判定**: {emoji} {r.verdict.value}")
            lines.append(f"**理由**: {r.reason}")
            if r.evidence:
                lines.append("**証拠**:")
                for e in r.evidence[:2]:
                    snippet = e[:200].replace("\n", " ")
                    lines.append(f"> {snippet}...")
            lines.append("")

    return "\n".join(lines)


def compute_score(results: list[VerificationResult]) -> float:
    """スコアを 0–100 で返す（公開 API）。"""
    verifiable = [r for r in results if VERDICT_SCORE[r.verdict] is not None]
    if not verifiable:
        return 0.0
    total = sum(VERDICT_SCORE[r.verdict] for r in verifiable)  # type: ignore
    return (total / len(verifiable)) * 100


# 後方互換のためのエイリアス
_compute_score = compute_score


def _count_verdicts(results: list[VerificationResult]) -> dict[Verdict, int]:
    counts = {v: 0 for v in Verdict}
    for r in results:
        counts[r.verdict] += 1
    return counts
