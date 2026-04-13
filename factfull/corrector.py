"""
factfull/corrector.py
======================
CONTRADICTED / PARTIAL なクレームを証拠に基づいて記事内で外科的に修正する。

方針: セクション単位の修正
1. ## / ### 見出しで記事をセクションに分割
2. 各クレームに最も関連するセクションを特定（文字 n-gram 重複スコア）
3. 同じセクションに複数のクレームがある場合はまとめて渡す
4. LLM にセクション本文のみの修正を依頼
5. 修正済みセクションで元のセクションを置換した文書を返す
"""
from __future__ import annotations
import re
from collections import defaultdict
from .verifier import VerificationResult, Verdict
from . import llm


# ── パブリック API ────────────────────────────────────────────────────────────

def correct(
    document: str,
    results: list[VerificationResult],
) -> tuple[str, int]:
    """
    CONTRADICTED / PARTIAL なクレームをセクション単位で修正した文書を返す。

    Returns:
        (corrected_document, n_fixed)
        n_fixed: 実際に修正を試みたセクション数（0 のときは元の文書をそのまま返す）
    """
    bad = [r for r in results if r.verdict in (Verdict.CONTRADICTED, Verdict.PARTIAL)]
    if not bad:
        return document, 0

    sections = _split_sections(document)

    # claim → セクションインデックス のマッピング
    headers = [h for h, _ in sections]
    bodies = [b for _, b in sections]
    section_to_claims: defaultdict[int, list[VerificationResult]] = defaultdict(list)

    for result in bad:
        idx = _find_best_section(result.claim, headers, bodies)
        if idx is not None:
            section_to_claims[idx].append(result)
        else:
            print(
                f"  [corrector] ⚠️  セクション特定失敗: {result.claim[:60]}",
                flush=True,
            )

    if not section_to_claims:
        return document, 0

    # セクションを修正
    modified_bodies = list(bodies)
    for idx, claims_in_section in sorted(section_to_claims.items()):
        header = headers[idx]
        body = bodies[idx]
        print(
            f"  [corrector] 修正: {header[:50]} ({len(claims_in_section)} 件)",
            flush=True,
        )
        modified_bodies[idx] = _rewrite_section(header, body, claims_in_section)

    # 文書を再構成（プリアンブルは別扱い）
    parts: list[str] = []
    for i, (header, _) in enumerate(sections):
        if header == "__preamble__":
            parts.append(modified_bodies[i])
        else:
            parts.append(f"{header}\n{modified_bodies[i]}")

    return "\n\n".join(parts), len(section_to_claims)


# ── 内部実装 ──────────────────────────────────────────────────────────────────

def _split_sections(text: str) -> list[tuple[str, str]]:
    """
    ## / ### 見出しで文書をセクションに分割する。
    YAML フロントマター（--- ... ---）は preamble として扱う。

    Returns:
        [(header, body), ...]
        プリアンブルは header = "__preamble__" で返す。
    """
    # YAML フロントマターを除去してプリアンブルに
    fm_match = re.match(r'^---\s*\n.*?\n---\s*\n', text, re.DOTALL)
    preamble_extra = ""
    if fm_match:
        preamble_extra = fm_match.group(0)
        text = text[fm_match.end():]

    header_pattern = re.compile(r'^(#{1,3} .+)$', re.MULTILINE)
    matches = list(header_pattern.finditer(text))

    sections: list[tuple[str, str]] = []

    # 最初の見出し前のテキスト（＋フロントマター）
    leading = (preamble_extra + text[: matches[0].start()]).strip() if matches else text.strip()
    if leading:
        sections.append(("__preamble__", leading))

    if not matches:
        return sections

    for i, m in enumerate(matches):
        header = m.group(0)
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        sections.append((header, body))

    return sections


# 修正対象から除外するセクションの見出しキーワード
_SKIP_SECTION_KEYWORDS = frozenset({
    "動画", "キーワード", "チャンネル", "YouTube",
    "再生時間", "外部で開く",
})


def _find_best_section(
    claim: str,
    headers: list[str],
    bodies: list[str],
) -> int | None:
    """
    claim に最も関連するセクションのインデックスを返す。
    信頼できるマッチがない場合は None。
    """
    # 日本語・英数字の混在に対応したトークナイズ
    claim_tokens = _tokenize(claim)
    if not claim_tokens:
        return None

    best_score = 0.0
    best_idx: int | None = None

    for i, (header, body) in enumerate(zip(headers, bodies)):
        if header == "__preamble__":
            continue
        if any(kw in header for kw in _SKIP_SECTION_KEYWORDS):
            continue
        if not body:
            continue

        section_tokens = _tokenize(header + " " + body)
        # Jaccard 係数（クレームトークン ∩ セクショントークン）/ クレームトークン
        overlap = len(claim_tokens & section_tokens)
        score = overlap / len(claim_tokens)

        if score > best_score:
            best_score = score
            best_idx = i

    # 重複率が低すぎる場合は None（誤マッチ防止）
    return best_idx if best_score >= 0.15 else None


def _tokenize(text: str) -> set[str]:
    """日本語・英語混在の簡易トークナイザ（set を返す）。"""
    tokens = re.findall(r'[A-Za-z0-9]+|[\u3040-\u9fff\u30a0-\u30ff]+', text)
    result: set[str] = set()
    for tok in tokens:
        if re.match(r'^[A-Za-z0-9]+$', tok):
            result.add(tok.lower())
        else:
            # 日本語は 2-gram
            for j in range(len(tok) - 1):
                result.add(tok[j: j + 2])
    return result


_REWRITE_PROMPT = """\
あなたは事実に忠実な編集者です。以下のセクションに {n_issues} 件の事実誤りが含まれています。
各誤りを証拠パッセージの情報に基づいて修正し、修正後のセクション本文のみを出力してください。

⚠️ 厳守ルール:
- 指摘された誤りの箇所のみを修正し、それ以外の文章は一字一句変えない
- マークダウン形式・箇条書き構造・見出しレベルは元のまま維持する
- 証拠パッセージにない情報を追加・補完しない
- 見出し行（# で始まる行）は出力しない（本文のみを出力）
- 修正前後を比較したコメントや説明は書かない

## セクション見出し
{header}

## 現在のセクション本文
{body}

## 修正が必要な箇所と証拠

{issues}

## 修正後のセクション本文（本文のみ。前置き・説明・見出し不要）
"""

_ISSUE_BLOCK = """\
### 誤り {n}
クレーム: {claim}
判定: {verdict} — {reason}
証拠パッセージ:
{evidence}
"""


def _rewrite_section(
    header: str,
    body: str,
    claims: list[VerificationResult],
) -> str:
    issues_text = "\n".join(
        _ISSUE_BLOCK.format(
            n=i + 1,
            claim=r.claim,
            verdict=r.verdict.value,
            reason=r.reason,
            evidence="\n".join(f"> {e[:300]}" for e in r.evidence[:3]),
        )
        for i, r in enumerate(claims)
    )

    prompt = _REWRITE_PROMPT.format(
        n_issues=len(claims),
        header=header,
        body=body,
        issues=issues_text,
    )

    result = llm.call(prompt, num_ctx=16384).strip()
    if not result:
        print(f"  [corrector] ⚠️  LLM が空を返したため元の本文を維持", flush=True)
        return body
    return result
