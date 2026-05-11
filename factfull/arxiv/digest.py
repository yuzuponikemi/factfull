"""
factfull/arxiv/digest.py
=========================
arXiv 論文の日本語要約生成 & ダイジェスト記事テンプレート。

フロー:
  1. 各論文の abstract + conclusion → LLM → 日本語要約 (2-3 文)
  2. 全論文の要約 → ダイジェスト記事 Markdown を生成
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime

from factfull.ingest.arxiv_feed import ArxivEntry


@dataclass
class PaperSummary:
    entry: ArxivEntry
    summary_ja: str          # 2-3文の日本語要約
    contributions_ja: str    # 主な貢献 (箇条書き2-3件)
    keywords: list[str] = field(default_factory=list)


@dataclass
class DigestResult:
    date: str                       # "2026-05-07"
    papers: list[PaperSummary]
    digest_id: str                  # "arxiv_digest_20260507"
    tags: list[str] = field(default_factory=list)
    intro_ja: str = ""


# ── LLM プロンプト ─────────────────────────────────────────────────────────────

_PAPER_SUMMARY_PROMPT = """\
以下の論文情報を日本語で要約してください。

タイトル: {title}
著者: {authors}
カテゴリ: {categories}

Abstract:
{abstract}

Conclusion:
{conclusion}

以下の JSON 形式で回答してください:
{{
  "summary_ja": "論文の内容を2-3文で説明する日本語要約。専門用語は英語のままでよい。",
  "contributions_ja": "主な貢献や手法を箇条書きで2-3件。各項目は「- 」で始める。",
  "keywords": ["キーワード1", "キーワード2", "キーワード3"]
}}
"""

_DIGEST_INTRO_PROMPT = """\
以下の論文リストをもとに、AI研究者・エンジニア向けのダイジェスト記事の導入文を2-3文の日本語で書いてください。
今日のトレンドや注目点を簡潔にまとめてください。

論文リスト:
{paper_list}

JSON で回答:
{{
  "intro_ja": "導入文（2-3文）",
  "tags": ["タグ1", "タグ2", "タグ3", "タグ4", "タグ5"]
}}
"""


# ── LLM 呼び出し ──────────────────────────────────────────────────────────────

def _call_llm(prompt: str, model: str) -> dict:
    import json
    import re
    from factfull.llm import call as llm_call

    raw = llm_call(prompt, model=model, num_ctx=8192)
    try:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        return json.loads(m.group(0)) if m else {}
    except Exception:
        return {}


def summarize_paper(entry: ArxivEntry, model: str = "gemma4:e4b") -> PaperSummary:
    """1論文を LLM で日本語要約する。"""
    prompt = _PAPER_SUMMARY_PROMPT.format(
        title=entry.title,
        authors=", ".join(entry.authors[:5]) + (" et al." if len(entry.authors) > 5 else ""),
        categories=", ".join(entry.categories[:3]),
        abstract=entry.abstract[:1500],
        conclusion=entry.conclusion[:1500] if entry.conclusion else "(not available)",
    )
    result = _call_llm(prompt, model)
    contribs = result.get("contributions_ja", "")
    if isinstance(contribs, list):
        contribs = "\n".join(str(c) for c in contribs)

    return PaperSummary(
        entry=entry,
        summary_ja=result.get("summary_ja", entry.abstract[:200]),
        contributions_ja=contribs,
        keywords=result.get("keywords", []),
    )


def build_digest(
    papers: list[PaperSummary],
    date: str,
    model: str = "gemma4:e4b",
) -> DigestResult:
    """論文リストからダイジェスト全体のメタデータを生成する。"""
    paper_list = "\n".join(
        f"- {p.entry.title} ({', '.join(p.entry.categories[:2])})"
        for p in papers
    )
    prompt = _DIGEST_INTRO_PROMPT.format(paper_list=paper_list)
    result = _call_llm(prompt, model)

    date_compact = date.replace("-", "")
    return DigestResult(
        date=date,
        papers=papers,
        digest_id=f"arxiv_digest_{date_compact}",
        tags=result.get("tags", []),
        intro_ja=result.get("intro_ja", ""),
    )


# ── Markdown 記事生成 ──────────────────────────────────────────────────────────

def render_digest_markdown(digest: DigestResult, kg_src: str) -> str:
    """DigestResult → ダイジェスト記事 Markdown を返す。"""
    date_ja = _date_ja(digest.date)
    tags_yaml = "\n".join(f"  - {t}" for t in digest.tags)

    front = f"""\
---
date: {digest.date}
categories:
  - AI論文
tags:
{tags_yaml}
---

# 今日のAI論文まとめ — {date_ja}

{digest.intro_ja}

<!-- more -->

## 概念グラフ
論文に登場する主要概念と関係をグラフで表示しています。各ノードにどの論文の概念かが付記されています。

<div class="kg-widget" data-src="/data/kg/{kg_src}.json" data-height="630"></div>

---
"""
    paper_sections = []
    for i, ps in enumerate(digest.papers, 1):
        e = ps.entry
        authors_str = ", ".join(e.authors[:3]) + (" et al." if len(e.authors) > 3 else "")
        cats_str = " · ".join(e.categories[:3])
        published_str = e.published.strftime("%Y-%m-%d") if e.published else ""

        section = f"""\
## {i}. {e.title}

**著者**: {authors_str}
**カテゴリ**: {cats_str}
**公開日**: {published_str}
**論文**: [{e.paper_id}]({e.arxiv_url})

{ps.summary_ja}

{ps.contributions_ja}
"""
        paper_sections.append(section)

    footer = "\n---\n*本記事は Abstract と Conclusion から自動生成されています。*\n"
    return front + "\n---\n\n".join(paper_sections) + footer


def _date_ja(date_str: str) -> str:
    """"2026-05-07" → "2026年5月7日" """
    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        return f"{dt.year}年{dt.month}月{dt.day}日"
    except Exception:
        return date_str
