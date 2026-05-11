"""
factfull/ingest/arxiv_feed.py
==============================
arXiv 新着論文フィードを取得し、Registry 未登録の論文を返す。

- arxiv ライブラリでカテゴリ別最新論文を取得
- abstract は arxiv API から直接取得
- conclusion は arxiv HTML 版 (arxiv.org/html/{id}) から trafilatura で抽出
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from factfull.registry import Registry


@dataclass
class ArxivEntry:
    paper_id: str           # "2506.12345"
    title: str
    abstract: str
    authors: list[str]
    categories: list[str]
    published: datetime
    arxiv_url: str          # https://arxiv.org/abs/2506.12345
    conclusion: str = ""    # fetch_conclusion() で補完


def fetch_recent_papers(
    categories: list[str],
    max_per_category: int = 30,
    lookback_days: int = 1,
) -> list[ArxivEntry]:
    """arXiv API でカテゴリ別最新論文を取得する（重複除去・新しい順）。"""
    import arxiv

    cutoff = (
        datetime.now(timezone.utc) - timedelta(days=lookback_days)
        if lookback_days > 0
        else None
    )

    papers: dict[str, ArxivEntry] = {}
    client = arxiv.Client()

    for cat in categories:
        search = arxiv.Search(
            query=f"cat:{cat}",
            max_results=max_per_category,
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Descending,
        )
        try:
            for result in client.results(search):
                # paper_id: "2506.12345v2" → "2506.12345"
                paper_id = re.sub(r"v\d+$", "", result.entry_id.split("/abs/")[-1])

                if cutoff and result.published and result.published < cutoff:
                    continue
                if paper_id in papers:
                    continue

                papers[paper_id] = ArxivEntry(
                    paper_id=paper_id,
                    title=result.title.replace("\n", " ").strip(),
                    abstract=result.summary.replace("\n", " ").strip(),
                    authors=[a.name for a in result.authors],
                    categories=result.categories,
                    published=result.published or datetime.now(timezone.utc),
                    arxiv_url=f"https://arxiv.org/abs/{paper_id}",
                )
        except Exception as e:
            print(f"  [arxiv] {cat} 取得エラー: {e}")

    return sorted(papers.values(), key=lambda p: p.published, reverse=True)


def fetch_conclusion(paper_id: str, timeout: int = 20) -> str:
    """arXiv HTML 版から Conclusion セクションを抽出する。

    arxiv.org/html/{id} を trafilatura で取得し、
    "Conclusion" 見出し以降のテキストを返す（最大3000文字）。
    """
    import trafilatura

    url = f"https://arxiv.org/html/{paper_id}"
    try:
        html = trafilatura.fetch_url(url, config=trafilatura.settings.use_config())
        if not html:
            return ""
        text = trafilatura.extract(html, include_links=False, include_tables=False) or ""
    except Exception:
        return ""

    if not text:
        return ""

    lines = text.splitlines()
    in_conclusion = False
    buf: list[str] = []

    for line in lines:
        stripped = line.strip()
        if not in_conclusion:
            # "5 Conclusion", "Conclusion", "Conclusions", "Concluding Remarks" などにマッチ
            if re.match(
                r"^(\d+\.?\s+)?[Cc]onclu(sion|ding|de)", stripped
            ):
                in_conclusion = True
        else:
            # 次の番号付きセクションが来たら終了
            if re.match(r"^\d+\.?\s+[A-Z]", stripped) and stripped.lower() not in (
                "conclusion", "conclusions"
            ):
                break
            buf.append(line)

    conclusion = "\n".join(buf).strip()
    if not conclusion:
        # フォールバック: テキスト後半 30% を使う
        start = max(0, int(len(text) * 0.7))
        conclusion = text[start:]

    return conclusion[:3000]


def find_new_papers(
    categories: list[str],
    registry: "Registry",
    papers_per_digest: int = 5,
    lookback_days: int = 1,
    max_per_category: int = 30,
) -> list[ArxivEntry]:
    """新着論文のうち Registry 未登録のものを papers_per_digest 件返す。"""
    all_papers = fetch_recent_papers(
        categories, max_per_category=max_per_category, lookback_days=lookback_days
    )
    new: list[ArxivEntry] = []
    for paper in all_papers:
        if not registry.exists("arxiv", paper.paper_id):
            new.append(paper)
            if len(new) >= papers_per_digest:
                break
    return new
