"""
factfull/publishers/substack.py
================================
Substack 非公式 API クライアント。
生成した記事をドラフトとして Substack に投稿する。

必要な環境変数（~/.config/factfull/.env で管理）:
    SUBSTACK_PUBLICATION_URL  例: https://ikemix.substack.com
    SUBSTACK_USER_ID          数値のユーザーID
    SUBSTACK_SID              substack.sid クッキー値
    SUBSTACK_LLI              substack.lli クッキー値（JWT）
    SUBSTACK_AWS              AWSALBTG と AWSALBTGCORS の生クッキー文字列
"""
from __future__ import annotations

import os
import re
import html as html_lib
from pathlib import Path

import requests


BLOG_BASE_URL = "https://soryu.news"


# ── Markdown → HTML ────────────────────────────────────────────────────────────

def _md_to_html(text: str) -> str:
    """Markdown を HTML に変換する。markdown パッケージがあれば使用、なければ簡易変換。"""
    try:
        import markdown
        return markdown.markdown(text, extensions=["extra", "nl2br"])
    except ImportError:
        # フォールバック: 段落・見出し・太字のみ
        lines = text.strip().split("\n")
        out = []
        for line in lines:
            line = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", line)
            line = re.sub(r"\*(.+?)\*", r"<em>\1</em>", line)
            if line.startswith("## "):
                out.append(f"<h2>{line[3:]}</h2>")
            elif line.startswith("### "):
                out.append(f"<h3>{line[4:]}</h3>")
            elif line.startswith("# "):
                out.append(f"<h1>{line[2:]}</h1>")
            elif line.startswith("- "):
                out.append(f"<li>{line[2:]}</li>")
            elif line.strip() == "":
                out.append("")
            else:
                out.append(f"<p>{line}</p>")
        return "\n".join(out)


def _strip_frontmatter(md: str) -> str:
    """YAML フロントマターと MkDocs 固有の要素を除去する。"""
    # YAML フロントマター除去
    md = re.sub(r"^---\n.*?\n---\n", "", md, flags=re.DOTALL)
    # <!-- more --> 除去
    md = md.replace("<!-- more -->", "")
    # 概念グラフセクション全体を除去（見出し＋説明文＋ウィジェット）
    md = re.sub(r"## 概念グラフ\n.*?(?=\n## |\Z)", "", md, flags=re.DOTALL)
    # MkDocs の KG ウィジェット除去（念のため残存分も除去）
    md = re.sub(r'<div class="kg-widget"[^>]*></div>', "", md)
    # iframe → YouTube リンクに変換
    md = re.sub(
        r'<iframe[^>]+src="https://www\.youtube\.com/embed/([^"]+)"[^>]*>.*?</iframe>',
        lambda m: f"[▶ YouTube で見る](https://www.youtube.com/watch?v={m.group(1).split('?')[0]})",
        md,
        flags=re.DOTALL,
    )
    # その他の iframe は除去
    md = re.sub(r"<iframe[^>]*>.*?</iframe>", "", md, flags=re.DOTALL)
    return md.strip()


def _slug_to_url(slug: str) -> str:
    """'2026-05-18-some-title' → 'https://soryu.news/blog/2026/05/2026-05-18-some-title/'"""
    m = re.match(r"(\d{4})-(\d{2})-\d{2}-.+", slug)
    if not m:
        return BLOG_BASE_URL
    year, month = m.group(1), m.group(2)
    return f"{BLOG_BASE_URL}/blog/{year}/{month}/{slug}/"


# ── Substack API クライアント ──────────────────────────────────────────────────

class SubstackClient:
    def __init__(
        self,
        publication_url: str,
        user_id: int,
        sid: str,
        lli: str,
        aws_cookies: str,
    ) -> None:
        self.base_url = publication_url.rstrip("/")
        self.user_id = user_id
        self._cookie_header = f"substack.sid={sid}; substack.lli={lli}; {aws_cookies}"
        self._headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
            "Referer": self.base_url + "/",
            "Origin": self.base_url,
            "Cookie": self._cookie_header,
        }

    def create_draft(self, title: str, subtitle: str, body_html: str) -> dict:
        payload = {
            "draft_title": title,
            "draft_subtitle": subtitle,
            "draft_body": body_html,
            "audience": "everyone",
            "draft_section_id": None,
            "draft_podcast_url": None,
            "draft_podcast_duration": None,
            "draft_bylines": [{"id": self.user_id, "is_guest": False}],
        }
        resp = requests.post(
            f"{self.base_url}/api/v1/drafts",
            headers=self._headers,
            json=payload,
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()

    @classmethod
    def from_env(cls) -> "SubstackClient":
        return cls(
            publication_url=os.environ["SUBSTACK_PUBLICATION_URL"],
            user_id=int(os.environ["SUBSTACK_USER_ID"]),
            sid=os.environ["SUBSTACK_SID"],
            lli=os.environ["SUBSTACK_LLI"],
            aws_cookies=os.environ["SUBSTACK_AWS"],
        )


# ── ドラフト作成ヘルパー ───────────────────────────────────────────────────────

def create_podcast_draft(
    client: SubstackClient,
    title_ja: str,
    excerpt: str,
    post_path: Path,
) -> dict:
    """Podcast ブログ記事ファイルから Substack ドラフトを作成する。

    Args:
        client: SubstackClient インスタンス
        title_ja: 日本語タイトル
        excerpt: 抜粋（サブタイトルとして使用）
        post_path: homupe に書き出した .md ファイルのパス

    Returns:
        Substack API のレスポンス dict
    """
    raw_md = post_path.read_text(encoding="utf-8")
    body_md = _strip_frontmatter(raw_md)

    slug = post_path.stem
    homupe_url = _slug_to_url(slug)
    body_md += f"\n\n---\n\n[層流で読む →]({homupe_url})"

    body_html = _md_to_html(body_md)
    return client.create_draft(
        title=title_ja,
        subtitle=excerpt,
        body_html=body_html,
    )


def create_arxiv_draft(
    client: SubstackClient,
    post_path: Path,
    date_str: str,
) -> dict:
    """arXiv ダイジェスト記事ファイルから Substack ドラフトを作成する。

    Args:
        client: SubstackClient インスタンス
        post_path: homupe に書き出した .md ファイルのパス
        date_str: "2026-05-18" 形式の日付

    Returns:
        Substack API のレスポンス dict
    """
    raw_md = post_path.read_text(encoding="utf-8")
    body_md = _strip_frontmatter(raw_md)

    slug = post_path.stem
    homupe_url = _slug_to_url(slug)
    body_md += f"\n\n---\n\n[層流で読む →]({homupe_url})"

    body_html = _md_to_html(body_md)
    return client.create_draft(
        title=f"arXiv AI 論文ダイジェスト {date_str}",
        subtitle="cs.AI / cs.LG / cs.CL の注目論文まとめ",
        body_html=body_html,
    )


def post_to_draft(client: SubstackClient, post_path: Path) -> dict:
    """homupe の .md ファイルを読んで Substack ドラフトを作成する。

    フロントマター・<!-- more -->・KG ウィジェットを除去し、
    H1 見出しをタイトル、直後の段落をサブタイトルとして使用する。

    Args:
        client: SubstackClient インスタンス
        post_path: homupe/docs/blog/posts/... の .md ファイルパス

    Returns:
        Substack API のレスポンス dict
    """
    raw = post_path.read_text(encoding="utf-8")
    body_md = _strip_frontmatter(raw)

    # H1 をタイトルとして抽出・除去
    title = post_path.stem  # フォールバック
    m = re.search(r"^# (.+)$", body_md, re.MULTILINE)
    if m:
        title = m.group(1).strip()
        body_md = body_md[m.end():].strip()

    # 最初の非空段落をサブタイトルとして抽出
    subtitle = ""
    for para in re.split(r"\n\n+", body_md):
        para = para.strip()
        if para and not para.startswith("#") and not para.startswith("<"):
            subtitle = para
            break

    # homupe リンクを末尾に追加
    slug = post_path.stem
    homupe_url = _slug_to_url(slug)
    body_md += f"\n\n---\n\n[層流で読む →]({homupe_url})"

    body_html = _md_to_html(body_md)
    return client.create_draft(title=title, subtitle=subtitle, body_html=body_html)


def substack_enabled() -> bool:
    """必要な環境変数がすべて設定されているか確認する。"""
    return all(
        os.environ.get(k)
        for k in ("SUBSTACK_PUBLICATION_URL", "SUBSTACK_USER_ID", "SUBSTACK_SID", "SUBSTACK_LLI", "SUBSTACK_AWS")
    )
