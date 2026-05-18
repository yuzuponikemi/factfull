"""
factfull/publishers/substack.py
================================
Substack 非公式 API クライアント。
生成した記事をドラフトとして Substack に投稿する。

Substack の draft_body は ProseMirror JSON 形式（HTML 不可）。

必要な環境変数（~/.config/factfull/.env で管理）:
    SUBSTACK_PUBLICATION_URL  例: https://ikemix.substack.com
    SUBSTACK_USER_ID          数値のユーザーID
    SUBSTACK_SID              substack.sid クッキー値
    SUBSTACK_LLI              substack.lli クッキー値（JWT）
    SUBSTACK_AWS              AWSALBTG と AWSALBTGCORS の生クッキー文字列
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

import requests


BLOG_BASE_URL = "https://soryu.news"


# ── Markdown → ProseMirror JSON ────────────────────────────────────────────────

def _inline_nodes(text: str) -> list:
    """インライン Markdown（太字・斜体・コード・リンク・改行）をノードリストに変換する。"""
    # 行内改行を hard_break に変換するため行単位で処理
    result: list = []
    lines = text.split("\n")
    for li, line in enumerate(lines):
        if li > 0:
            result.append({"type": "hard_break"})
        if not line:
            continue
        pattern = re.compile(
            r"\*\*(.+?)\*\*"          # bold
            r"|\*(.+?)\*"             # italic
            r"|`(.+?)`"               # code
            r"|\[([^\]]+)\]\(([^)]+)\)"  # link
        )
        pos = 0
        for m in pattern.finditer(line):
            if m.start() > pos:
                result.append({"type": "text", "text": line[pos:m.start()]})
            if m.group(1):
                result.append({"type": "text", "text": m.group(1), "marks": [{"type": "bold"}]})
            elif m.group(2):
                result.append({"type": "text", "text": m.group(2), "marks": [{"type": "italic"}]})
            elif m.group(3):
                result.append({"type": "text", "text": m.group(3), "marks": [{"type": "code"}]})
            elif m.group(4):
                result.append({
                    "type": "text",
                    "text": m.group(4),
                    "marks": [{"type": "link", "attrs": {"href": m.group(5), "target": "_blank"}}],
                })
            pos = m.end()
        if pos < len(line):
            result.append({"type": "text", "text": line[pos:]})
    return result or [{"type": "text", "text": ""}]


def _md_to_prosemirror(text: str) -> str:
    """Markdown テキストを Substack の ProseMirror JSON 文字列に変換する。"""
    nodes: list = []
    blocks = re.split(r"\n\n+", text.strip())

    for block in blocks:
        block = block.strip()
        if not block:
            continue

        # 見出し（## / ### / ####）
        h = re.match(r"^(#{1,4})\s+(.+)$", block)
        if h:
            nodes.append({
                "type": "heading",
                "attrs": {"level": len(h.group(1))},
                "content": _inline_nodes(h.group(2).strip()),
            })
            continue

        # 水平線
        if re.match(r"^-{3,}$", block):
            nodes.append({"type": "horizontal_rule"})
            continue

        # 箇条書き（- item または * item）
        if re.match(r"^[-*] ", block):
            items = []
            for line in block.split("\n"):
                m = re.match(r"^[-*] (.+)$", line)
                if m:
                    items.append({
                        "type": "list_item",
                        "content": [{"type": "paragraph", "content": _inline_nodes(m.group(1))}],
                    })
            if items:
                nodes.append({"type": "bullet_list", "content": items})
            continue

        # 引用ブロック（> で始まる行）
        if block.startswith(">"):
            # 複数段落の引用をサポート（空の > 行で区切る）
            raw_lines = block.split("\n")
            stripped = [re.sub(r"^>[ ]?", "", l) for l in raw_lines]
            inner_text = "\n".join(stripped)
            inner_blocks = re.split(r"\n\n+", inner_text.strip())
            bq_content = []
            for ib in inner_blocks:
                ib = ib.strip()
                if ib:
                    bq_content.append({"type": "paragraph", "content": _inline_nodes(ib)})
            if bq_content:
                nodes.append({"type": "blockquote", "content": bq_content})
            continue

        # 通常段落
        nodes.append({"type": "paragraph", "content": _inline_nodes(block)})

    return json.dumps({"type": "doc", "content": nodes}, ensure_ascii=False)


# ── 前処理 ────────────────────────────────────────────────────────────────────

def _strip_frontmatter(md: str) -> str:
    """YAML フロントマターと Substack 向けに不要な要素を除去する。"""
    md = re.sub(r"^---\n.*?\n---\n", "", md, flags=re.DOTALL)
    md = md.replace("<!-- more -->", "")
    for section in ("概念グラフ", "動画", "キーワード"):
        md = re.sub(rf"## {section}\n.*?(?=\n## |\Z)", "", md, flags=re.DOTALL)
    md = re.sub(r'<div class="kg-widget"[^>]*></div>', "", md)
    md = re.sub(r"_生成条件:.*?_\n?", "", md)
    md = re.sub(r"<iframe[^>]*>.*?</iframe>", "", md, flags=re.DOTALL)
    # 残存 HTML タグを除去
    md = re.sub(r"<[^>]+>", "", md)
    return md.strip()


def _slug_to_url(slug: str) -> str:
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
        cookie = f"substack.sid={sid}; substack.lli={lli}; {aws_cookies}"
        self._headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
            "Referer": self.base_url + "/",
            "Origin": self.base_url,
            "Cookie": cookie,
        }

    def create_draft(self, title: str, subtitle: str, body_json: str) -> dict:
        payload = {
            "draft_title": title,
            "draft_subtitle": subtitle,
            "draft_body": body_json,
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


# ── ドラフト作成 ──────────────────────────────────────────────────────────────

def post_to_draft(client: SubstackClient, post_path: Path) -> dict:
    """homupe の .md ファイルを読んで Substack ドラフトを作成する。"""
    raw = post_path.read_text(encoding="utf-8")
    body_md = _strip_frontmatter(raw)

    # H1 → タイトル（本文から除去）
    title = post_path.stem
    m = re.search(r"^# (.+)$", body_md, re.MULTILINE)
    if m:
        title = m.group(1).strip()
        body_md = body_md[m.end():].strip()

    # 最初の非空段落 → サブタイトル（本文からも除去して重複防止）
    subtitle = ""
    paras = re.split(r"\n\n+", body_md)
    for i, para in enumerate(paras):
        para = para.strip()
        if para and not para.startswith("#"):
            subtitle = para
            paras = paras[i + 1:]
            break
    body_md = "\n\n".join(paras).strip()

    # 末尾に homupe リンクを追加
    homupe_url = _slug_to_url(post_path.stem)
    body_md += f"\n\n---\n\n[層流で読む →]({homupe_url})"

    body_json = _md_to_prosemirror(body_md)
    return client.create_draft(title=title, subtitle=subtitle, body_json=body_json)


def substack_enabled() -> bool:
    return all(
        os.environ.get(k)
        for k in ("SUBSTACK_PUBLICATION_URL", "SUBSTACK_USER_ID", "SUBSTACK_SID", "SUBSTACK_LLI", "SUBSTACK_AWS")
    )
