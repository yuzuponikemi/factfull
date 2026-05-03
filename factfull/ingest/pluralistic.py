"""
factfull/ingest/pluralistic.py
================================
Pluralistic (Cory Doctorow) の RSS フィードから記事と重要リンクを取得する。

フロー:
  1. RSS フィードから N 日分の記事 URL を取得
  2. 各記事を取得して本文テキストと外部リンクを抽出
  3. リンクをフィルタリングして「重要リンク」のみ残す
  4. (source_type, source_id) のリストを返す → Registry に一括登録

重要リンク判定:
  - 除外: archive.org, SNS, Amazon, 購入サイト, イベント, self-link
  - 残す: ニュース/ブログ/Substack/学術/政策 サイト
"""
from __future__ import annotations

import re
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
from datetime import datetime, timezone, timedelta
from typing import Any

RSS_URL = "https://pluralistic.net/feed/"

# ── ノイズドメインのプレフィックス ──────────────────────────────────────────────
_NOISE_DOMAINS = {
    # archive
    "web.archive.org", "archive.org",
    # social / microblogging
    "twitter.com", "x.com", "mastodon.social", "facebook.com",
    "instagram.com", "linkedin.com", "threads.net",
    "mamot.fr", "bsky.app", "medium.com",
    # video
    "youtube.com", "youtu.be", "vimeo.com",
    # book retail / publishers
    "amazon.com", "amzn.to", "bookshop.org", "indiebound.org",
    "us.macmillan.com", "macmillan.com", "harpercollins.com",
    "strandbooks.com", "booksoup.com", "penguinrandomhouse.com",
    "simonandschuster.com", "powells.com",
    # events / ticketing
    "eventbrite.com", "eventbrite.co.uk", "meetup.com",
    "luma.com", "tickettailor.com", "sxswlondon.com",
    "eventbrite.ca",
    # payments / crowdfunding
    "paypal.com", "patreon.com", "ko-fi.com",
    # CDN / infrastructure
    "c0.wp.com", "i0.wp.com", "i1.wp.com", "i2.wp.com",
    "creativecommons.org", "wordpress.org", "wordpress.com",
    "deflect.ca",
    # misc noise
    "xkcd.com", "flickr.com", "tumblr.com", "deviantart.com",
    "gmpg.org",           # XML namespace
    "omny.fm",            # podcast hosting
    "ipetitions.com",     # petitions
    # fundraising
    "actblue.com", "secure.actblue.com",
    # events
    "re-publica.com", "otherland-berlin.de", "howthelightgetsin.org",
    "encuentroderechosdigitales.com", "sxsw.com", "votingvillage.org",
    # Doctorow's book promo sites (not news/analysis)
    "thebezzle.org", "lost-cause.org", "seizethemeansofcomputation.org",
    "redteamblues.com", "chokepointcapitalism.com",
    # self-links
    "pluralistic.net",
    "craphound.com",
    "boingboing.net",
}

_NOISE_PATTERNS = [
    r"^/",              # relative paths
    r"^mailto:",
    r"^#",
    r"\.(jpg|jpeg|png|gif|pdf|mp3|mp4|zip|css|js|woff|woff2|svg|ico)$",
    r"/wp-content/",    # WordPress assets
    r"/wp-includes/",
]


def fetch_rss(days: int = 7) -> list[dict[str, str]]:
    """RSS フィードから直近 days 日分の記事を返す。

    Returns: [{"url": ..., "title": ..., "date": ...}, ...]
    """
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    req = urllib.request.Request(RSS_URL, headers={"User-Agent": "factfull/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        xml_bytes = resp.read()

    root = ET.fromstring(xml_bytes)
    ns = {"content": "http://purl.org/rss/1.0/modules/content/"}
    channel = root.find("channel")
    articles = []

    for item in channel.findall("item"):
        link  = (item.findtext("link") or "").strip()
        title = (item.findtext("title") or "").strip()
        pub   = item.findtext("pubDate") or ""
        try:
            # RFC 2822: "Mon, 21 Apr 2026 00:00:00 +0000"
            from email.utils import parsedate_to_datetime
            dt = parsedate_to_datetime(pub)
            if dt < cutoff:
                continue
        except Exception:
            pass
        articles.append({"url": link, "title": title, "date": pub[:16]})

    return articles


def extract_links(article_url: str) -> list[str]:
    """記事 URL をフェッチして重要な外部リンクを返す。"""
    try:
        req = urllib.request.Request(
            article_url,
            headers={"User-Agent": "factfull/1.0"},
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            html = resp.read().decode("utf-8", errors="replace")
    except Exception as e:
        print(f"  [pluralistic] fetch error: {e}", flush=True)
        return []

    # href 抽出（簡易）
    hrefs = re.findall(r'href=["\']([^"\']+)["\']', html, re.IGNORECASE)
    seen: set[str] = set()
    result: list[str] = []

    for href in hrefs:
        href = href.strip()
        if not href or href in seen:
            continue
        # ノイズパターン除外
        if any(re.search(p, href, re.IGNORECASE) for p in _NOISE_PATTERNS):
            continue
        # 絶対 URL に正規化
        if not href.startswith("http"):
            href = urllib.parse.urljoin(article_url, href)
        parsed = urllib.parse.urlparse(href)
        netloc = parsed.netloc.lower()
        domain = netloc[4:] if netloc.startswith("www.") else netloc
        # ノイズドメイン除外
        if any(domain == nd or domain.endswith("." + nd) for nd in _NOISE_DOMAINS):
            continue
        # クエリなしの URL に正規化（追跡パラメータ除去）
        clean = urllib.parse.urlunparse(parsed._replace(query="", fragment=""))
        if clean in seen:
            continue
        seen.add(clean)
        result.append(clean)

    return result


def collect(days: int = 7) -> list[dict[str, str]]:
    """RSS + リンク抽出をまとめて実行。

    Returns:
        [{"source_type": "web", "source_id": url, "title": title}, ...]
        記事本体 + 重要リンク先をフラットに返す。
    """
    articles = fetch_rss(days=days)
    print(f"  [pluralistic] {len(articles)} 記事 ({days}日分)", flush=True)

    entries: list[dict[str, str]] = []
    seen_ids: set[str] = set()

    def _add(url: str, title: str = "") -> None:
        if url not in seen_ids:
            seen_ids.add(url)
            entries.append({"source_type": "web", "source_id": url, "title": title})

    for art in articles:
        _add(art["url"], art["title"])
        print(f"  [{art['date']}] {art['title'][:55]}", flush=True)

        links = extract_links(art["url"])
        print(f"    → 重要リンク: {len(links)} 件", flush=True)
        for link in links:
            _add(link)

    return entries
