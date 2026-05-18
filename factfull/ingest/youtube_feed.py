"""
factfull/ingest/youtube_feed.py
================================
YouTube チャンネル RSS フィードから新着エピソードを検出する。

feedparser を使用（API キー不要）。
各チャンネルの RSS は最新 15 件を返すため、daily cron での利用を前提とする。
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from factfull.registry import Registry

_RSS_BASE = "https://www.youtube.com/feeds/videos.xml?channel_id={channel_id}"


@dataclass
class FeedEntry:
    video_id: str
    title: str
    published: datetime
    channel_name: str


def fetch_channel_entries(
    channel_id: str,
    channel_name: str = "",
    lookback_days: int = 30,
) -> list[FeedEntry]:
    """
    YouTube RSS フィードから FeedEntry リストを返す。

    Args:
        channel_id: YouTube チャンネル ID (UC...)
        channel_name: ログ表示用チャンネル名
        lookback_days: この日数より古いエントリは除外。0 = 制限なし
    """
    import feedparser  # type: ignore

    url = _RSS_BASE.format(channel_id=channel_id)
    feed = feedparser.parse(url)

    if not feed.entries:
        print(f"  [feed] {channel_name or channel_id}: エントリなし（アクセス失敗の可能性）")
        return []

    cutoff = (
        datetime.now(timezone.utc) - timedelta(days=lookback_days)
        if lookback_days > 0
        else None
    )

    entries: list[FeedEntry] = []
    for e in feed.entries:
        video_id = _extract_video_id(e)
        if not video_id:
            continue

        published = _parse_published(e)
        if cutoff and published and published < cutoff:
            continue

        entries.append(FeedEntry(
            video_id=video_id,
            title=e.get("title", ""),
            published=published or datetime.now(timezone.utc),
            channel_name=channel_name or feed.feed.get("title", channel_id),
        ))

    return entries


def find_new_entries(
    channel_id: str,
    channel_name: str,
    registry: "Registry",
    lookback_days: int = 30,
    max_new: int = 3,
) -> list[FeedEntry]:
    """
    Registry に登録されていない新着エントリのみ返す。

    Args:
        channel_id: YouTube チャンネル ID
        channel_name: チャンネル名
        registry: Registry インスタンス
        lookback_days: 古いエントリを除外する日数
        max_new: 返す最大件数
    """
    entries = fetch_channel_entries(channel_id, channel_name, lookback_days)
    new: list[FeedEntry] = []
    for entry in entries:
        if not registry.exists("podcast", entry.video_id):
            new.append(entry)
            if len(new) >= max_new:
                break
    return new


def get_video_duration_seconds(video_id: str) -> int | None:
    """YouTube 動画ページから再生時間（秒）を取得する。取得失敗時は None を返す。"""
    import re
    import urllib.request

    url = f"https://www.youtube.com/watch?v={video_id}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            html = r.read().decode("utf-8")
    except Exception:
        return None

    m = re.search(r'"lengthSeconds":"(\d+)"', html)
    return int(m.group(1)) if m else None


def _extract_video_id(entry) -> str | None:
    """feedparser エントリから YouTube video_id を抽出する。"""
    # yt:videoId タグが最も確実
    vid = entry.get("yt_videoid") or entry.get("media_videoid")
    if vid:
        return vid.strip()

    # link から抽出
    link = entry.get("link", "")
    if "v=" in link:
        return link.split("v=")[-1].split("&")[0].strip()

    return None


def _parse_published(entry) -> datetime | None:
    """feedparser エントリの published_parsed を UTC datetime に変換する。"""
    import time as _time
    pp = entry.get("published_parsed")
    if pp:
        return datetime.fromtimestamp(_time.mktime(pp), tz=timezone.utc)
    return None
