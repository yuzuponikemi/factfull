"""
factfull/ingest/rss_downloader.py
===================================
RSS フィード（または Apple Podcasts ID）から全エピソード音声をダウンロードする。

使い方:
    # Apple Podcasts ID を指定
    uv run python -m factfull.ingest.rss_downloader \\
        --apple-id 1444665909 --output ~/podcasts/off_topic

    # RSS URL を直接指定
    uv run python -m factfull.ingest.rss_downloader \\
        --rss https://example.com/feed.xml --output ~/podcasts/my_podcast

出力レイアウト:
    output/
    ├── feed_metadata.json     # フィード全体のメタデータ
    ├── index.json             # エピソード一覧（status: done/pending/failed）
    └── audio/
        ├── 2018-11-23_ep001_intro.mp3
        ├── 2018-11-23_ep001_intro.json
        └── ...
"""
from __future__ import annotations

import json
import re
import time
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from xml.etree import ElementTree as ET


@dataclass
class EpisodeEntry:
    ep_num: int             # 連番 (RSS 順。0 = 番号不明)
    pub_date: str           # "2026-05-06"
    title: str
    audio_url: str
    duration: str           # "HH:MM:SS" or ""
    description: str        # HTML 含む可能性あり
    status: str = "pending" # pending / done / failed
    file_path: str = ""     # ダウンロード済みの場合のパス
    error: str = ""


def get_rss_url(apple_id: str) -> str:
    """Apple Podcasts ID から RSS フィード URL を取得する。"""
    import urllib.request
    url = f"https://itunes.apple.com/lookup?id={apple_id}"
    with urllib.request.urlopen(url, timeout=15) as r:
        data = json.loads(r.read())
    results = data.get("results", [])
    if not results:
        raise ValueError(f"Apple Podcasts ID {apple_id} が見つかりません")
    feed_url = results[0].get("feedUrl", "")
    if not feed_url:
        raise ValueError(f"feedUrl が取得できませんでした: {results[0]}")
    return feed_url


def parse_rss(rss_url: str) -> tuple[dict, list[EpisodeEntry]]:
    """RSS フィードを取得してフィードメタデータとエピソードリストを返す。"""
    import urllib.request
    with urllib.request.urlopen(rss_url, timeout=30) as r:
        xml_data = r.read()

    root = ET.fromstring(xml_data)
    channel = root.find("channel")
    if channel is None:
        raise ValueError("RSS に <channel> が見つかりません")

    feed_meta = {
        "title":       channel.findtext("title", ""),
        "description": channel.findtext("description", ""),
        "link":        channel.findtext("link", ""),
        "rss_url":     rss_url,
    }

    items = channel.findall("item")
    entries: list[EpisodeEntry] = []

    for i, item in enumerate(reversed(items)):  # 古い順に番号付け
        title   = item.findtext("title", "").strip()
        pub_raw = item.findtext("pubDate", "")
        pub_date = _parse_date(pub_raw)

        enc = item.find("enclosure")
        audio_url = enc.get("url", "") if enc is not None else ""

        dur_el = item.find("{http://www.itunes.com/dtds/podcast-1.0.dtd}duration")
        duration = dur_el.text.strip() if dur_el is not None and dur_el.text else ""

        desc = item.findtext("description", "") or \
               (item.findtext("{http://www.itunes.com/dtds/podcast-1.0.dtd}summary", "") or "")

        entries.append(EpisodeEntry(
            ep_num=i + 1,
            pub_date=pub_date,
            title=title,
            audio_url=audio_url,
            duration=duration,
            description=desc[:500],
        ))

    return feed_meta, entries


def download_all(
    entries: list[EpisodeEntry],
    audio_dir: Path,
    index_path: Path,
    max_episodes: int = 0,   # 0 = 全件
    delay: float = 0.5,      # ダウンロード間隔（秒）
) -> dict[str, int]:
    """全エピソードをダウンロードする。既存ファイルはスキップ。"""
    audio_dir.mkdir(parents=True, exist_ok=True)

    to_download = entries if not max_episodes else entries[:max_episodes]
    stats = {"done": 0, "skipped": 0, "failed": 0}

    for ep in to_download:
        if not ep.audio_url:
            print(f"  [{ep.ep_num:03d}] スキップ（URL なし）: {ep.title[:50]}")
            ep.status = "failed"
            ep.error = "no audio url"
            stats["failed"] += 1
            continue

        ext = _guess_ext(ep.audio_url)
        filename = _safe_filename(ep.pub_date, ep.ep_num, ep.title, ext)
        out_path = audio_dir / filename

        if out_path.exists() and out_path.stat().st_size > 0:
            ep.status = "done"
            ep.file_path = str(out_path)
            stats["skipped"] += 1
            print(f"  [{ep.ep_num:03d}] 既存スキップ: {filename}")
            continue

        print(f"  [{ep.ep_num:03d}] ダウンロード中: {ep.title[:50]}...", end="", flush=True)
        try:
            _download_file(ep.audio_url, out_path)
            size_mb = out_path.stat().st_size / 1024 / 1024
            print(f" {size_mb:.1f}MB ✓")

            # メタデータ JSON を隣に保存
            meta_path = out_path.with_suffix(".json")
            meta_path.write_text(
                json.dumps(asdict(ep), ensure_ascii=False, indent=2), encoding="utf-8"
            )

            ep.status = "done"
            ep.file_path = str(out_path)
            stats["done"] += 1

            # index を都度更新（中断しても再開できるように）
            _save_index(index_path, entries)

            time.sleep(delay)

        except Exception as e:
            print(f" ERROR: {e}")
            ep.status = "failed"
            ep.error = str(e)[:200]
            stats["failed"] += 1

    _save_index(index_path, entries)
    return stats


def _download_file(url: str, out_path: Path, timeout: int = 60) -> None:
    """HTTP GET でファイルをダウンロードする（リダイレクト追従）。"""
    import urllib.request
    req = urllib.request.Request(url, headers={"User-Agent": "factfull-podcast-archiver/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r, open(out_path, "wb") as f:
        while chunk := r.read(65536):
            f.write(chunk)


def _save_index(path: Path, entries: list[EpisodeEntry]) -> None:
    path.write_text(
        json.dumps([asdict(e) for e in entries], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _parse_date(pub_raw: str) -> str:
    """'Wed, 06 May 2026 ...' → '2026-05-06'"""
    months = {"Jan":"01","Feb":"02","Mar":"03","Apr":"04","May":"05","Jun":"06",
              "Jul":"07","Aug":"08","Sep":"09","Oct":"10","Nov":"11","Dec":"12"}
    m = re.search(r"(\d{1,2})\s+([A-Z][a-z]{2})\s+(\d{4})", pub_raw)
    if m:
        d, mon, y = m.groups()
        return f"{y}-{months.get(mon, '00')}-{int(d):02d}"
    return pub_raw[:10]


def _guess_ext(url: str) -> str:
    u = url.split("?")[0].lower()
    for ext in (".mp3", ".m4a", ".ogg", ".aac", ".wav"):
        if u.endswith(ext):
            return ext
    return ".mp3"


def _safe_filename(date: str, num: int, title: str, ext: str) -> str:
    slug = unicodedata.normalize("NFKC", title.lower())
    slug = re.sub(r"[^\w\s-]", "", slug)
    slug = re.sub(r"[\s_-]+", "_", slug.strip())[:60]
    return f"{date}_ep{num:03d}_{slug}{ext}"


# ── CLI ───────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="RSS Podcast Downloader")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--apple-id", metavar="ID", help="Apple Podcasts の数字 ID")
    src.add_argument("--rss",      metavar="URL", help="RSS フィード URL を直接指定")
    parser.add_argument("--output", required=True, metavar="DIR", help="出力ディレクトリ")
    parser.add_argument("--max",    type=int, default=0, help="ダウンロード上限（0=全件）")
    parser.add_argument("--delay",  type=float, default=0.3, help="DL間隔秒（default: 0.3）")
    parser.add_argument("--list",   action="store_true", help="一覧表示のみ（DL しない）")
    args = parser.parse_args(argv)

    out_dir = Path(args.output).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    # RSS URL 解決
    if args.apple_id:
        print(f"iTunes API から RSS を取得中... (id={args.apple_id})")
        rss_url = get_rss_url(args.apple_id)
        print(f"  → {rss_url}")
    else:
        rss_url = args.rss

    # フィード解析
    print("RSS を解析中...")
    feed_meta, entries = parse_rss(rss_url)
    print(f"  フィード: {feed_meta['title']}")
    print(f"  エピソード数: {len(entries)}")

    # フィードメタデータ保存
    meta_path = out_dir / "feed_metadata.json"
    meta_path.write_text(json.dumps(feed_meta, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.list:
        for ep in entries:
            print(f"  [{ep.ep_num:03d}] {ep.pub_date}  {ep.title[:60]}")
        return

    # ダウンロード
    audio_dir  = out_dir / "audio"
    index_path = out_dir / "index.json"
    n = args.max or len(entries)
    print(f"\n{n} 件をダウンロード開始 → {audio_dir}\n")

    stats = download_all(entries, audio_dir, index_path, max_episodes=args.max, delay=args.delay)

    print(f"\n完了: ✓{stats['done']} skipped={stats['skipped']} failed={stats['failed']}")
    print(f"出力: {out_dir}")


if __name__ == "__main__":
    main()
