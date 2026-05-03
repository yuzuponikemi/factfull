#!/usr/bin/env python3
"""
homupe の podcast-episodes.md からエピソードを読み込み
factfull レジストリに一括登録する。

使い方:
    uv run scripts/import_homupe.py
    uv run scripts/import_homupe.py --dry-run   # 登録せず一覧表示のみ
    uv run scripts/import_homupe.py --episodes-md /path/to/podcast-episodes.md
"""
import argparse
import re
from pathlib import Path

DEFAULT_MD = Path(__file__).parent.parent.parent / "homupe" / "docs" / "podcast-episodes.md"
YT_ID_RE = re.compile(r"youtube\.com/watch\?v=([\w-]+)")
SECTION_RE = re.compile(r"^##\s+(.+)")


def extract_episodes(md_path: Path, channel: str | None = None) -> list[dict]:
    """Markdown テーブルから (video_id, title, date, channel) を抽出する。

    channel: "lex" | "dwarkesh" | None（全件）
    """
    episodes = []
    current_section = ""
    for line in md_path.read_text(encoding="utf-8").splitlines():
        sec_m = SECTION_RE.match(line)
        if sec_m:
            current_section = sec_m.group(1)
            continue

        m = YT_ID_RE.search(line)
        if not m:
            continue

        # チャンネルフィルタ
        section_lower = current_section.lower()
        if channel == "lex" and "lex" not in section_lower:
            continue
        if channel == "dwarkesh" and "dwarkesh" not in section_lower:
            continue

        video_id = m.group(1)
        title_m = re.search(r"\[([^\]]+)\]\(https://www\.youtube", line)
        title = title_m.group(1) if title_m else ""
        date_m = re.match(r"\|\s*(\d{4}-\d{2}-\d{2})\s*\|", line)
        date = date_m.group(1) if date_m else ""

        episodes.append({
            "video_id": video_id,
            "title": title,
            "date": date,
            "channel": current_section,
        })
    return episodes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes-md", type=Path, default=DEFAULT_MD)
    parser.add_argument("--channel", choices=["lex", "dwarkesh"], help="チャンネルを絞り込む")
    parser.add_argument("--dry-run", action="store_true", help="登録せず表示のみ")
    args = parser.parse_args()

    if not args.episodes_md.exists():
        print(f"ファイルが見つかりません: {args.episodes_md}")
        return

    episodes = extract_episodes(args.episodes_md, channel=args.channel)
    print(f"抽出: {len(episodes)} 件\n")

    if args.dry_run:
        for ep in episodes:
            print(f"  [{ep['date']}] {ep['video_id']}  {ep['title'][:60]}")
        return

    from factfull.registry import Registry
    added, skipped = 0, 0
    with Registry() as reg:
        for ep in episodes:
            if reg.add("podcast", ep["video_id"], title=ep["title"]):
                print(f"  ✅ {ep['video_id']}  {ep['title'][:55]}")
                added += 1
            else:
                status = (reg.get("podcast", ep["video_id"]) or {}).get("status", "?")
                print(f"  ⏭  {ep['video_id']}  [{status}]  {ep['title'][:45]}")
                skipped += 1

    print(f"\n追加: {added} 件 / スキップ: {skipped} 件")
    print("→ 'uv run scripts/batch_process.py run --type podcast' で処理開始")


if __name__ == "__main__":
    main()
