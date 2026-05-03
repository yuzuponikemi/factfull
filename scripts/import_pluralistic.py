#!/usr/bin/env python3
"""
Pluralistic (Cory Doctorow) の RSS から記事とリンク先を
factfull レジストリに一括登録する。

使い方:
    uv run scripts/import_pluralistic.py             # 直近7日
    uv run scripts/import_pluralistic.py --days 14   # 直近14日
    uv run scripts/import_pluralistic.py --dry-run   # 登録せず確認のみ
"""
import argparse
import os


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=7, help="取得する日数（デフォルト: 7）")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    os.environ.setdefault("NEO4J_URI", "bolt://localhost:7687")
    os.environ.setdefault("NEO4J_USER", "neo4j")
    os.environ.setdefault("NEO4J_PASSWORD", "factfull123")

    print(f"Pluralistic.net — 直近 {args.days} 日分を取得中...\n")

    from factfull.ingest.pluralistic import collect
    entries = collect(days=args.days)

    print(f"\n合計: {len(entries)} 件（記事 + リンク先）\n")

    if args.dry_run:
        for e in entries:
            print(f"  [web] {e['source_id'][:80]}")
        return

    from factfull.registry import Registry
    added, skipped = 0, 0
    with Registry() as reg:
        for e in entries:
            if reg.add(e["source_type"], e["source_id"], title=e.get("title", "")):
                added += 1
            else:
                skipped += 1

    print(f"追加: {added} 件 / スキップ（既存）: {skipped} 件")
    print("→ 'uv run scripts/batch_process.py run --type web --graph-only' で処理開始")


if __name__ == "__main__":
    main()
