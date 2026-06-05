#!/usr/bin/env python3
"""
書籍章立て解決ツール
====================
複数の Web ソース（Wikipedia EN/RU/ZH/JA、Open Library 等）から
書籍の正確な章立てを取得し、consensus で検証する。

使い方:

  # 基本（タイトル + 著者）
  python pipelines/book_chapters.py "La Société de consommation" "Jean Baudrillard"

  # 日本語タイトルでも可
  python pipelines/book_chapters.py "消費社会の神話と構造" "ボードリヤール"

  # 既存の book_guide.md の章テーブルを置換
  python pipelines/book_chapters.py "La Société de consommation" "Baudrillard" \\
      --fix-guide ~/book-research/data/run_20260510_072055/book_guide.md

  # 追加 URL を指定（出版社サイト等）
  python pipelines/book_chapters.py "The Society of the Spectacle" "Guy Debord" \\
      --extra-urls "https://en.wikipedia.org/wiki/The_Society_of_the_Spectacle"

  # JSON 出力
  python pipelines/book_chapters.py "Thus Spoke Zarathustra" "Nietzsche" --json
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path

# .env ロード（TAVILY_API_KEY 等）— factfull/.env → book-research/.env の順で探す
_REPO_ROOT = Path(__file__).parent.parent
for _env_path in (_REPO_ROOT / ".env", _REPO_ROOT.parent / "book-research" / ".env"):
    if _env_path.exists():
        for _line in _env_path.read_text(encoding="utf-8").splitlines():
            _line = _line.strip()
            if not _line or _line.startswith("#") or "=" not in _line:
                continue
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())
        break


def main() -> None:
    parser = argparse.ArgumentParser(
        description="書籍章立て解決ツール",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("book_title", help="書籍タイトル（原題 or 翻訳）")
    parser.add_argument("author", help="著者名")
    parser.add_argument(
        "--model", default="gemma4:e4b",
        help="Ollama モデル（デフォルト: gemma4:e4b）",
    )
    parser.add_argument(
        "--extra-urls", nargs="+", default=[],
        metavar="URL",
        help="追加で参照する URL",
    )
    parser.add_argument(
        "--fix-guide",
        metavar="GUIDE_PATH",
        help="指定した book_guide.md の章テーブルを置換する",
    )
    parser.add_argument(
        "--json", action="store_true", dest="json_output",
        help="JSON 形式で出力",
    )
    parser.add_argument(
        "--min-confidence", type=float, default=0.0,
        help="この confidence 未満の場合は非ゼロ終了（デフォルト: 0.0 = 常に成功）",
    )
    parser.add_argument(
        "--translate-to",
        metavar="LANG",
        default=None,
        help="章タイトルを翻訳する言語コード（例: ja, en, fr）",
    )
    args = parser.parse_args()

    from factfull.book.chapter_resolver import ChapterResolver

    resolver = ChapterResolver(model=args.model)
    result = resolver.resolve(
        args.book_title, args.author,
        extra_urls=args.extra_urls or None,
        translate_to=args.translate_to,
    )

    # ── 出力 ──────────────────────────────────────────────────────────────────

    if args.json_output:
        print(json.dumps({
            "book_title": result.book_title,
            "author": result.author,
            "confidence": result.confidence,
            "sources_agreed": result.sources_agreed,
            "is_reliable": result.is_reliable(),
            "sources_consulted": result.sources_consulted,
            "chapters": [
                {"num": ch.num, "title": ch.title, "part": ch.part}
                for ch in result.chapters
            ],
        }, ensure_ascii=False, indent=2))
    else:
        print(f"\n{'='*60}")
        print(f"  書籍: {result.book_title}")
        print(f"  著者: {result.author}")
        print(f"  信頼度: {result.confidence:.2f}  |  合意ソース数: {result.sources_agreed}")
        print(f"  信頼できる: {'✅ YES' if result.is_reliable() else '⚠️  NO（ソース不足または不一致）'}")
        print(f"{'='*60}")

        if result.chapters:
            print("\n  章立て:")
            for ch in result.chapters:
                part_label = f"  [{ch.part}]" if ch.part else ""
                num_label = f"[{ch.num}] " if ch.num else "- "
                print(f"    {num_label}{ch.title}{part_label}")
        else:
            print("\n  ⚠️  章立てが取得できませんでした")

        if result.notes:
            print(f"\n  Note: {result.notes}")

        print(f"\n  参照ソース:")
        for s in result.sources_consulted:
            print(f"    {s}")

    # ── book_guide.md の修正 ──────────────────────────────────────────────────

    if args.fix_guide and result.chapters:
        guide_path = Path(args.fix_guide)
        if not guide_path.exists():
            print(f"\n⚠️  ガイドファイルが見つかりません: {guide_path}", file=sys.stderr)
            sys.exit(1)

        if not result.is_reliable():
            print(
                f"\n⚠️  confidence={result.confidence:.2f} / sources_agreed={result.sources_agreed} — "
                f"信頼度が低いため --fix-guide をスキップします。",
                file=sys.stderr,
            )
            sys.exit(1)

        original = guide_path.read_text(encoding="utf-8")
        fixed = _replace_chapter_table(original, result.chapters)

        if fixed == original:
            print("\n  [fix] 既存の章テーブルと一致 — 変更なし")
        else:
            guide_path.write_text(fixed, encoding="utf-8")
            print(f"\n✅ ガイドの章テーブルを更新しました: {guide_path}")
            print("   変更内容を確認してください。")

    # ── 終了コード ────────────────────────────────────────────────────────────

    if args.min_confidence > 0 and result.confidence < args.min_confidence:
        print(
            f"\n⚠️  confidence {result.confidence:.2f} < {args.min_confidence}",
            file=sys.stderr,
        )
        sys.exit(1)


def _replace_chapter_table(guide_text: str, chapters) -> str:
    """
    book_guide.md の「## 本書の構成と読解フェーズ」テーブルを
    解決済み章立てで置換する。

    テーブルが見つからない場合は先頭に章立てセクションを挿入する。
    """
    new_table = _build_chapter_table(chapters)

    # 既存のテーブルを探して置換
    # パターン: Markdown テーブルヘッダー行を含む2行以上のブロック
    table_pattern = re.compile(
        r"(## 本書の構成[^\n]*\n+)"   # セクションヘッダー
        r"(\|[^\n]+\|\n"               # ヘッダー行
        r"\|[-| :]+\|\n"               # セパレーター行
        r"(?:\|[^\n]+\|\n)*)",         # データ行
        re.MULTILINE,
    )
    m = table_pattern.search(guide_text)
    if m:
        replacement = m.group(1) + new_table + "\n"
        return guide_text[: m.start(2)] + new_table + "\n" + guide_text[m.end():]

    # テーブルが見つからない場合: 最初の ## セクションの前に挿入
    first_section = re.search(r"\n## ", guide_text)
    if first_section:
        insert_pos = first_section.start()
        section = (
            "\n\n## 本書の構成（検証済み章立て）\n\n"
            + new_table
            + "\n"
        )
        return guide_text[:insert_pos] + section + guide_text[insert_pos:]

    return guide_text + "\n\n## 本書の構成（検証済み章立て）\n\n" + new_table + "\n"


def _build_chapter_table(chapters) -> str:
    """ChapterEntry リストから Markdown テーブルを生成する。"""
    rows = ["| # | タイトル | 部・篇 |", "| :- | :- | :- |"]
    for ch in chapters:
        rows.append(f"| {ch.num} | {ch.title} | {ch.part} |")
    return "\n".join(rows)


if __name__ == "__main__":
    main()
