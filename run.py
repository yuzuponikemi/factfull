#!/usr/bin/env python3
"""
factfull CLI

使い方:
  python run.py --truth <truth_dir_or_file> [--truth ...] --target <target.md> [--output report.md]

オプション:
  --truth   Truth ソース（ディレクトリまたはファイルを複数指定可）
  --target  チェック対象のドキュメント
  --output  レポート出力先（省略時は stdout）
  --top-k   各クレームで取得する証拠パッセージ数 (default: 5)
  --max-claims  抽出するクレームの最大数 (default: 30)
  --backend ollama | anthropic (default: 環境変数 FACTFULL_LLM_BACKEND)
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

from factfull.indexer import build_index
from factfull.claim_extractor import extract
from factfull.retriever import retrieve
from factfull.verifier import verify
from factfull.reporter import generate_report


SUPPORTED_SUFFIXES = {".txt", ".md", ".rst"}


def collect_truth_paths(sources: list[str]) -> list[Path]:
    paths: list[Path] = []
    for src in sources:
        p = Path(src)
        if p.is_dir():
            for f in sorted(p.rglob("*")):
                if f.is_file() and f.suffix.lower() in SUPPORTED_SUFFIXES:
                    paths.append(f)
        elif p.is_file():
            paths.append(p)
        else:
            print(f"[warn] {src} が見つかりません", file=sys.stderr, flush=True)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="factfull: claim-based fact checker")
    parser.add_argument("--truth", nargs="+", required=True, help="Truth ソース（ディレクトリ or ファイル）")
    parser.add_argument("--target", required=True, help="チェック対象ドキュメント")
    parser.add_argument("--output", default=None, help="レポート出力先ファイル（省略時 stdout）")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-claims", type=int, default=30)
    args = parser.parse_args()

    # 1. Truth ソースをインデックス化
    truth_paths = collect_truth_paths(args.truth)
    if not truth_paths:
        print("[error] Truth ソースが見つかりません", file=sys.stderr, flush=True)
        sys.exit(1)
    print(f"[1/4] Truth ソース: {len(truth_paths)} ファイル → インデックス構築中...", file=sys.stderr, flush=True)
    bm25, chunks = build_index(truth_paths)
    print(f"      チャンク数: {len(chunks)}", file=sys.stderr, flush=True)

    # 2. チェック対象を読み込む
    target_path = Path(args.target)
    target_text = target_path.read_text(encoding="utf-8")

    # 3. クレーム抽出
    print(f"[2/4] クレーム抽出中...", file=sys.stderr, flush=True)
    claims = extract(target_text, max_claims=args.max_claims)
    print(f"      抽出クレーム数: {len(claims)}", file=sys.stderr, flush=True)

    # 4. クレームごとに検証
    print(f"[3/4] {len(claims)} クレームを検証中...", file=sys.stderr, flush=True)
    results = []
    for i, claim in enumerate(claims, 1):
        print(f"      {i}/{len(claims)}: {claim[:60]}...", file=sys.stderr, flush=True)
        evidence = retrieve(claim, bm25, chunks, top_k=args.top_k)
        result = verify(claim, evidence)
        results.append(result)
    print("", file=sys.stderr, flush=True)

    # 5. レポート生成
    print("[4/4] レポート生成中...", file=sys.stderr, flush=True)
    report = generate_report(
        results,
        target_name=target_path.name,
        truth_names=[p.name for p in truth_paths],
    )

    if args.output:
        Path(args.output).write_text(report, encoding="utf-8")
        print(f"[done] レポートを保存しました: {args.output}", file=sys.stderr, flush=True)
    else:
        print(report)


if __name__ == "__main__":
    main()
