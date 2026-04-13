#!/usr/bin/env python3
"""
refine_loop.py — ファクトチェック自己改善ループ CLI
====================================================
パイプライン:
  1. ファクトチェック（claim 抽出 → BM25 検索 → LLM 検証）
  2. スコア < 閾値なら外科的修正 → 1 に戻る
  3. スコア ≥ 閾値で確定 → --editorial なら編集後記を追加

使い方:
  python refine_loop.py \\
    --truth <transcript.txt または ディレクトリ> \\
    --target <summary_ja.md> \\
    [--threshold 95] \\
    [--max-iter 5] \\
    [--editorial]          # ← 合格後に編集後記を自動追加

オプション:
  --truth        Truth ソース（ファイル or ディレクトリ、複数指定可）
  --target       チェック・修正対象のドキュメント（上書き保存される）
  --threshold    合格スコア（0–100, default: 95）
  --max-iter     最大イテレーション数（default: 5）
  --top-k        各クレームで取得する証拠パッセージ数（default: 5）
  --max-claims   抽出するクレームの最大数（default: 30）
  --output-dir   レポート・中間ファイルの保存先（省略時は --target と同じディレクトリ）
  --backend      ollama | anthropic（default: 環境変数 FACTFULL_LLM_BACKEND）
  --editorial    ループ完了後に編集後記（AI 考察）を末尾に追加する
"""
from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path

from factfull.indexer import build_index
from factfull.claim_extractor import extract
from factfull.retriever import retrieve
from factfull.verifier import verify
from factfull.reporter import generate_report, compute_score
from factfull.corrector import correct
from factfull.editorial import append_editorial_note

SUPPORTED_SUFFIXES = {".txt", ".md", ".rst"}


# ── Truth パス収集 ────────────────────────────────────────────────────────────

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


# ── 1回のファクトチェック ─────────────────────────────────────────────────────

def run_factcheck(document: str, bm25, chunks, top_k: int, max_claims: int):
    """クレーム抽出 → 検証 → VerificationResult のリストを返す。"""
    claims = extract(document, max_claims=max_claims)
    print(f"  抽出クレーム数: {len(claims)}", flush=True)

    results = []
    for i, claim in enumerate(claims, 1):
        print(f"  [{i}/{len(claims)}] {claim[:70]}...", flush=True)
        evidence = retrieve(claim, bm25, chunks, top_k=top_k)
        results.append(verify(claim, evidence))
    return results


# ── メイン ────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="factfull 自己改善ループ: ファクトチェック → 修正 を繰り返す",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--truth", nargs="+", required=True,
                        help="Truth ソース（ファイル or ディレクトリ、複数可）")
    parser.add_argument("--target", required=True,
                        help="チェック・修正対象のドキュメント")
    parser.add_argument("--threshold", type=float, default=95.0,
                        help="合格スコア (default: 95)")
    parser.add_argument("--max-iter", type=int, default=5,
                        help="最大イテレーション数 (default: 5)")
    parser.add_argument("--top-k", type=int, default=5,
                        help="各クレームの証拠パッセージ数 (default: 5)")
    parser.add_argument("--max-claims", type=int, default=30,
                        help="抽出クレームの最大数 (default: 30)")
    parser.add_argument("--output-dir", default=None,
                        help="レポート・中間ファイルの保存先")
    parser.add_argument("--backend", default=None,
                        help="ollama | anthropic (default: FACTFULL_LLM_BACKEND 環境変数)")
    parser.add_argument("--editorial", action="store_true",
                        help="ループ完了後に編集後記（AI 考察）を末尾に追加する")
    args = parser.parse_args()

    # バックエンド設定
    if args.backend:
        os.environ["FACTFULL_LLM_BACKEND"] = args.backend

    target_path = Path(args.target)
    if not target_path.exists():
        print(f"[error] --target が見つかりません: {target_path}", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.output_dir) if args.output_dir else target_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Truth インデックス（1回だけ構築）
    truth_paths = collect_truth_paths(args.truth)
    if not truth_paths:
        print("[error] Truth ソースが見つかりません", file=sys.stderr)
        sys.exit(1)
    print(f"\n📚 Truth ソース: {len(truth_paths)} ファイル", flush=True)
    bm25, chunks = build_index(truth_paths)
    print(f"   チャンク数: {len(chunks)}", flush=True)

    # ── 初期ドキュメント読み込み
    document = target_path.read_text(encoding="utf-8")

    best_score = -1.0
    best_document = document
    final_score = 0.0

    for iteration in range(1, args.max_iter + 1):
        _print_header(f"イテレーション {iteration}/{args.max_iter}")

        # ── ファクトチェック
        print("\n📋 クレーム抽出・検証中...", flush=True)
        results = run_factcheck(document, bm25, chunks, args.top_k, args.max_claims)

        final_score = compute_score(results)
        n_bad = sum(1 for r in results
                    if r.verdict.value in ("contradicted", "partial"))
        n_supported = sum(1 for r in results if r.verdict.value == "supported")
        n_unverifiable = sum(1 for r in results if r.verdict.value == "unverifiable")

        print(
            f"\n📊 スコア: {final_score:.0f}/100"
            f"  (✅{n_supported} ❌{n_bad} ❓{n_unverifiable})",
            flush=True,
        )

        # ── レポート保存
        report = generate_report(
            results,
            target_name=target_path.name,
            truth_names=[p.name for p in truth_paths],
        )
        report_path = output_dir / f"fact_check_iter{iteration:02d}.md"
        report_path.write_text(report, encoding="utf-8")
        print(f"📄 レポート: {report_path}", flush=True)

        # ベストを更新
        if final_score > best_score:
            best_score = final_score
            best_document = document

        # ── 合格判定
        if final_score >= args.threshold:
            print(
                f"\n✅ スコア {final_score:.0f} ≥ {args.threshold:.0f} → 完了！",
                flush=True,
            )
            break

        # 最終イテレーション
        if iteration == args.max_iter:
            print(
                f"\n⚠️  {args.max_iter} 回試行後もスコア {final_score:.0f}"
                f" < {args.threshold:.0f}",
                flush=True,
            )
            print(f"   ベストスコア: {best_score:.0f}", flush=True)
            # ベスト版を採用
            document = best_document
            break

        # ── 修正フェーズ
        print(f"\n✏️  修正中 (問題あり: {n_bad} 件)...", flush=True)
        corrected, n_fixed = correct(document, results)

        if n_fixed == 0:
            print("   修正対象のセクションが特定できませんでした。ループを終了します。",
                  flush=True)
            break

        # 修正版を中間保存
        interim_path = output_dir / f"summary_ja_iter{iteration:02d}.md"
        interim_path.write_text(corrected, encoding="utf-8")
        print(f"💾 中間保存: {interim_path}", flush=True)

        document = corrected

    # ── 編集後記の追加（--editorial フラグがある場合）
    if args.editorial:
        _print_header("編集後記を生成中")
        document = append_editorial_note(document)

    # ── 最終版を target に上書き保存
    target_path.write_text(document, encoding="utf-8")
    print(f"\n💾 最終版を保存: {target_path}", flush=True)

    # ── サマリー
    _print_header("完了")
    print(f"最終スコア: {final_score:.0f}/100", flush=True)
    print(f"ベストスコア: {best_score:.0f}/100", flush=True)
    if final_score >= args.threshold:
        print("🎉 目標スコア達成！", flush=True)
    else:
        print("💡 スコアが目標未達です。プロンプトや --max-iter を見直してください。",
              flush=True)
    if args.editorial:
        print("📝 編集後記を追加済み", flush=True)


def _print_header(text: str) -> None:
    line = "=" * 60
    print(f"\n{line}", flush=True)
    print(f"  {text}", flush=True)
    print(line, flush=True)


if __name__ == "__main__":
    main()
