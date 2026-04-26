"""
factfull/podcast/steps/factcheck.py
-------------------------------------
Step 5-6: ファクトチェックループ + 批評 + 編集後記
"""
from __future__ import annotations

import os
from pathlib import Path


def run_factcheck(result, config) -> float:
    """
    ファクトチェックループを実行して summary_ja.md を上書き保存し、
    最終スコアを返す。
    """
    return _factcheck_loop(config, result.summary_path, result.episode_dir)


def _factcheck_loop(config, summary_path: Path, output_dir: Path) -> float:
    from factfull.indexer import build_index
    from factfull.claim_extractor import extract
    from factfull.retriever import retrieve
    from factfull.verifier import verify
    from factfull.reporter import generate_report, compute_score
    from factfull.corrector import correct

    os.environ["FACTFULL_OLLAMA_MODEL"] = config.factcheck_model

    truth_path = output_dir / "transcript_en.txt"
    if not truth_path.exists():
        print("[warn] transcript_en.txt が見つかりません。ファクトチェックをスキップ。")
        return 0.0

    _header("ファクトチェック開始")
    bm25, chunks = build_index([truth_path])
    print(f"  チャンク数: {len(chunks)}", flush=True)

    document = summary_path.read_text(encoding="utf-8")
    best_score, best_document, final_score = -1.0, document, 0.0

    for iteration in range(1, config.max_iter + 1):
        _header(f"イテレーション {iteration}/{config.max_iter}")

        claims = extract(document, max_claims=config.max_claims)
        print(f"  抽出クレーム数: {len(claims)}", flush=True)

        results = []
        for i, claim in enumerate(claims, 1):
            print(f"  [{i}/{len(claims)}] {claim[:70]}...", flush=True)
            evidence = retrieve(claim, bm25, chunks, top_k=config.top_k)
            results.append(verify(claim, evidence))

        final_score = compute_score(results)
        n_bad = sum(1 for r in results if r.verdict.value in ("contradicted", "partial"))
        n_ok  = sum(1 for r in results if r.verdict.value == "supported")
        n_unk = sum(1 for r in results if r.verdict.value == "unverifiable")
        print(f"\n📊 スコア: {final_score:.0f}/100  (✅{n_ok} ❌{n_bad} ❓{n_unk})", flush=True)

        report = generate_report(results, target_name=summary_path.name, truth_names=[truth_path.name])
        report_path = output_dir / f"fact_check_iter{iteration:02d}.md"
        report_path.write_text(report, encoding="utf-8")
        print(f"📄 レポート: {report_path.name}", flush=True)

        if final_score > best_score:
            best_score, best_document = final_score, document

        if final_score >= config.threshold:
            print(f"\n✅ スコア {final_score:.0f} ≥ {config.threshold:.0f} → 完了！", flush=True)
            break

        if iteration == config.max_iter:
            print(f"\n⚠️  {config.max_iter} 回試行後もスコア {final_score:.0f} < {config.threshold:.0f}", flush=True)
            document = best_document
            break

        print(f"\n✏️  修正中 (問題あり: {n_bad} 件)...", flush=True)
        corrected, n_fixed = correct(document, results)
        if n_fixed == 0:
            print("   修正対象のセクションが特定できません。ループを終了。", flush=True)
            break
        interim_path = output_dir / f"summary_ja_iter{iteration:02d}.md"
        interim_path.write_text(corrected, encoding="utf-8")
        document = corrected

    editorial_model = config.editorial_model or config.factcheck_model
    os.environ["FACTFULL_OLLAMA_MODEL"] = editorial_model

    if config.critique:
        _header("批評的読みを生成中")
        from factfull.critique import append_critique
        document = append_critique(document)

    if config.editorial:
        _header("編集後記を生成中")
        from factfull.editorial import append_editorial_note
        document = append_editorial_note(document)

    if "スコア: TBD" in document:
        document = document.replace("スコア: TBD", f"スコア: {best_score:.0f}/100")

    summary_path.write_text(document, encoding="utf-8")
    print(f"\n💾 最終版を保存: {summary_path}", flush=True)
    _header("完了")
    print(f"最終スコア: {final_score:.0f}/100  ベスト: {best_score:.0f}/100", flush=True)
    return best_score


def _header(text: str) -> None:
    line = "=" * 60
    print(f"\n{line}\n  {text}\n{line}", flush=True)
