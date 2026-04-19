"""
factfull/podcast/pipeline.py
=============================
ポッドキャスト翻訳記事パイプラインの共通ライブラリ。

pipelines/lex.py や pipelines/dwarkesh.py から以下の形で使う:

    from factfull.podcast.pipeline import PipelineConfig, run_pipeline

    config = PipelineConfig(analyze_model="gemma4:26b", ...)
    run_pipeline(config, youtube_url, regen=args.regen)
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path


# ── 結果 ─────────────────────────────────────────────────────────────────────

@dataclass
class PipelineResult:
    """
    run_pipeline() の戻り値。
    上位パイプライン（ブログ投稿・SNS 配信など）が必要な情報をすべて含む。
    """
    video_id: str
    title: str          # YouTube の英語タイトル
    channel: str        # チャンネル名
    summary_path: Path  # summary_ja.md のパス
    episode_dir: Path   # エピソードディレクトリ
    score: float        # ファクトチェックスコア (0–100)
    metadata: dict      # metadata.json の全内容


# ── 設定 ─────────────────────────────────────────────────────────────────────

@dataclass
class PipelineConfig:
    # --- モデル ---
    translate_model: str = "translategemma:12b"
    analyze_model: str = "gemma4:26b"
    factcheck_model: str = "gemma4:e4b"
    editorial_model: str | None = None  # None のとき factcheck_model を使用

    # --- チャンクサイズ ---
    translate_chunk_size: int = 6000   # Step 3 翻訳チャンク（英語文字数）
    summary_chunk_size: int = 5000     # Pass 1 要約チャンク（日本語文字数）

    # --- ファクトチェックループ ---
    threshold: float = 95.0
    max_iter: int = 5
    max_claims: int = 50
    top_k: int = 5

    # --- 機能フラグ ---
    editorial: bool = True
    fetch_comments: bool = False

    # --- 出力先 ---
    output_base: Path = field(default_factory=lambda: Path.home() / "podcasts")

    # --- コンテンツ設定 ---
    blog_name: str = "SoryuNews"
    reader_persona: str = "英語圏情報にアクセスしたい日本語話者のエンジニア・研究者"
    n_questions: int = 4   # 「問いとして残るもの」の合計問い数（前半・後半で均等分割）


# ── エントリポイント ───────────────────────────────────────────────────────────

def run_pipeline(
    config: PipelineConfig,
    youtube_url: str,
    regen: bool = False,
) -> PipelineResult:
    """
    エンドツーエンドパイプライン:
      Step 1: メタデータ取得
      Step 2: 英語トランスクリプト取得
      Step 3: 日本語翻訳
      Step 4: 日本語記事生成（Map-Reduce）
        Pass 1  : チャンク別要点抽出
        Pass 1.5: 英語引用抽出
        Pass 2a/b: 論点生成（前半・後半）
        Pass 2c : 概要・注目発言・キーワード
        Pass 2d : 問いとして残るもの
      Step 5: ファクトチェック + 修正ループ
      Step 6: 編集後記追加（config.editorial=True の場合）

    regen=True のとき、同じ video_id の既存ディレクトリに
    section_summaries.json があれば Pass 1 をスキップする。

    Returns:
        PipelineResult — 上位パイプラインが必要な情報をすべて含む
    """
    from factfull.podcast.archiver import PodcastArchiver

    # factcheck フェーズで使うモデルを env var で設定（llm.py が実行時に再読み取り）
    os.environ["FACTFULL_LLM_BACKEND"] = "ollama"
    os.environ["FACTFULL_OLLAMA_MODEL"] = config.factcheck_model

    arch = PodcastArchiver(youtube_url=youtube_url, config=config)

    skip_pass1 = False
    if regen:
        existing = _find_existing_dir(config.output_base, arch.video_id)
        if existing:
            print(f"  [regen] 既存ディレクトリを使用: {existing.name}")
            arch.out_dir = existing
            _load_existing_data(arch, existing)
            skip_pass1 = (existing / "section_summaries.json").exists()
            if not skip_pass1:
                print("  [regen] section_summaries.json なし → Pass 1 から実行")
        else:
            print("  [regen] 既存ディレクトリが見つかりません → フルで実行")
            regen = False

    if not regen or not _has_transcript(arch):
        arch.fetch_metadata()
        arch.fetch_transcript()
        arch.translate_to_japanese()

    arch.generate_summary(skip_pass1=skip_pass1)

    if config.fetch_comments:
        arch.fetch_comments()
        arch.summarize_comments()

    summary_path = arch.out_dir / "summary_ja.md"
    score = _run_factcheck_loop(config, summary_path, arch.out_dir)

    return PipelineResult(
        video_id=arch.video_id,
        title=arch.metadata.get("title", ""),
        channel=arch.metadata.get("channel", ""),
        summary_path=summary_path,
        episode_dir=arch.out_dir,
        score=score,
        metadata=arch.metadata,
    )


# ── ヘルパー ──────────────────────────────────────────────────────────────────

def _find_existing_dir(output_base: Path, video_id: str) -> Path | None:
    """video_id に対応する既存エピソードディレクトリを返す（最新優先）。"""
    candidates = sorted(output_base.glob(f"{video_id}_*"), reverse=True)
    for d in candidates:
        if (d / "transcript_en.txt").exists():
            return d
    return None


def _has_transcript(arch) -> bool:
    return bool(arch.transcript_en) or (arch.out_dir / "transcript_en.txt").exists()


def _load_existing_data(arch, ep_dir: Path) -> None:
    """既存ディレクトリのデータをアーカイバのメモリにロードする。"""
    meta_path = ep_dir / "metadata.json"
    if meta_path.exists():
        arch.metadata = json.loads(meta_path.read_text(encoding="utf-8"))

    en_path = ep_dir / "transcript_en.txt"
    if en_path.exists():
        arch.transcript_en = en_path.read_text(encoding="utf-8")

    ja_path = ep_dir / "transcript_ja.txt"
    if ja_path.exists():
        arch.transcript_ja = ja_path.read_text(encoding="utf-8")

    ts_path = ep_dir / "transcript_en_timestamped.json"
    if ts_path.exists():
        arch.transcript_raw = json.loads(ts_path.read_text(encoding="utf-8"))


def _run_factcheck_loop(
    config: PipelineConfig,
    summary_path: Path,
    output_dir: Path,
) -> float:
    """
    refine_loop と同等のファクトチェック + 修正ループ。
    summary_path を上書き保存して最終スコアを返す。
    """
    from factfull.indexer import build_index
    from factfull.claim_extractor import extract
    from factfull.retriever import retrieve
    from factfull.verifier import verify
    from factfull.reporter import generate_report, compute_score
    from factfull.corrector import correct
    from factfull.editorial import append_editorial_note

    os.environ["FACTFULL_OLLAMA_MODEL"] = config.factcheck_model

    truth_path = output_dir / "transcript_en.txt"
    if not truth_path.exists():
        print("[warn] transcript_en.txt が見つかりません。ファクトチェックをスキップ。")
        return 0.0

    _header("ファクトチェック開始")
    bm25, chunks = build_index([truth_path])
    print(f"  チャンク数: {len(chunks)}", flush=True)

    document = summary_path.read_text(encoding="utf-8")
    best_score = -1.0
    best_document = document
    final_score = 0.0

    for iteration in range(1, config.max_iter + 1):
        _header(f"イテレーション {iteration}/{config.max_iter}")

        # クレーム抽出 → 検証
        print("\n📋 クレーム抽出・検証中...", flush=True)
        claims = extract(document, max_claims=config.max_claims)
        print(f"  抽出クレーム数: {len(claims)}", flush=True)

        results = []
        for i, claim in enumerate(claims, 1):
            print(f"  [{i}/{len(claims)}] {claim[:70]}...", flush=True)
            evidence = retrieve(claim, bm25, chunks, top_k=config.top_k)
            results.append(verify(claim, evidence))

        final_score = compute_score(results)
        n_bad = sum(1 for r in results if r.verdict.value in ("contradicted", "partial"))
        n_ok = sum(1 for r in results if r.verdict.value == "supported")
        n_unk = sum(1 for r in results if r.verdict.value == "unverifiable")
        print(f"\n📊 スコア: {final_score:.0f}/100  (✅{n_ok} ❌{n_bad} ❓{n_unk})", flush=True)

        # レポート保存
        report = generate_report(
            results,
            target_name=summary_path.name,
            truth_names=[truth_path.name],
        )
        report_path = output_dir / f"fact_check_iter{iteration:02d}.md"
        report_path.write_text(report, encoding="utf-8")
        print(f"📄 レポート: {report_path.name}", flush=True)

        if final_score > best_score:
            best_score = final_score
            best_document = document

        if final_score >= config.threshold:
            print(f"\n✅ スコア {final_score:.0f} ≥ {config.threshold:.0f} → 完了！", flush=True)
            break

        if iteration == config.max_iter:
            print(
                f"\n⚠️  {config.max_iter} 回試行後もスコア {final_score:.0f} < {config.threshold:.0f}",
                flush=True,
            )
            print(f"   ベストスコア: {best_score:.0f}", flush=True)
            document = best_document
            break

        # 修正フェーズ
        print(f"\n✏️  修正中 (問題あり: {n_bad} 件)...", flush=True)
        corrected, n_fixed = correct(document, results)
        if n_fixed == 0:
            print("   修正対象のセクションが特定できません。ループを終了。", flush=True)
            break
        interim_path = output_dir / f"summary_ja_iter{iteration:02d}.md"
        interim_path.write_text(corrected, encoding="utf-8")
        print(f"💾 中間保存: {interim_path.name}", flush=True)
        document = corrected

    # 編集後記
    if config.editorial:
        _header("編集後記を生成中")
        editorial_model = config.editorial_model or config.factcheck_model
        os.environ["FACTFULL_OLLAMA_MODEL"] = editorial_model
        document = append_editorial_note(document)

    # スコアメタデータを更新
    if "スコア: TBD" in document:
        document = document.replace("スコア: TBD", f"スコア: {best_score:.0f}/100")
        print(f"📋 スコアを更新: {best_score:.0f}/100", flush=True)

    summary_path.write_text(document, encoding="utf-8")
    print(f"\n💾 最終版を保存: {summary_path}", flush=True)
    _header("完了")
    print(f"最終スコア: {final_score:.0f}/100  ベスト: {best_score:.0f}/100", flush=True)

    return best_score


def _header(text: str) -> None:
    line = "=" * 60
    print(f"\n{line}", flush=True)
    print(f"  {text}", flush=True)
    print(line, flush=True)
