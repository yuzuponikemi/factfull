#!/usr/bin/env python3
"""
バッチ処理: 複数ソースをレジストリ管理しながら一括処理する。

使い方:
    # キューに追加
    uv run scripts/batch_process.py add podcast Q8Fkpi18QXU
    uv run scripts/batch_process.py add paper 1706.03762
    uv run scripts/batch_process.py add web https://example.com/article

    # ファイルから一括追加（1行1エントリ: "type source_id" 形式）
    uv run scripts/batch_process.py add-file queue.txt

    # 未処理を全件実行
    uv run scripts/batch_process.py run

    # 特定種別だけ実行
    uv run scripts/batch_process.py run --type podcast

    # グラフ専用モード（翻訳・要約・ファクトチェックをスキップ）
    uv run scripts/batch_process.py run --graph-only
    uv run scripts/batch_process.py run --type podcast --graph-only

    # 失敗分をリトライ
    uv run scripts/batch_process.py retry

    # 一覧・統計表示
    uv run scripts/batch_process.py status
    uv run scripts/batch_process.py list [--type podcast] [--status done]
"""
import argparse
import os
import sys
from pathlib import Path

MODEL = "gemma4:e4b"


def _setup_env() -> None:
    os.environ.setdefault("FACTFULL_OLLAMA_MODEL", MODEL)
    os.environ.setdefault("FACTFULL_LLM_BACKEND", "ollama")
    os.environ.setdefault("NEO4J_URI", "bolt://localhost:7687")
    os.environ.setdefault("NEO4J_USER", "neo4j")
    os.environ.setdefault("NEO4J_PASSWORD", "factfull123")


# ── 各ソース種別の処理ロジック ────────────────────────────────────────────────

def _process_podcast(source_id: str) -> str:
    """YouTube video_id → フルパイプライン（要約・ファクトチェック込み）。"""
    from factfull.podcast.pipeline import PipelineConfig, run_pipeline
    url = f"https://www.youtube.com/watch?v={source_id}"
    config = PipelineConfig(
        translate_model=MODEL,
        analyze_model=MODEL,
        factcheck_model=MODEL,
        write_graph=True,
        critique=False,
        editorial=False,
    )
    result = run_pipeline(config, url, regen=True)
    return result.title


def _process_podcast_graph_only(source_id: str) -> str:
    """YouTube video_id → トランスクリプト取得 + 話者分離 + エンティティ/トリプル抽出 + Neo4j書き込み。

    HF_TOKEN が設定されていれば話者分離（pyannote）を実行し、
    [Speaker] prefix 付きトランスクリプトをエンティティ抽出に使う。
    """
    import json as _json
    from pathlib import Path as _Path
    from factfull.podcast.archiver import PodcastArchiver
    from factfull.ingest.chunker import chunk_text
    from factfull.core.types import SourceDoc
    from factfull.extract.entity import extract_entities
    from factfull.extract.relation import extract_relations
    from factfull.graph.neo4j import Neo4jClient

    url = f"https://www.youtube.com/watch?v={source_id}"
    arch = PodcastArchiver(url)

    # 既存ディレクトリがあればトランスクリプト・メタデータを再利用
    en_path = arch.out_dir / "transcript_en.txt"
    if en_path.exists():
        print(f"  [regen] トランスクリプト再利用: {arch.out_dir.name}", flush=True)
        arch.transcript_en = en_path.read_text(encoding="utf-8")
        meta_path = arch.out_dir / "metadata.json"
        if meta_path.exists():
            arch.metadata = _json.loads(meta_path.read_text(encoding="utf-8"))
    else:
        arch.fetch_metadata()
        arch.fetch_transcript()

    title   = arch.metadata.get("title", source_id)
    channel = arch.metadata.get("channel", "")

    # ── 話者分離（HF_TOKEN があれば実行） ────────────────────────────────
    diarized_path = arch.out_dir / "transcript_en_diarized.txt"
    hf_token = os.environ.get("HF_TOKEN", "")
    if hf_token and not diarized_path.exists():
        ts_path = arch.out_dir / "transcript_en_timestamped.json"
        if ts_path.exists():
            try:
                from factfull.ingest.diarize import diarize_episode
                diarize_episode(
                    video_id=source_id,
                    episode_dir=arch.out_dir,
                    title=title,
                    channel=channel,
                    hf_token=hf_token,
                )
            except Exception as e:
                print(f"  [diarize] スキップ（エラー）: {e}", flush=True)

    # 話者付きトランスクリプトがあればそちらを優先
    is_diarized = diarized_path.exists()
    if is_diarized:
        text = diarized_path.read_text(encoding="utf-8")
        print(f"  [diarize] 話者付きトランスクリプト使用", flush=True)
    else:
        text = arch.transcript_en

    source = SourceDoc(
        source_type="podcast",
        source_id=source_id,
        title=title,
        text=text,
        chunks=[],
        metadata={
            "channel": channel,
            "youtube_url": url,
            "diarized": is_diarized,
            **arch.metadata,
        },
    )

    # 抽出モード選択: summary_ja.md があれば高品質サマリー抽出を優先
    summary_path = arch.out_dir / "summary_ja.md"
    if summary_path.exists():
        from factfull.extract.podcast_extract import extract_from_summary
        summary_text = summary_path.read_text(encoding="utf-8")
        print(f"  [extract] サマリーベース抽出 (gemma4:26b)", flush=True)
        entities, triples = extract_from_summary(
            summary_text, source_id=source_id, model="gemma4:26b"
        )
    elif is_diarized:
        from factfull.extract.podcast_extract import extract_podcast
        print(f"  [extract] 話者付きトランスクリプト抽出", flush=True)
        entities, triples = extract_podcast(text, source_id=source_id)
    else:
        chunks = [c.text for c in chunk_text(text, chunk_size=4000, overlap=200)]
        print(f"  エンティティ抽出中... ({len(chunks[:10])} チャンク)", flush=True)
        entities = extract_entities(chunks[:10], source_id=source_id)
        triples = extract_relations(chunks[:10], entities, source_id=source_id) if entities else []

    with Neo4jClient() as g:
        g.setup_schema()
        from factfull.core.types import ProcessedDoc
        pdoc = ProcessedDoc(source=source, entities=entities, triples=triples)
        g.write_processed_doc(pdoc, clear_old=True)

    print(f"  entities={len(entities)}, triples={len(triples)}", flush=True)
    return title


def _process_paper(source_id: str) -> str:
    """arXiv ID → 処理してタイトルを返す。"""
    from pathlib import Path as _Path
    from factfull.ingest.paper import ingest_arxiv
    from factfull.process import process
    from factfull.graph.neo4j import Neo4jClient

    doc = ingest_arxiv(source_id, output_dir=_Path.home() / "papers" / "arxiv")
    pdoc = process(doc, model=MODEL, summarize=False, extract=True,
                   max_chunks_for_extract=5)
    with Neo4jClient() as g:
        g.setup_schema()
        g.write_processed_doc(pdoc)
    return doc.title


def _process_web(source_id: str) -> str:
    """URL → 処理してタイトルを返す。"""
    from factfull.ingest.web import ingest_url
    from factfull.process import process
    from factfull.graph.neo4j import Neo4jClient

    doc = ingest_url(source_id)
    pdoc = process(doc, model=MODEL, summarize=False, extract=True,
                   max_chunks_for_extract=5)
    with Neo4jClient() as g:
        g.setup_schema()
        g.write_processed_doc(pdoc)
    return doc.title


_PROCESSORS = {
    "podcast": _process_podcast,
    "paper":   _process_paper,
    "web":     _process_web,
}


# ── サブコマンド ──────────────────────────────────────────────────────────────

def cmd_add(args: argparse.Namespace) -> None:
    from factfull.registry import Registry
    with Registry() as reg:
        added = reg.add(args.source_type, args.source_id)
        if added:
            print(f"  ✅ 追加: [{args.source_type}] {args.source_id}")
        else:
            print(f"  ⏭  スキップ（既存）: [{args.source_type}] {args.source_id}")


def cmd_add_file(args: argparse.Namespace) -> None:
    from factfull.registry import Registry
    lines = Path(args.file).read_text(encoding="utf-8").splitlines()
    added_count = 0
    with Registry() as reg:
        for line in lines:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(None, 1)
            if len(parts) != 2:
                print(f"  ⚠️  スキップ（形式エラー）: {line!r}")
                continue
            source_type, source_id = parts
            if source_type not in _PROCESSORS:
                print(f"  ⚠️  未対応の種別: {source_type!r}")
                continue
            if reg.add(source_type, source_id):
                print(f"  ✅ [{source_type}] {source_id}")
                added_count += 1
            else:
                print(f"  ⏭  スキップ（既存）: [{source_type}] {source_id}")
    print(f"\n追加: {added_count} 件")


def cmd_run(args: argparse.Namespace) -> None:
    from factfull.registry import Registry
    graph_only = getattr(args, "graph_only", False)
    with Registry() as reg:
        items = reg.pending(source_type=args.type or None)
        if not items:
            print("pending なし。")
            return

        mode = "graph-only" if graph_only else "full"
        print(f"処理対象: {len(items)} 件  モード: {mode}\n")
        done, failed = 0, 0

        for item in items:
            stype, sid = item["source_type"], item["source_id"]
            if graph_only and stype == "podcast":
                processor = _process_podcast_graph_only
            else:
                processor = _PROCESSORS.get(stype)
            if processor is None:
                print(f"  ⚠️  [{stype}] {sid} — 未対応の種別、スキップ")
                continue

            print(f"  ▶ [{stype}] {sid}")
            reg.mark_processing(stype, sid)
            try:
                title = processor(sid)
                reg.mark_done(stype, sid, title=title, graph_written=True)
                print(f"  ✅ 完了: {title[:60]}")
                done += 1
            except Exception as e:
                reg.mark_failed(stype, sid, error=str(e))
                print(f"  ❌ 失敗: {e}")
                failed += 1

        print(f"\n完了: {done} 件 / 失敗: {failed} 件")


def cmd_retry(args: argparse.Namespace) -> None:
    from factfull.registry import Registry
    with Registry() as reg:
        items = reg.failed(source_type=args.type or None)
        if not items:
            print("failed なし。")
            return
        for item in items:
            reg.retry(item["source_type"], item["source_id"])
            print(f"  🔄 retry: [{item['source_type']}] {item['source_id']}")
        print(f"\n{len(items)} 件を pending に戻しました。'run' で再実行してください。")


def cmd_status(args: argparse.Namespace) -> None:
    from factfull.registry import Registry
    with Registry() as reg:
        s = reg.stats()
        print("=== ステータス別 ===")
        for status, cnt in sorted(s["by_status"].items()):
            print(f"  {status:<12}: {cnt}")
        print("\n=== 種別別 ===")
        for stype, cnt in sorted(s["by_type"].items()):
            print(f"  {stype:<12}: {cnt}")


def cmd_list(args: argparse.Namespace) -> None:
    from factfull.registry import Registry
    with Registry() as reg:
        items = reg.list_all(
            source_type=args.type or None,
            status=args.status or None,
            limit=args.limit,
        )
        if not items:
            print("該当なし。")
            return
        for item in items:
            icon = {"done": "✅", "failed": "❌", "pending": "⏳", "processing": "🔄"}.get(item["status"], "?")
            title = (item["title"] or item["source_id"])[:50]
            print(f"  {icon} [{item['source_type']}] {title}")
            if item["status"] == "failed" and item["error"]:
                print(f"       エラー: {item['error'][:80]}")


# ── エントリポイント ──────────────────────────────────────────────────────────

def main() -> None:
    _setup_env()

    parser = argparse.ArgumentParser(description="factfull バッチ処理")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_add = sub.add_parser("add", help="ソースを1件追加")
    p_add.add_argument("source_type", choices=list(_PROCESSORS))
    p_add.add_argument("source_id")

    p_af = sub.add_parser("add-file", help="ファイルから一括追加")
    p_af.add_argument("file", help="'type source_id' 形式のテキストファイル")

    p_run = sub.add_parser("run", help="pending を実行")
    p_run.add_argument("--type", choices=list(_PROCESSORS), help="種別を絞り込む")
    p_run.add_argument("--graph-only", action="store_true", help="podcast: 要約・ファクトチェックをスキップしてグラフ書き込みのみ")

    p_retry = sub.add_parser("retry", help="failed を pending に戻す")
    p_retry.add_argument("--type", choices=list(_PROCESSORS))

    sub.add_parser("status", help="統計表示")

    p_list = sub.add_parser("list", help="一覧表示")
    p_list.add_argument("--type", choices=list(_PROCESSORS))
    p_list.add_argument("--status", choices=["pending", "processing", "done", "failed"])
    p_list.add_argument("--limit", type=int, default=50)

    args = parser.parse_args()

    dispatch = {
        "add":      cmd_add,
        "add-file": cmd_add_file,
        "run":      cmd_run,
        "retry":    cmd_retry,
        "status":   cmd_status,
        "list":     cmd_list,
    }
    dispatch[args.cmd](args)


if __name__ == "__main__":
    main()
