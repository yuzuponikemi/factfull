"""
nightly_pipeline.py
====================
新着コンテンツを自動検出 → 記事生成 → homupe 投稿 → git push。

フロー:
  Phase 0: arXiv 新着論文を取得 → KG 登録 → 日次ダイジェスト記事生成（平日のみ）
  Phase 1: RSS フィードで新着エピソードを検出 → Registry に追加
  Phase 2: pending エピソードを直列処理（translate → factcheck → KG → 記事投稿）
  Phase 3: homupe の変更を git commit & push（--push 時のみ）

使い方:
    uv run python nightly_pipeline.py                  # 通常実行
    uv run python nightly_pipeline.py --dry-run        # 検出のみ（処理しない）
    uv run python nightly_pipeline.py --push           # 処理後 homupe を git push
    uv run python nightly_pipeline.py --max 1          # 1 件だけ処理
    uv run python nightly_pipeline.py --channel lex_fridman  # 特定チャンネルのみ
    uv run python nightly_pipeline.py --skip-arxiv     # arXiv フェーズをスキップ
    uv run python nightly_pipeline.py --arxiv-only     # arXiv フェーズのみ実行
"""
from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml  # type: ignore


# ── 設定読み込み ──────────────────────────────────────────────────────────────

REPO_ROOT        = Path(__file__).parent
CONFIG_PATH      = REPO_ROOT / "config" / "channels.yaml"
ARXIV_CONFIG_PATH = REPO_ROOT / "config" / "arxiv_feeds.yaml"
HOMUPE_ROOT      = Path(os.environ.get("HOMUPE_ROOT",
                   str(Path.home() / "source" / "personal" / "homupe")))


# ── 常時ロギング（stdout/stderr を固定ログにも複製） ────────────────────────────

LOG_PATH = Path("/tmp/factfull_logs/nightly.log")


class _Tee:
    """書き込みを複数ストリームへ複製する薄いラッパー。"""

    def __init__(self, primary, *extras) -> None:
        self._primary = primary
        self._extras = extras

    def write(self, data: str) -> int:
        for s in self._extras:
            s.write(data)
            s.flush()
        return self._primary.write(data)

    def flush(self) -> None:
        self._primary.flush()
        for s in self._extras:
            s.flush()

    def __getattr__(self, name):
        return getattr(self._primary, name)


def _setup_tee_logging(log_path: Path = LOG_PATH) -> None:
    """stdout/stderr を log_path にも常時複製する。
    すでに同じファイルへリダイレクトされている場合（launchd の StandardOutPath 等）は
    二重書き込みを避けるためスキップする。"""
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # stdout が既に log_path と同じファイル（inode 一致）なら何もしない
    try:
        if log_path.exists():
            stdout_stat = os.fstat(sys.stdout.fileno())
            log_stat = os.stat(log_path)
            if (stdout_stat.st_ino, stdout_stat.st_dev) == (log_stat.st_ino, log_stat.st_dev):
                return
    except (OSError, AttributeError):
        pass

    log_file = open(log_path, "a", buffering=1, encoding="utf-8")
    log_file.write(f"\n===== {datetime.now():%Y-%m-%d %H:%M:%S} session start =====\n")
    sys.stdout = _Tee(sys.stdout, log_file)
    sys.stderr = _Tee(sys.stderr, log_file)


@dataclass
class ChannelConfig:
    id: str
    name: str
    channel_id: str
    enabled: bool = True
    max_per_run: int = 2
    lookback_days: int = 30


def load_channels(path: Path = CONFIG_PATH) -> list[ChannelConfig]:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return [ChannelConfig(**ch) for ch in data["channels"]]


@dataclass
class ArxivFeedConfig:
    categories: list[str]
    papers_per_digest: int = 5
    lookback_days: int = 1
    max_per_category: int = 30
    summarize_model: str = "gemma4:e4b"


def load_arxiv_config(path: Path = ARXIV_CONFIG_PATH) -> ArxivFeedConfig:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    enabled_cats = [c["id"] for c in data["categories"] if c.get("enabled", True)]
    return ArxivFeedConfig(
        categories=enabled_cats,
        papers_per_digest=data.get("papers_per_digest", 5),
        lookback_days=data.get("lookback_days", 1),
        max_per_category=data.get("max_per_category", 30),
        summarize_model=data.get("summarize_model", "gemma4:e4b"),
    )


# ── Pipeline 設定 ──────────────────────────────────────────────────────────────

def _make_pipeline_config():
    from factfull.podcast.pipeline import PipelineConfig
    return PipelineConfig(
        translate_model  = "translategemma:12b",
        analyze_model    = "gemma4:26b",
        extract_model    = "gemma4:26b",
        factcheck_model  = "gemma4:e4b",
        editorial_model  = None,
        threshold        = 95.0,
        max_iter         = 5,
        max_claims       = 50,
        top_k            = 5,
        critique         = True,
        editorial        = True,
        fetch_comments   = False,
        write_graph      = True,
        output_base      = Path.home() / "podcasts",
        blog_name        = "SoryuNews",
        reader_persona   = "英語圏情報にアクセスしたい日本語話者のエンジニア・研究者",
        n_questions      = 4,
    )


META_MODEL = "gemma4:e4b"


# ── Phase 0: arXiv 日次ダイジェスト ──────────────────────────────────────────

def process_arxiv_papers(
    arxiv_cfg: ArxivFeedConfig,
    registry,
    blog_dir: Path,
    dry_run: bool = False,
) -> bool:
    """arXiv 新着論文を取得 → KG 登録 → ダイジェスト記事を生成する。

    平日のみ実行。ダイジェスト (arxiv_digest_YYYYMMDD) が既に登録済みなら即リターン。

    Returns:
        True if digest was created, False otherwise.
    """
    from datetime import date

    today = date.today()

    # 平日チェック (0=月 ... 4=金)
    if today.weekday() >= 5:
        print("  [arXiv] 土日 → スキップ")
        return False

    digest_id = f"arxiv_digest_{today.strftime('%Y%m%d')}"
    if registry.exists("arxiv_digest", digest_id):
        print(f"  [arXiv] 本日分 {digest_id} は処理済み → スキップ")
        return False

    from factfull.ingest.arxiv_feed import find_new_papers, fetch_conclusion
    from factfull.extract.entity import extract_entities
    from factfull.extract.relation import extract_relations
    from factfull.core.types import SourceDoc, ProcessedDoc
    from factfull.graph.neo4j import Neo4jClient
    from factfull.arxiv.digest import summarize_paper, build_digest
    from factfull.publishers.homupe import create_arxiv_digest_post

    # Step 1: 新着論文取得
    print(f"  [arXiv] カテゴリ {arxiv_cfg.categories} から最新論文を取得中...")
    papers = find_new_papers(
        categories=arxiv_cfg.categories,
        registry=registry,
        papers_per_digest=arxiv_cfg.papers_per_digest,
        lookback_days=arxiv_cfg.lookback_days,
        max_per_category=arxiv_cfg.max_per_category,
    )

    if not papers:
        print("  [arXiv] 新着論文なし → スキップ")
        return False

    print(f"  [arXiv] {len(papers)} 件の新着論文を処理します")
    if dry_run:
        for p in papers:
            print(f"    [DRY-RUN] {p.paper_id}: {p.title[:60]}")
        return False

    # Step 2: Conclusion 取得 + エンティティ抽出 → Neo4j
    model = arxiv_cfg.summarize_model
    processed: list = []

    with Neo4jClient() as client:
        for paper in papers:
            print(f"\n  📄 [{paper.paper_id}] {paper.title[:55]}...")
            registry.add("arxiv", paper.paper_id, title=paper.title)
            registry.mark_processing("arxiv", paper.paper_id)

            try:
                # Conclusion 取得
                print(f"    → conclusion 取得中...")
                paper.conclusion = fetch_conclusion(paper.paper_id)
                if paper.conclusion:
                    print(f"    → {len(paper.conclusion)} 文字")
                else:
                    print(f"    → (取得できず、abstract のみ使用)")

                # テキスト = abstract + conclusion
                source_text = f"{paper.abstract}\n\n{paper.conclusion}".strip()
                source_id = f"arxiv_{paper.paper_id}"

                # エンティティ・関係抽出
                chunks = [source_text]
                entities = extract_entities(chunks, source_id=source_id, model=model)
                print(f"    → entities: {len(entities)}")
                relations = extract_relations(chunks, entities, source_id=source_id, model=model)
                print(f"    → relations: {len(relations)}")

                # Neo4j 書き込み
                source = SourceDoc(
                    source_type="arxiv",
                    source_id=source_id,
                    title=paper.title,
                    text=source_text,
                    metadata={
                        "paper_id": paper.paper_id,
                        "authors": paper.authors[:5],
                        "categories": paper.categories,
                        "arxiv_url": paper.arxiv_url,
                        "published": paper.published.isoformat() if paper.published else "",
                    },
                )
                pdoc = ProcessedDoc(source=source, entities=entities, triples=relations)
                client.write_processed_doc(pdoc, clear_old=True)

                registry.mark_done("arxiv", paper.paper_id, graph_written=True)
                processed.append(paper)

            except Exception as e:
                err = f"{type(e).__name__}: {e}"
                print(f"    [ERROR] {err}", file=sys.stderr)
                registry.mark_failed("arxiv", paper.paper_id, error=err)

    if not processed:
        print("  [arXiv] 処理成功論文なし → ダイジェスト生成スキップ")
        return False

    # Step 3: 日本語要約 → ダイジェスト生成
    print(f"\n  [arXiv] 日本語要約を生成中 ({len(processed)} 件)...")
    summaries = []
    for paper in processed:
        print(f"    → {paper.paper_id}: {paper.title[:50]}...")
        ps = summarize_paper(paper, model=model)
        summaries.append(ps)

    date_str = today.strftime("%Y-%m-%d")
    digest = build_digest(summaries, date=date_str, model=model)

    # Step 4: 記事生成 → homupe
    print(f"\n  [arXiv] ダイジェスト記事を生成中...")
    post_path = create_arxiv_digest_post(digest, blog_dir=blog_dir)

    registry.add("arxiv_digest", digest_id, title=f"arXiv ダイジェスト {date_str}")
    registry.mark_done("arxiv_digest", digest_id, graph_written=True)

    print(f"  ✅ [arXiv] ダイジェスト完了: {post_path.name}")
    return True


# ── Phase 1: 新着検出 → Registry ──────────────────────────────────────────────

def scan_feeds(
    channels: list[ChannelConfig],
    registry,
    channel_filter: str | None = None,
) -> int:
    """RSS を巡回して新着を Registry に追加。追加件数を返す。"""
    from factfull.ingest.youtube_feed import find_new_entries

    added = 0
    for ch in channels:
        if not ch.enabled:
            continue
        if channel_filter and ch.id != channel_filter:
            continue

        new_entries = find_new_entries(
            channel_id   = ch.channel_id,
            channel_name = ch.name,
            registry     = registry,
            lookback_days= ch.lookback_days,
            max_new      = ch.max_per_run,
        )

        for entry in new_entries:
            if registry.add("podcast", entry.video_id, title=entry.title):
                print(f"  + 追加: [{ch.name}] {entry.title} ({entry.video_id})")
                added += 1
            # else: すでに存在（skip_if_exists=True なので問題なし）

        total = len(new_entries)
        print(f"  {ch.name}: 新着 {total} 件 / 追加 {sum(1 for e in new_entries if registry.exists('podcast', e.video_id))} 件")

    return added


# ── Phase 2: pending 処理 ─────────────────────────────────────────────────────

def process_pending(
    registry,
    blog_dir: Path,
    max_episodes: int = 6,
    dry_run: bool = False,
) -> tuple[int, int]:
    """
    Registry の pending エピソードを処理する。

    Returns:
        (成功件数, 失敗件数)
    """
    from factfull.podcast.pipeline import run_pipeline
    from factfull.publishers.homupe import generate_blog_metadata, create_blog_post

    pending = registry.pending("podcast")[:max_episodes]
    if not pending:
        print("  pending なし")
        return 0, 0

    print(f"  処理対象: {len(pending)} 件（上限 {max_episodes}）")
    pipeline_config = _make_pipeline_config()

    ok = fail = 0
    for item in pending:
        vid = item["source_id"]
        title = item.get("title") or vid
        url = f"https://www.youtube.com/watch?v={vid}"

        print(f"\n  ▶ [{vid}] {title[:60]}")

        if dry_run:
            print("    [DRY-RUN] スキップ")
            continue

        from factfull.ingest.youtube_feed import get_video_duration_seconds
        dur = get_video_duration_seconds(vid)
        if dur is not None and dur < 3600:
            print(f"    [SKIP] 短尺動画 ({dur}s < 3600s)")
            registry.mark_failed("podcast", vid, error=f"short_video:{dur}s")
            fail += 1
            continue

        registry.mark_processing("podcast", vid)
        try:
            result = run_pipeline(pipeline_config, url)
            print(f"    score={result.score:.0f}  KG=✓")

            meta = generate_blog_metadata(result, model=META_MODEL)
            post_path = create_blog_post(result, meta, blog_dir=blog_dir)
            print(f"    投稿: {post_path.name}")

            registry.mark_done(
                "podcast", vid,
                title=result.title,
                graph_written=pipeline_config.write_graph,
            )
            ok += 1

        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            print(f"    [ERROR] {err}", file=sys.stderr)
            registry.mark_failed("podcast", vid, error=err)
            fail += 1

    return ok, fail


# ── Phase 3: homupe git push ──────────────────────────────────────────────────

def git_push_homupe(homupe_root: Path, dry_run: bool = False) -> bool:
    """homupe の変更を add → commit → push する。変更がなければ何もしない。"""
    def run(cmd: list[str]) -> subprocess.CompletedProcess:
        return subprocess.run(cmd, cwd=str(homupe_root), capture_output=True, text=True)

    # 変更確認
    status = run(["git", "status", "--porcelain"])
    if not status.stdout.strip():
        print("  homupe: 変更なし")
        return False

    changed_files = status.stdout.strip().splitlines()
    print(f"  homupe: {len(changed_files)} ファイル変更あり")

    if dry_run:
        print("  [DRY-RUN] git push をスキップ")
        return False

    today = datetime.now().strftime("%Y-%m-%d")
    msg = f"feat: nightly articles {today}"

    run(["git", "add", "docs/blog/posts/", "docs/data/kg/", "docs/data/synthesis/"])
    commit = run(["git", "commit", "-m", msg])
    if commit.returncode != 0:
        print(f"  [WARN] git commit 失敗: {commit.stderr.strip()}")
        return False

    push = run(["git", "push"])
    if push.returncode != 0:
        print(f"  [WARN] git push 失敗: {push.stderr.strip()}")
        return False

    print(f"  ✅ homupe push 完了: {msg}")
    return True


# ── メイン ────────────────────────────────────────────────────────────────────

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="factfull 夜間ポッドキャストパイプライン",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--dry-run",       action="store_true", help="検出のみ（処理・投稿しない）")
    parser.add_argument("--push",          action="store_true", help="処理後 homupe を git push")
    parser.add_argument("--max",           type=int, default=6,  help="1回の実行で処理する上限（default: 6）")
    parser.add_argument("--channel",       default=None, help="チャンネル ID で絞り込み（例: lex_fridman）")
    parser.add_argument("--skip-scan",     action="store_true", help="RSS スキャンをスキップして pending のみ処理")
    parser.add_argument("--skip-arxiv",    action="store_true", help="arXiv フェーズをスキップ")
    parser.add_argument("--arxiv-only",    action="store_true", help="arXiv フェーズのみ実行（podcast をスキップ）")
    parser.add_argument("--skip-substack", action="store_true", help="Substack ドラフト作成をスキップ")
    args = parser.parse_args()

    # stdout/stderr を /tmp/factfull_logs/nightly.log にも複製
    # （マニュアル実行・launchd 実行どちらでも tail -f で追跡可能にする）
    _setup_tee_logging()

    # ~/.config/factfull/.env から認証情報を読み込む（存在する場合のみ）
    _secrets = Path.home() / ".config" / "factfull" / ".env"
    if _secrets.exists():
        for _line in _secrets.read_text(encoding="utf-8").splitlines():
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _, _v = _line.partition("=")
                os.environ.setdefault(_k.strip(), _v.strip())

    # 環境変数デフォルト
    os.environ.setdefault("NEO4J_URI",      "bolt://localhost:7687")
    os.environ.setdefault("NEO4J_USER",     "neo4j")
    os.environ.setdefault("NEO4J_PASSWORD", "factfull123")
    os.environ.setdefault("FACTFULL_LLM_BACKEND", "ollama")

    started_at = datetime.now()
    print(f"\n{'='*60}")
    print(f"[nightly] 開始: {started_at:%Y-%m-%d %H:%M:%S}")
    print(f"{'='*60}")

    from factfull.registry import Registry
    from factfull.publishers.homupe import default_blog_dir

    channels    = load_channels()
    arxiv_cfg   = load_arxiv_config()
    blog_dir    = default_blog_dir(HOMUPE_ROOT)
    blog_dir.mkdir(parents=True, exist_ok=True)

    # 処理前のファイル一覧をスナップショット（新規作成検出用）
    existing_posts: set[Path] = set(blog_dir.glob("*.md"))

    arxiv_created = False
    ok = fail = 0

    with Registry() as reg:
        # ── Phase 0: arXiv ダイジェスト ──
        if not args.skip_arxiv:
            print("\n[Phase 0] arXiv 日次ダイジェスト（平日のみ）")
            arxiv_created = process_arxiv_papers(
                arxiv_cfg, reg, blog_dir, dry_run=args.dry_run
            )
        else:
            print("\n[Phase 0] スキップ（--skip-arxiv）")

        if args.arxiv_only:
            print("\n[--arxiv-only] Phase 1/2 をスキップ")
        else:
            # ── Phase 1: RSS スキャン ──
            if not args.skip_scan:
                print("\n[Phase 1] RSS スキャン")
                added = scan_feeds(channels, reg, channel_filter=args.channel)
                print(f"\n  合計追加: {added} 件")
            else:
                print("\n[Phase 1] スキップ（--skip-scan）")

            # ── Phase 2: pending 処理 ──
            pending_count = len(reg.pending("podcast"))
            print(f"\n[Phase 2] pending 処理（{pending_count} 件 / 上限 {args.max}）")
            ok, fail = process_pending(reg, blog_dir, max_episodes=args.max, dry_run=args.dry_run)
            print(f"\n[Phase 2] 完了: 成功 {ok}, 失敗 {fail}")

        stats = reg.stats()
        print(f"  Registry 状況: {stats['by_status']}")

    # ── Phase 3: Substack ドラフト ──
    any_new = ok > 0 or arxiv_created
    new_posts = sorted(set(blog_dir.glob("*.md")) - existing_posts)

    if args.skip_substack or args.dry_run:
        print("\n[Phase 3] Substack スキップ")
    elif not new_posts:
        print("\n[Phase 3] 新規記事なし — Substack スキップ")
    else:
        from factfull.publishers.substack import SubstackClient, post_to_draft, substack_enabled
        print(f"\n[Phase 3] Substack ドラフト作成（{len(new_posts)} 件）")
        if not substack_enabled():
            print("  ⚠ SUBSTACK_* 環境変数が未設定 — スキップ")
        else:
            try:
                client = SubstackClient.from_env()
                for post_path in new_posts:
                    try:
                        draft = post_to_draft(client, post_path)
                        draft_id = draft.get("id", "?")
                        print(f"  ✅ ドラフト作成: {post_path.name}  (id={draft_id})")
                    except Exception as e:
                        print(f"  ❌ 失敗: {post_path.name}  {e}", file=sys.stderr)
            except Exception as e:
                print(f"  ❌ Substack ログイン失敗: {e}", file=sys.stderr)

    # ── Phase 4: git push ──
    if args.push and any_new:
        print("\n[Phase 4] homupe git push")
        git_push_homupe(HOMUPE_ROOT, dry_run=args.dry_run)
    elif args.push and not any_new:
        print("\n[Phase 4] 新規投稿なし — push スキップ")

    elapsed = (datetime.now() - started_at).total_seconds()
    print(f"\n[nightly] 完了: {elapsed:.0f}秒")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
