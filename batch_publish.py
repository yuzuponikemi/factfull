"""
batch_publish.py
=================
未投稿エピソードをまとめて処理して homupe に投稿する。

使い方:
    uv run python batch_publish.py
    uv run python batch_publish.py --dry-run
    uv run python batch_publish.py --video-ids KBPOTklFTiU,mDG_Hx3BSUE
"""
from __future__ import annotations

import os
import re
import sys
from datetime import datetime
from pathlib import Path


# ── 環境変数 ────────────────────────────────────────────────────────────────────

os.environ.setdefault("NEO4J_URI", "bolt://localhost:7687")
os.environ.setdefault("NEO4J_USER", "neo4j")
os.environ.setdefault("NEO4J_PASSWORD", "factfull123")
os.environ.setdefault("FACTFULL_LLM_BACKEND", "ollama")


# ── ヘルパー ────────────────────────────────────────────────────────────────────

def _collect_unpublished(podcast_base: Path, blog_base: Path) -> list[dict]:
    """未投稿エピソードのリストを返す。"""
    import json

    # homupe 掲載済み video_id を収集
    blog_vids: set[str] = set()
    for md in blog_base.rglob("*.md"):
        for m in re.finditer(r"youtube\.com/watch\?v=([\w-]+)", md.read_text(errors="ignore")):
            blog_vids.add(m.group(1))

    # ローカルエピソードを列挙
    id_to_dir: dict[str, Path] = {}
    for d in sorted(podcast_base.iterdir()):
        if not d.is_dir():
            continue
        parts = d.name.rsplit("_", 1)
        vid = parts[0] if len(parts) == 2 and parts[1].isdigit() and len(parts[1]) == 8 else d.name
        # 最新の中身ありディレクトリを選ぶ
        if vid not in id_to_dir or d.name > id_to_dir[vid].name:
            if any(True for f in d.iterdir() if f.is_file()):
                id_to_dir[vid] = d

    episodes = []
    for vid, ep_dir in sorted(id_to_dir.items()):
        if vid in blog_vids:
            continue
        meta_path = ep_dir / "metadata.json"
        title = vid
        url = f"https://www.youtube.com/watch?v={vid}"
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text())
                title = meta.get("title", vid)
                url = meta.get("url", url) or url
            except Exception:
                pass
        has_summary = (ep_dir / "summary_ja.md").exists()
        has_ja = (ep_dir / "transcript_ja.txt").exists()
        episodes.append({
            "vid": vid,
            "url": url,
            "title": title,
            "ep_dir": ep_dir,
            "has_summary": has_summary,
            "has_ja": has_ja,
            "regen": has_summary or has_ja,  # 処理済みデータがあれば再利用
        })
    return episodes


def _process(ep: dict, config, blog_dir: Path) -> bool:
    from factfull.podcast.pipeline import run_pipeline
    from factfull.publishers.homupe import generate_blog_metadata, create_blog_post

    vid = ep["vid"]
    print(f"\n  {'='*55}", flush=True)
    print(f"  [{vid}] {ep['title'][:70]}", flush=True)
    print(f"  regen={ep['regen']}  has_summary={ep['has_summary']}", flush=True)

    try:
        result = run_pipeline(config, ep["url"], regen=ep["regen"])
        print(f"  → score={result.score:.1f}", flush=True)

        meta = generate_blog_metadata(result, model=config.analyze_model)
        post_path = create_blog_post(result, meta, blog_dir=blog_dir)
        print(f"  → 投稿: {post_path}", flush=True)
        return True
    except Exception as e:
        print(f"  [ERROR] {type(e).__name__}: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return False


# ── メイン ────────────────────────────────────────────────────────────────────

def main() -> None:
    import argparse
    from factfull.podcast.pipeline import PipelineConfig
    from factfull.publishers.homupe import default_blog_dir

    parser = argparse.ArgumentParser(description="未投稿エピソードをまとめて処理")
    parser.add_argument("--dry-run", action="store_true", help="リスト表示のみ")
    parser.add_argument("--video-ids", default=None, help="対象 video_id をカンマ区切りで指定")
    parser.add_argument("--model", default="gemma4:26b")
    args = parser.parse_args()

    podcast_base = Path.home() / "podcasts"
    blog_base = Path.home() / "source/personal/homupe/docs/blog/posts"
    blog_dir = default_blog_dir()
    blog_dir.mkdir(parents=True, exist_ok=True)

    episodes = _collect_unpublished(podcast_base, blog_base)

    if args.video_ids:
        target_ids = {v.strip() for v in args.video_ids.split(",")}
        episodes = [e for e in episodes if e["vid"] in target_ids]

    print(f"\n未投稿エピソード: {len(episodes)} 件", flush=True)
    for i, ep in enumerate(episodes, 1):
        flag = "regen" if ep["regen"] else "full "
        print(f"  [{i:2d}] ({flag}) {ep['vid']}  {ep['title'][:60]}", flush=True)

    if args.dry_run or not episodes:
        return

    config = PipelineConfig(
        write_graph=False,          # KG は登録済み
        analyze_model=args.model,
        translate_model="translategemma:12b",
        factcheck_model="gemma4:e4b",
    )

    started = datetime.now()
    ok = 0
    for ep in episodes:
        success = _process(ep, config, blog_dir)
        if success:
            ok += 1

    elapsed = (datetime.now() - started).total_seconds()
    print(f"\n完了: 成功 {ok}/{len(episodes)} 件  ({elapsed/60:.0f} 分)", flush=True)


if __name__ == "__main__":
    main()
