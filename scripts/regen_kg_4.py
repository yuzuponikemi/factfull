"""Regenerate the 4 missing KG JSONs from cached ~/podcasts output dirs.

These articles were published with orphan kg-widget references because the
KG export silently failed (Neo4j unreachable at generation time). The cached
transcripts/summaries still exist, so we re-run only the KG extraction →
Neo4j write → JSON export — no re-download, transcription, or re-summarize.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

os.environ["NEO4J_URI"] = "bolt://localhost:7687"
os.environ["NEO4J_USER"] = "neo4j"
os.environ["NEO4J_PASSWORD"] = "your_password_here"  # kg-builder-neo4j actual pw
os.environ["FACTFULL_LLM_BACKEND"] = "ollama"

from factfull.podcast.pipeline import PipelineConfig
from factfull.podcast.steps.graph import write_to_graph
from factfull.publishers.homupe import _export_kg_json

PODCASTS = Path.home() / "podcasts"
# any dir under homupe/docs/blog/posts/YYYY/MM → parents[4] == homupe root
BLOG_DIR = Path("/Users/ikmx/source/personal/homupe/docs/blog/posts/2026/06")

DIRS = {
    "1M3Vdl6DRkU": "1M3Vdl6DRkU_20260530",
    "sRKBGVFVYAw": "sRKBGVFVYAw_20260518",
    "xmkSf5IS-zw": "xmkSf5IS-zw_20260509",
    "Jj-kBHzUohs": "Jj-kBHzUohs_20260606",
}

config = PipelineConfig(extract_model="gemma4:26b")

results = {}
for vid, dname in DIRS.items():
    ep = PODCASTS / dname
    print(f"\n{'#'*70}\n# {vid}  ({dname})\n{'#'*70}", flush=True)
    if not (ep / "summary_ja.md").exists():
        print(f"  ❌ summary_ja.md なし → スキップ", flush=True)
        results[vid] = "no-summary"
        continue
    meta = json.loads((ep / "metadata.json").read_text(encoding="utf-8"))
    result = SimpleNamespace(
        video_id=vid,
        title=meta.get("title", ""),
        channel=meta.get("channel", ""),
        summary_path=ep / "summary_ja.md",
        episode_dir=ep,
        metadata=meta,
        score=0.0,
    )
    try:
        write_to_graph(result, config)
    except Exception as e:
        print(f"  ❌ write_to_graph 失敗: {e}", flush=True)
        results[vid] = f"write-failed: {e}"
        continue
    ok = _export_kg_json(vid, BLOG_DIR)
    results[vid] = "exported" if ok else "export-failed"

print(f"\n{'='*70}\nRESULTS:", flush=True)
for vid, status in results.items():
    print(f"  {vid}: {status}", flush=True)
