"""
factfull/export/synthesis_graph.py
====================================
合成記事（dialectical-*.md）の kg_sources を読み取り、
2つのソースのサブグラフを「クラスター付き JSON」として出力する。

出力形式:
  {
    "sources": [{"source_id": ..., "title": ..., "type": "book"|"podcast"}, ...],
    "nodes":   [{"id", "label", "type", "description", "color", "cluster": 0|1}, ...],
    "links":   [{"source", "target", "type", "color"}, ...]
  }

CLI:
    uv run python -m factfull.export.synthesis_graph --all --output ../homupe/docs/data/synthesis/
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from factfull.graph.neo4j import Neo4jClient


_NODE_COLORS: dict[str, str] = {
    "person":       "#4A90D9",
    "work":         "#F5A623",
    "concept":      "#7ED321",
    "organization": "#9B59B6",
    "place":        "#E67E22",
    "claim":        "#E74C3C",
    "product":      "#1ABC9C",
    "method":       "#3498DB",
    "event":        "#F39C12",
    "measurement":  "#95A5A6",
}

# クラスター 0/1 の色調（薄いリング or 背景で区別）
CLUSTER_TINT = ["#2a3a5a", "#2a4a2a"]  # 青系 / 緑系

_EDGE_COLORS: dict[str, str] = {
    "DEPENDS_ON":   "#8888bb",
    "CONTRADICTS":  "#E74C3C",
    "EVOLVES_INTO": "#2ECC71",
    "AUTHORED":     "#4A90D9",
    "ARGUES_THAT":  "#F5A623",
    "SAYS":         "#9B59B6",
    "RELATED_TO":   "#667799",
    "PART_OF":      "#667799",
    "WORKS_AT":     "#667799",
    "BUILDS_ON":    "#8888bb",
    "MENTIONS":     "#555555",
}


def _shorten(name: str) -> str:
    if name.startswith("[Aporia] "):
        body = name[9:]
        return "❓ " + (body[:50] + "…" if len(body) > 50 else body)
    return name[:60] + ("…" if len(name) > 60 else "")


def export_synthesis_graph(
    source_ids: list[str],
    client: Neo4jClient,
) -> dict[str, Any]:
    """2つの source_id のサブグラフを cluster 付きで結合する。"""

    sources_meta = []
    all_nodes: dict[str, dict] = {}   # id → node dict (重複排除)
    all_links: list[dict] = []
    seen_links: set[tuple] = set()

    for cluster_idx, sid in enumerate(source_ids[:2]):
        # ソースメタ情報
        src_r = client.run_cypher(
            "MATCH (s:Source {source_id:$sid}) RETURN s.title as title, s.source_type as stype",
            {"sid": sid},
        )
        title = src_r[0]["title"] if src_r else sid
        stype = src_r[0]["stype"] if src_r else "unknown"
        sources_meta.append({"source_id": sid, "title": title, "type": stype})

        # ノード取得（podcast は関係数上位 20 件に絞る）
        is_podcast = stype == "podcast"
        if is_podcast:
            nodes_r = client.run_cypher(
                """
                MATCH (s:Source {source_id:$sid})-[:MENTIONS]->(e:Entity)
                OPTIONAL MATCH (e)-[r]-()
                WITH e, count(r) as degree
                ORDER BY degree DESC
                LIMIT 20
                RETURN e.name as name, e.type as type, e.description as desc
                """,
                {"sid": sid},
            )
        else:
            nodes_r = client.run_cypher(
                """
                MATCH (s:Source {source_id:$sid})-[:MENTIONS]->(e:Entity)
                RETURN e.name as name, e.type as type, e.description as desc
                """,
                {"sid": sid},
            )
        node_ids = {r["name"] for r in nodes_r}

        for r in nodes_r:
            nid = r["name"]
            if nid not in all_nodes:
                all_nodes[nid] = {
                    "id":          nid,
                    "label":       _shorten(nid),
                    "type":        r["type"],
                    "description": (r["desc"] or ""),
                    "color":       _NODE_COLORS.get(r["type"], "#888888"),
                    "cluster":     cluster_idx,
                }
            # 両クラスターに属するノードは cluster=2（ブリッジ）
            elif all_nodes[nid]["cluster"] != cluster_idx:
                all_nodes[nid]["cluster"] = 2

        # エッジ取得（同一ソース内）
        edges_r = client.run_cypher(
            """
            MATCH (s:Source {source_id:$sid})-[:MENTIONS]->(a:Entity)-[r]->(b:Entity)
            WHERE b.name IN $node_ids AND type(r) <> 'MENTIONS'
            RETURN a.name as src, type(r) as rel, b.name as tgt
            """,
            {"sid": sid, "node_ids": list(node_ids)},
        )
        for e in edges_r:
            key = (e["src"], e["rel"], e["tgt"])
            if key not in seen_links:
                seen_links.add(key)
                all_links.append({
                    "source": e["src"],
                    "target": e["tgt"],
                    "type":   e["rel"],
                    "color":  _EDGE_COLORS.get(e["rel"], "#888888"),
                })

    # クラスター間の共有エンティティを検索（bridge links）
    if len(source_ids) >= 2:
        sid0, sid1 = source_ids[0], source_ids[1]
        bridge_r = client.run_cypher(
            """
            MATCH (s1:Source {source_id:$sid0})-[:MENTIONS]->(e:Entity)<-[:MENTIONS]-(s2:Source {source_id:$sid1})
            RETURN e.name as name
            """,
            {"sid0": sid0, "sid1": sid1},
        )
        for row in bridge_r:
            nid = row["name"]
            if nid in all_nodes:
                all_nodes[nid]["cluster"] = 2  # ブリッジ

    return {
        "sources": sources_meta,
        "nodes":   list(all_nodes.values()),
        "links":   all_links,
    }


def _parse_kg_sources(md_path: Path) -> list[str]:
    """markdown frontmatter から kg_sources リストを抽出する。"""
    text = md_path.read_text(encoding="utf-8")
    m = re.search(r"^kg_sources:\s*\n((?:  - .+\n)+)", text, re.MULTILINE)
    if not m:
        return []
    return [line.strip()[2:].strip() for line in m.group(1).splitlines()]


def export_all_synthesis(
    docs_dir: Path,
    output_dir: Path,
    neo4j_uri: str      = "bolt://localhost:7687",
    neo4j_user: str     = "neo4j",
    neo4j_password: str = "factfull123",
) -> list[str]:
    """docs_dir 以下の dialectical-*.md を走査して合成グラフ JSON を出力する。"""
    os.environ.setdefault("NEO4J_URI", neo4j_uri)
    os.environ.setdefault("NEO4J_USER", neo4j_user)
    os.environ.setdefault("NEO4J_PASSWORD", neo4j_password)

    output_dir.mkdir(parents=True, exist_ok=True)
    exported: list[str] = []

    md_files = sorted(docs_dir.rglob("*dialectical*.md"))
    with Neo4jClient() as client:
        for md in md_files:
            sources = _parse_kg_sources(md)
            if len(sources) < 2:
                print(f"  SKIP (kg_sources < 2): {md.name}")
                continue
            slug = md.stem  # e.g. 2026-05-01-dialectical-wittgenstein-ai-language
            data = export_synthesis_graph(sources, client)
            out  = output_dir / f"{slug}.json"
            out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            n0 = sum(1 for n in data["nodes"] if n["cluster"] == 0)
            n1 = sum(1 for n in data["nodes"] if n["cluster"] == 1)
            nb = sum(1 for n in data["nodes"] if n["cluster"] == 2)
            print(f"  ✓ {out.name}  cluster0={n0} cluster1={n1} bridge={nb} links={len(data['links'])}")
            exported.append(slug)

    return exported


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="合成記事の概念グラフ JSON を出力する")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--docs",   default="../homupe/docs")
    parser.add_argument("--output", default="../homupe/docs/data/synthesis")
    args = parser.parse_args()

    if args.all:
        print(f"合成グラフを {args.output} に出力中...")
        exported = export_all_synthesis(Path(args.docs), Path(args.output))
        print(f"\n完了: {len(exported)} 件")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
