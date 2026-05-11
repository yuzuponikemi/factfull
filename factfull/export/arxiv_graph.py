"""
factfull/export/arxiv_graph.py
================================
arXiv ダイジェスト用の統合 KG JSON を生成する。

複数論文のエンティティを1グラフに統合し、
各ノードに `papers` 配列（paper_id / title / url）を埋め込む。

KG JSON 形式:
  {
    "digest_id": "arxiv_digest_20260507",
    "nodes": [
      {
        "id": "Chain-of-Thought",
        "label": "Chain-of-Thought",
        "type": "method",
        "description": "...",
        "color": "#3498DB",
        "papers": [
          {"paper_id": "2506.12345", "title": "...", "url": "https://arxiv.org/abs/2506.12345"}
        ]
      }, ...
    ],
    "links": [
      {"source": "...", "target": "...", "type": "RELATED_TO", "color": "#666666"}
    ]
  }
"""
from __future__ import annotations

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

_EDGE_COLORS: dict[str, str] = {
    "DEPENDS_ON":   "#888888",
    "CONTRADICTS":  "#E74C3C",
    "EVOLVES_INTO": "#2ECC71",
    "AUTHORED":     "#4A90D9",
    "ARGUES_THAT":  "#F5A623",
    "SAYS":         "#9B59B6",
    "RELATED_TO":   "#666666",
    "PART_OF":      "#555555",
}


def _shorten_label(name: str) -> str:
    if len(name) > 60:
        return name[:58] + "…"
    return name


def export_arxiv_digest_graph(
    paper_source_ids: list[str],
    digest_id: str,
    client: Neo4jClient,
) -> dict[str, Any]:
    """複数の arXiv 論文エンティティを統合した KG JSON を生成する。

    Args:
        paper_source_ids: Neo4j の source_id リスト (例: ["arxiv_2506.12345", ...])
        digest_id: ダイジスト識別子 ("arxiv_digest_20260507")
        client: Neo4jClient インスタンス

    Returns:
        nodes / links を持つ dict
    """
    if not paper_source_ids:
        return {"digest_id": digest_id, "nodes": [], "links": []}

    # エンティティ + 所属論文を一括取得
    rows = client.run_cypher(
        """
        MATCH (s:Source)-[:MENTIONS]->(e:Entity)
        WHERE s.source_id IN $sids
        RETURN
            e.name        AS name,
            e.type        AS type,
            e.description AS desc,
            s.source_id   AS source_id,
            s.title       AS source_title
        """,
        {"sids": paper_source_ids},
    )

    # エンティティ名をキーに集約
    entity_types: dict[str, str] = {}
    entity_descs: dict[str, str] = {}
    entity_papers: dict[str, list[dict]] = {}

    for r in rows:
        name = r["name"]
        if name not in entity_types:
            entity_types[name] = r["type"] or "concept"
            entity_descs[name] = r["desc"] or ""
            entity_papers[name] = []

        # paper_id: "arxiv_2506.12345" → "2506.12345"
        raw_sid = r["source_id"] or ""
        paper_id = raw_sid.removeprefix("arxiv_")
        paper_info = {
            "paper_id": paper_id,
            "title":    r["source_title"] or paper_id,
            "url":      f"https://arxiv.org/abs/{paper_id}",
        }
        if paper_info not in entity_papers[name]:
            entity_papers[name].append(paper_info)

    node_ids = set(entity_types.keys())
    nodes = [
        {
            "id":          name,
            "label":       _shorten_label(name),
            "type":        entity_types[name],
            "description": entity_descs[name],
            "color":       _NODE_COLORS.get(entity_types[name], "#888888"),
            "papers":      entity_papers[name],
        }
        for name in node_ids
    ]

    # エッジ
    edge_rows = client.run_cypher(
        """
        MATCH (s:Source)-[:MENTIONS]->(a:Entity)-[r]->(b:Entity)
        WHERE s.source_id IN $sids AND b.name IN $node_ids
        RETURN a.name AS src, type(r) AS rel, b.name AS tgt
        """,
        {"sids": paper_source_ids, "node_ids": list(node_ids)},
    )

    links = [
        {
            "source": r["src"],
            "target": r["tgt"],
            "type":   r["rel"],
            "color":  _EDGE_COLORS.get(r["rel"], "#AAAAAA"),
        }
        for r in edge_rows
    ]

    return {
        "digest_id": digest_id,
        "nodes":     nodes,
        "links":     links,
    }
