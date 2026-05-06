"""
factfull/export/book_graph.py
==============================
Neo4j の書籍サブグラフを homupe 向け JSON に出力する。

CLI:
    uv run python -m factfull.export.book_graph --all --output ../homupe/docs/data/kg/
    uv run python -m factfull.export.book_graph --source-id book_wittgenstein_... --output ../homupe/docs/data/kg/
"""
from __future__ import annotations

import json
import os
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


def _shorten_label(name: str, etype: str) -> str:
    if name.startswith("[Aporia] "):
        body = name[9:]
        return "❓ " + (body[:55] + "…" if len(body) > 55 else body)
    if len(name) > 60:
        return name[:58] + "…"
    return name


def export_book_graph(source_id: str, client: Neo4jClient) -> dict[str, Any]:
    """1書籍の概念グラフを nodes/links 形式に変換する。"""
    source_r = client.run_cypher(
        "MATCH (s:Source {source_id: $sid}) RETURN s.title as title",
        {"sid": source_id},
    )
    title = source_r[0]["title"] if source_r else source_id

    nodes_r = client.run_cypher(
        """
        MATCH (s:Source {source_id: $sid})-[:MENTIONS]->(e:Entity)
        RETURN e.name as name, e.type as type, e.description as desc
        """,
        {"sid": source_id},
    )

    node_ids = {r["name"] for r in nodes_r}
    nodes = [
        {
            "id":          r["name"],
            "label":       _shorten_label(r["name"], r["type"]),
            "type":        r["type"],
            "description": (r["desc"] or ""),
            "color":       _NODE_COLORS.get(r["type"], "#888888"),
        }
        for r in nodes_r
    ]

    edges_r = client.run_cypher(
        """
        MATCH (s:Source {source_id: $sid})-[:MENTIONS]->(a:Entity)-[r]->(b:Entity)
        WHERE b.name IN $node_ids
        RETURN a.name as src, type(r) as rel, b.name as tgt
        """,
        {"sid": source_id, "node_ids": list(node_ids)},
    )

    links = [
        {
            "source": r["src"],
            "target": r["tgt"],
            "type":   r["rel"],
            "color":  _EDGE_COLORS.get(r["rel"], "#AAAAAA"),
        }
        for r in edges_r
    ]

    return {
        "source_id": source_id,
        "title":     title,
        "nodes":     nodes,
        "links":     links,
    }


def export_all_sources(
    output_dir: Path,
    source_type: str    = "book",
    neo4j_uri: str      = "bolt://localhost:7687",
    neo4j_user: str     = "neo4j",
    neo4j_password: str = "factfull123",
) -> list[str]:
    """指定 source_type の全ソースを JSON ファイルに書き出す。"""
    os.environ.setdefault("NEO4J_URI", neo4j_uri)
    os.environ.setdefault("NEO4J_USER", neo4j_user)
    os.environ.setdefault("NEO4J_PASSWORD", neo4j_password)

    output_dir.mkdir(parents=True, exist_ok=True)
    exported: list[str] = []

    with Neo4jClient() as client:
        sources_r = client.run_cypher(
            "MATCH (s:Source {source_type: $stype}) RETURN s.source_id as sid ORDER BY sid",
            {"stype": source_type},
        )
        for row in sources_r:
            sid = row["sid"]
            data = export_book_graph(sid, client)
            out_path = output_dir / f"{sid}.json"
            out_path.write_text(
                json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            print(f"  ✓ {out_path.name}  nodes={len(data['nodes'])}  links={len(data['links'])}")
            exported.append(sid)

    return exported


def export_all_books(
    output_dir: Path,
    neo4j_uri: str      = "bolt://localhost:7687",
    neo4j_user: str     = "neo4j",
    neo4j_password: str = "factfull123",
) -> list[str]:
    return export_all_sources(output_dir, "book", neo4j_uri, neo4j_user, neo4j_password)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="概念グラフ JSON を出力する")
    parser.add_argument("--all", action="store_true", help="全ソースを出力")
    parser.add_argument("--source-id", help="特定の source_id")
    parser.add_argument("--source-type", default="book", help="source_type フィルタ (default: book)")
    parser.add_argument(
        "--output",
        default="../homupe/docs/data/kg",
        help="出力ディレクトリ (default: ../homupe/docs/data/kg/)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output)

    if args.all:
        print(f"{args.source_type} ソースを {output_dir} に出力中...")
        exported = export_all_sources(output_dir, args.source_type)
        print(f"\n完了: {len(exported)} 件")

    elif args.source_id:
        os.environ.setdefault("NEO4J_URI", "bolt://localhost:7687")
        os.environ.setdefault("NEO4J_USER", "neo4j")
        os.environ.setdefault("NEO4J_PASSWORD", "factfull123")
        output_dir.mkdir(parents=True, exist_ok=True)
        with Neo4jClient() as client:
            data = export_book_graph(args.source_id, client)
        out_path = output_dir / f"{args.source_id}.json"
        out_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"完了: {out_path}  nodes={len(data['nodes'])}  links={len(data['links'])}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
