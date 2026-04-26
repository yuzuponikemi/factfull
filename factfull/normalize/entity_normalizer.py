"""
factfull/normalize/entity_normalizer.py
=========================================
Neo4j の Entity ノードを Wikipedia でバッチ正規化する。

対象: person / organization / concept / framework エンティティ
処理: canonical_name への改名 + wikipedia_url / wikidata_qid の付与

使い方:
    python -m factfull.normalize.entity_normalizer --dry-run
    python -m factfull.normalize.entity_normalizer --types person,organization
"""
from __future__ import annotations

import argparse
from typing import Sequence

from factfull.graph.neo4j import Neo4jClient
from factfull.normalize.wiki_linker import WikiLinker


_DEFAULT_TYPES = ("person", "organization")
_MIN_CONFIDENCE = 0.7


def normalize_entities(
    client: Neo4jClient,
    linker: WikiLinker,
    types: Sequence[str] = _DEFAULT_TYPES,
    dry_run: bool = False,
    min_confidence: float = _MIN_CONFIDENCE,
    limit: int = 500,
) -> dict[str, int]:
    """Neo4j のエンティティを Wikipedia でバッチ正規化する。

    Returns:
        stats dict: linked / skipped / renamed / failed の件数
    """
    stats = {"linked": 0, "skipped": 0, "renamed": 0, "failed": 0}

    # wikipedia_url が未設定のエンティティのみ対象
    type_list = list(types)
    rows = client.run_cypher(
        """
        MATCH (e:Entity)
        WHERE e.type IN $types AND (e.wikipedia_url IS NULL OR e.wikipedia_url = '')
        RETURN e.name AS name, e.type AS type
        ORDER BY e.confidence DESC
        LIMIT $limit
        """,
        {"types": type_list, "limit": limit},
    )

    total = len(rows)
    print(f"  対象エンティティ: {total} 件 (types={type_list})", flush=True)

    for i, row in enumerate(rows, 1):
        name = row["name"]
        etype = row["type"]
        try:
            result = linker.link(name)
            if not result.found or result.confidence < min_confidence:
                stats["skipped"] += 1
                continue

            stats["linked"] += 1
            canonical = result.canonical_name
            renamed = canonical != name

            if dry_run:
                flag = "RENAME" if renamed else "OK"
                print(
                    f"  [{i}/{total}] {flag}  '{name}' → '{canonical}'"
                    f"  (conf={result.confidence:.2f}"
                    + (f", qid={result.wikidata_qid}" if result.wikidata_qid else "")
                    + ")",
                    flush=True,
                )
                if renamed:
                    stats["renamed"] += 1
                continue

            # Neo4j 更新
            if renamed:
                # 旧ノードを削除して正規名でマージ
                # (MENTIONS と ARGUES_THAT も付け替える)
                client.run_cypher(
                    """
                    MATCH (old:Entity {name: $old_name})
                    MERGE (new:Entity {name: $new_name})
                    ON CREATE SET new.type = old.type,
                                  new.confidence = old.confidence,
                                  new.description = old.description
                    ON MATCH  SET new.confidence = CASE
                                    WHEN old.confidence > new.confidence THEN old.confidence
                                    ELSE new.confidence END
                    WITH old, new
                    CALL {
                        WITH old, new
                        MATCH (src)-[r:MENTIONS]->(old)
                        MERGE (src)-[:MENTIONS]->(new)
                        DELETE r
                    }
                    CALL {
                        WITH old, new
                        MATCH (p)-[r:ARGUES_THAT]->(old)
                        MERGE (p)-[:ARGUES_THAT]->(new)
                        DELETE r
                    }
                    DELETE old
                    """,
                    {"old_name": name, "new_name": canonical},
                )
                stats["renamed"] += 1

            # Wikipedia メタデータを付与
            client.run_cypher(
                """
                MATCH (e:Entity {name: $name})
                SET e.wikipedia_url = $url,
                    e.wikidata_qid  = $qid
                """,
                {
                    "name": canonical,
                    "url": result.wikipedia_url,
                    "qid": result.wikidata_qid or "",
                },
            )

            print(
                f"  [{i}/{total}] {'RENAME' if renamed else 'OK  '}"
                f"  '{name}' → '{canonical}'"
                f"  conf={result.confidence:.2f}"
                + (f"  qid={result.wikidata_qid}" if result.wikidata_qid else ""),
                flush=True,
            )

        except Exception as e:
            print(f"  [{i}/{total}] FAIL  '{name}': {e}", flush=True)
            stats["failed"] += 1

    return stats


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Neo4j エンティティを Wikipedia で正規化")
    parser.add_argument("--types", default="person,organization",
                        help="対象エンティティタイプ（カンマ区切り）")
    parser.add_argument("--dry-run", action="store_true",
                        help="変更を加えず結果だけ表示")
    parser.add_argument("--min-confidence", type=float, default=_MIN_CONFIDENCE,
                        help=f"リンク信頼度の下限（デフォルト: {_MIN_CONFIDENCE}）")
    parser.add_argument("--limit", type=int, default=500,
                        help="処理件数の上限")
    parser.add_argument("--no-qid", action="store_true",
                        help="Wikidata QID 取得をスキップ（オフライン時）")
    args = parser.parse_args()

    types = [t.strip() for t in args.types.split(",")]

    with Neo4jClient() as client, WikiLinker() as linker:
        stats = normalize_entities(
            client, linker,
            types=types,
            dry_run=args.dry_run,
            min_confidence=args.min_confidence,
            limit=args.limit,
        )

    print(f"\n結果: {stats}")


if __name__ == "__main__":
    main()
