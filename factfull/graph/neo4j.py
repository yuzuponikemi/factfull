"""
factfull/graph/neo4j.py
========================
Neo4j 知識グラフクライアント。

移植元: kg-builder/src/kg_builder/graph/neo4j_client.py
変更点:
  - Paper/Concept → Source/Entity に汎用化（podcast・書籍・Web も収録）
  - 設定は環境変数から直接取得（kg-builder の get_settings() 依存を排除）
  - write_processed_doc() を追加してパイプライン完了後のワンショット書き込みを実現

グラフスキーマ:
  (:Source {source_type, source_id, title, ...})
  (:Entity {name, type, description, confidence})
  (:Source)-[:MENTIONS {confidence}]->(:Entity)
  (:Entity)-[:IS_A|USES|ENABLES|...]->(:Entity)

環境変数:
  NEO4J_URI      default: bolt://localhost:7687
  NEO4J_USER     default: neo4j
  NEO4J_PASSWORD default: password

使い方:
    from factfull.graph.neo4j import Neo4jClient

    with Neo4jClient() as g:
        g.setup_schema()
        g.write_processed_doc(processed_doc)
        stats = g.get_statistics()
"""
from __future__ import annotations

import logging
import os
from typing import Any

from factfull.core.types import Entity, ProcessedDoc, SourceDoc, Triple

logger = logging.getLogger(__name__)

_DEFAULT_URI = "bolt://localhost:7687"
_DEFAULT_USER = "neo4j"
_DEFAULT_PASSWORD = "password"


class Neo4jClient:
    """factfull の ProcessedDoc を Neo4j グラフに書き込むクライアント。"""

    def __init__(
        self,
        uri: str | None = None,
        username: str | None = None,
        password: str | None = None,
    ) -> None:
        try:
            from neo4j import GraphDatabase  # type: ignore
            from neo4j.exceptions import ServiceUnavailable  # type: ignore
        except ImportError as e:
            raise ImportError("Neo4j 連携には neo4j パッケージが必要です: pip install neo4j") from e

        self.uri = uri or os.environ.get("NEO4J_URI", _DEFAULT_URI)
        self.username = username or os.environ.get("NEO4J_USER", _DEFAULT_USER)
        self.password = password or os.environ.get("NEO4J_PASSWORD", _DEFAULT_PASSWORD)

        self.driver = GraphDatabase.driver(self.uri, auth=(self.username, self.password))
        try:
            with self.driver.session() as s:
                s.run("RETURN 1")
            logger.info("Neo4j に接続しました: %s", self.uri)
        except ServiceUnavailable as e:
            self.driver.close()
            raise ConnectionError(f"Neo4j に接続できません ({self.uri}): {e}") from e

    def close(self) -> None:
        self.driver.close()

    def __enter__(self) -> Neo4jClient:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    # ── スキーマ初期化 ────────────────────────────────────────────────────────

    def setup_schema(self) -> None:
        """制約とインデックスを作成する（冪等）。"""
        statements = [
            "CREATE CONSTRAINT source_id IF NOT EXISTS FOR (s:Source) REQUIRE s.source_id IS UNIQUE",
            "CREATE CONSTRAINT entity_name IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE",
            "CREATE INDEX source_type IF NOT EXISTS FOR (s:Source) ON (s.source_type)",
            "CREATE INDEX entity_type IF NOT EXISTS FOR (e:Entity) ON (e.type)",
        ]
        with self.driver.session() as s:
            for stmt in statements:
                try:
                    s.run(stmt)
                except Exception as e:
                    logger.debug("schema stmt skipped (%s): %s", stmt[:50], e)
        logger.info("スキーマ設定完了")

    # ── 単体書き込み ──────────────────────────────────────────────────────────

    def upsert_source(self, doc: SourceDoc) -> None:
        """SourceDoc を Source ノードとして upsert する。"""
        props = {
            "source_type": doc.source_type,
            "title": doc.title,
            "created_at": doc.created_at,
            **{k: v for k, v in doc.metadata.items() if isinstance(v, (str, int, float, bool))},
        }
        with self.driver.session() as s:
            s.run(
                """
                MERGE (n:Source {source_id: $source_id})
                ON CREATE SET n += $props, n.created_at = datetime()
                ON MATCH  SET n += $props
                """,
                source_id=doc.source_id,
                props=props,
            )

    def upsert_entity(self, entity: Entity) -> None:
        """Entity ノードを upsert する。"""
        with self.driver.session() as s:
            s.run(
                """
                MERGE (e:Entity {name: $name})
                ON CREATE SET e.type = $type, e.description = $desc,
                              e.confidence = $conf, e.created_at = datetime()
                ON MATCH  SET e.confidence = CASE
                                WHEN $conf > e.confidence THEN $conf
                                ELSE e.confidence END,
                             e.description = CASE
                                WHEN $desc IS NOT NULL AND size($desc) > size(coalesce(e.description, '')) THEN $desc
                                ELSE coalesce(e.description, $desc) END
                """,
                name=entity.name,
                type=entity.type,
                desc=entity.description,
                conf=entity.confidence,
            )

    def upsert_triple(self, triple: Triple) -> None:
        """Triple を Entity 間のリレーションシップとして upsert する。"""
        rel_type = triple.predicate.upper().replace("-", "_").replace(" ", "_")
        with self.driver.session() as s:
            s.run(
                f"""
                MATCH (s:Entity {{name: $subject}})
                MATCH (o:Entity {{name: $object}})
                MERGE (s)-[r:{rel_type}]->(o)
                ON CREATE SET r.confidence = $conf, r.source_id = $src, r.created_at = datetime()
                ON MATCH  SET r.confidence = CASE
                                WHEN $conf > r.confidence THEN $conf
                                ELSE r.confidence END
                """,
                subject=triple.subject,
                object=triple.object,
                conf=triple.confidence,
                src=triple.source_id,
            )

    def link_source_to_entity(self, source_id: str, entity_name: str, confidence: float = 1.0) -> None:
        """Source -[:MENTIONS]-> Entity リレーションを作る。"""
        with self.driver.session() as s:
            s.run(
                """
                MATCH (src:Source {source_id: $source_id})
                MATCH (ent:Entity {name: $entity_name})
                MERGE (src)-[r:MENTIONS]->(ent)
                ON CREATE SET r.confidence = $conf, r.created_at = datetime()
                ON MATCH  SET r.confidence = $conf
                """,
                source_id=source_id,
                entity_name=entity_name,
                conf=confidence,
            )

    # ── ワンショット書き込み ──────────────────────────────────────────────────

    def clear_source_relations(self, source_id: str) -> None:
        """このソースが生成したエンティティ間リレーションを全削除する。

        再処理時に古い（低品質な）トリプルを一掃するために使う。
        Source-[:MENTIONS]->Entity は残す。
        """
        with self.driver.session() as s:
            s.run(
                "MATCH ()-[r]->() WHERE r.source_id = $sid DELETE r",
                sid=source_id,
            )

    def write_processed_doc(self, pdoc: ProcessedDoc, clear_old: bool = False) -> None:
        """ProcessedDoc を丸ごとグラフに書き込む。

        1. Source ノード upsert
        2. (clear_old=True の場合) 旧トリプル削除
        3. Entity ノード upsert（全件）
        4. Triple → Entity 間リレーション upsert
        5. Source -[:MENTIONS]-> Entity リレーション作成
        """
        print(f"  [neo4j] 書き込み開始: {pdoc.source.source_id}", flush=True)

        self.upsert_source(pdoc.source)

        if clear_old:
            self.clear_source_relations(pdoc.source.source_id)

        for e in pdoc.entities:
            self.upsert_entity(e)

        for t in pdoc.triples:
            try:
                self.upsert_triple(t)
            except Exception as exc:
                logger.debug("triple upsert skipped (%s→%s): %s", t.subject, t.object, exc)

        for e in pdoc.entities:
            self.link_source_to_entity(pdoc.source.source_id, e.name, e.confidence)

        print(
            f"  [neo4j] 完了: entities={len(pdoc.entities)}, triples={len(pdoc.triples)}",
            flush=True,
        )

    # ── クエリ ────────────────────────────────────────────────────────────────

    def get_statistics(self) -> dict[str, int]:
        """ノード数・リレーション数を返す。"""
        with self.driver.session() as s:
            return {
                "sources":  s.run("MATCH (n:Source) RETURN count(n) AS c").single()["c"],
                "entities": s.run("MATCH (n:Entity) RETURN count(n) AS c").single()["c"],
                "mentions": s.run("MATCH ()-[r:MENTIONS]->() RETURN count(r) AS c").single()["c"],
                "relations": s.run("MATCH ()-[r]->() WHERE type(r) <> 'MENTIONS' RETURN count(r) AS c").single()["c"],
            }

    def search_entities(self, term: str, limit: int = 20) -> list[dict[str, Any]]:
        """エンティティ名の部分一致検索（大文字小文字無視）。"""
        with self.driver.session() as s:
            result = s.run(
                """
                MATCH (e:Entity)
                WHERE toLower(e.name) CONTAINS toLower($term)
                RETURN e.name AS name, e.type AS type, e.confidence AS confidence
                ORDER BY e.confidence DESC LIMIT $limit
                """,
                term=term,
                limit=limit,
            )
            return [dict(r) for r in result]

    def get_entity_relations(self, entity_name: str) -> list[dict[str, Any]]:
        """エンティティに繋がる全リレーションを返す。"""
        with self.driver.session() as s:
            result = s.run(
                """
                MATCH (e:Entity {name: $name})-[r]-(other:Entity)
                RETURN e.name AS source, type(r) AS relation,
                       other.name AS target, r.confidence AS confidence
                """,
                name=entity_name,
            )
            return [dict(r) for r in result]

    def get_source_entities(self, source_id: str) -> list[dict[str, Any]]:
        """ソースが MENTIONS しているエンティティ一覧を返す。"""
        with self.driver.session() as s:
            result = s.run(
                """
                MATCH (s:Source {source_id: $source_id})-[r:MENTIONS]->(e:Entity)
                RETURN e.name AS name, e.type AS type, r.confidence AS confidence
                ORDER BY r.confidence DESC
                """,
                source_id=source_id,
            )
            return [dict(r) for r in result]

    def run_cypher(self, query: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """任意の Cypher クエリを実行する。"""
        with self.driver.session() as s:
            return [dict(r) for r in s.run(query, params or {})]
