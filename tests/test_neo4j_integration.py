"""
Neo4j 統合テスト（factfull-neo4j コンテナが必要）。

起動:
    docker run -d --name factfull-neo4j \
      -p 7474:7474 -p 7687:7687 \
      -e NEO4J_AUTH=neo4j/factfull123 \
      neo4j:5.16-community

実行:
    uv run --with pytest pytest tests/test_neo4j_integration.py -v -s
"""
import os
import pytest

NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "factfull123"


def _neo4j_available() -> bool:
    try:
        from neo4j import GraphDatabase  # type: ignore
        d = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        with d.session() as s:
            s.run("RETURN 1")
        d.close()
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _neo4j_available(),
    reason="Neo4j (bolt://localhost:7687) が起動していません",
)


@pytest.fixture(scope="module")
def client():
    os.environ["NEO4J_URI"] = NEO4J_URI
    os.environ["NEO4J_USER"] = NEO4J_USER
    os.environ["NEO4J_PASSWORD"] = NEO4J_PASSWORD

    from factfull.graph.neo4j import Neo4jClient
    with Neo4jClient() as g:
        # テスト用クリーンアップ（テストノードだけ削除）
        g.run_cypher("MATCH (n) WHERE n.test_marker = true DETACH DELETE n")
        yield g
        g.run_cypher("MATCH (n) WHERE n.test_marker = true DETACH DELETE n")


class TestConnection:
    def test_connect(self, client):
        stats = client.get_statistics()
        assert isinstance(stats, dict)
        assert "sources" in stats
        assert "entities" in stats
        print(f"\n  接続成功: {stats}")

    def test_setup_schema(self, client):
        # 冪等なので何度呼んでも OK
        client.setup_schema()
        client.setup_schema()


class TestUpsertEntity:
    def test_create_entity(self, client):
        from factfull.core.types import Entity
        e = Entity(name="__test_OpenAI__", type="organization",
                   description="Test org", confidence=0.99)
        client.upsert_entity(e)
        # 検索で確認
        results = client.search_entities("__test_OpenAI__")
        assert len(results) == 1
        assert results[0]["name"] == "__test_OpenAI__"
        # テストマーカーを付けてクリーンアップ対象にする
        client.run_cypher(
            "MATCH (e:Entity {name: '__test_OpenAI__'}) SET e.test_marker = true"
        )

    def test_upsert_higher_confidence_wins(self, client):
        from factfull.core.types import Entity
        e_low  = Entity(name="__test_LLM__", type="concept", confidence=0.5)
        e_high = Entity(name="__test_LLM__", type="concept", confidence=0.95)
        client.upsert_entity(e_low)
        client.upsert_entity(e_high)
        results = client.search_entities("__test_LLM__")
        assert results[0]["confidence"] == 0.95
        client.run_cypher(
            "MATCH (e:Entity {name: '__test_LLM__'}) SET e.test_marker = true"
        )


class TestUpsertSource:
    def test_create_source(self, client):
        from factfull.core.types import SourceDoc
        doc = SourceDoc(
            source_type="podcast",
            source_id="__test_ep_001__",
            title="Test Episode",
            text="transcript text",
            metadata={"channel": "TestCast"},
        )
        client.upsert_source(doc)
        result = client.run_cypher(
            "MATCH (s:Source {source_id: '__test_ep_001__'}) RETURN s"
        )
        assert len(result) == 1
        client.run_cypher(
            "MATCH (s:Source {source_id: '__test_ep_001__'}) SET s.test_marker = true"
        )


class TestUpsertTriple:
    def test_create_triple(self, client):
        from factfull.core.types import Entity, Triple
        # エンティティを先に作成
        client.upsert_entity(Entity(name="__test_Sam__", type="person", confidence=0.9))
        client.upsert_entity(Entity(name="__test_Anthropic__", type="organization", confidence=0.9))
        client.run_cypher("MATCH (e:Entity) WHERE e.name IN ['__test_Sam__', '__test_Anthropic__'] SET e.test_marker = true")

        t = Triple(subject="__test_Sam__", predicate="works_at",
                   object="__test_Anthropic__", confidence=0.88)
        client.upsert_triple(t)

        rels = client.get_entity_relations("__test_Sam__")
        assert len(rels) >= 1
        predicates = {r["relation"] for r in rels}
        assert "WORKS_AT" in predicates
        print(f"\n  トリプル確認: {rels}")


class TestWriteProcessedDoc:
    def test_write_full_doc(self, client):
        from factfull.core.types import SourceDoc, ProcessedDoc, Entity, Triple

        source = SourceDoc(
            source_type="paper",
            source_id="__test_paper_arxiv_001__",
            title="Test Paper on Knowledge Graphs",
            text="Knowledge graphs store entities and relations.",
            metadata={"arxiv_id": "0000.00001"},
        )
        entities = [
            Entity(name="__test_KnowledgeGraph__", type="concept", confidence=0.95),
            Entity(name="__test_Entity__", type="concept", confidence=0.90),
        ]
        triples = [
            Triple(subject="__test_KnowledgeGraph__", predicate="part_of",
                   object="__test_Entity__", confidence=0.85),
        ]
        pdoc = ProcessedDoc(source=source, summary="要約テスト", entities=entities,
                            triples=triples, score=88.0)

        client.write_processed_doc(pdoc)

        # Source ノード確認
        src_result = client.run_cypher(
            "MATCH (s:Source {source_id: '__test_paper_arxiv_001__'}) RETURN s.title AS title"
        )
        assert len(src_result) == 1
        assert src_result[0]["title"] == "Test Paper on Knowledge Graphs"

        # MENTIONS リレーション確認
        ents = client.get_source_entities("__test_paper_arxiv_001__")
        entity_names = {e["name"] for e in ents}
        assert "__test_KnowledgeGraph__" in entity_names
        assert "__test_Entity__" in entity_names
        print(f"\n  MENTIONS リレーション: {ents}")

        # トリプル確認
        rels = client.get_entity_relations("__test_KnowledgeGraph__")
        assert len(rels) >= 1
        print(f"\n  Entity 間リレーション: {rels}")

        # テストマーカー付与
        client.run_cypher(
            "MATCH (n) WHERE n.source_id = '__test_paper_arxiv_001__' "
            "OR n.name IN ['__test_KnowledgeGraph__', '__test_Entity__'] "
            "SET n.test_marker = true"
        )


class TestSearchAndQuery:
    def test_search_entities_partial_match(self, client):
        from factfull.core.types import Entity
        client.upsert_entity(Entity(name="__test_Transformer__", type="method", confidence=0.9))
        client.run_cypher("MATCH (e:Entity {name: '__test_Transformer__'}) SET e.test_marker = true")

        results = client.search_entities("__test_Trans")
        names = [r["name"] for r in results]
        assert "__test_Transformer__" in names

    def test_statistics_increments(self, client):
        stats = client.get_statistics()
        assert stats["entities"] > 0
        print(f"\n  グラフ統計: {stats}")
