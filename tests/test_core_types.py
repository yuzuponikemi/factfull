"""factfull/core/types.py のテスト。"""
import pytest
from factfull.core.types import (
    SourceDoc, ProcessedDoc, Entity, Triple,
    SOURCE_TYPES, ENTITY_TYPES,
)


def make_source_doc(**kwargs) -> SourceDoc:
    defaults = dict(
        source_type="podcast",
        source_id="abc123",
        title="Test Episode",
        text="This is a test transcript.",
        text_ja="これはテストのトランスクリプト。",
        chunks=["chunk1", "chunk2"],
        metadata={"channel": "Test Channel"},
    )
    return SourceDoc(**{**defaults, **kwargs})


class TestSourceDoc:
    def test_defaults(self):
        doc = make_source_doc()
        assert doc.source_type == "podcast"
        assert doc.source_id == "abc123"
        assert doc.created_at  # 自動セット

    def test_roundtrip(self):
        doc = make_source_doc()
        restored = SourceDoc.from_dict(doc.to_dict())
        assert restored.source_id == doc.source_id
        assert restored.title == doc.title
        assert restored.chunks == doc.chunks
        assert restored.metadata == doc.metadata

    def test_source_types_valid(self):
        for st in SOURCE_TYPES:
            doc = make_source_doc(source_type=st)
            assert doc.source_type == st

    def test_empty_text_allowed(self):
        doc = make_source_doc(text="", text_ja="")
        assert doc.text == ""

    def test_metadata_preserved(self):
        meta = {"channel": "Lex", "duration": 7200, "score": 0.95, "live": False}
        doc = make_source_doc(metadata=meta)
        assert SourceDoc.from_dict(doc.to_dict()).metadata == meta


class TestEntity:
    def test_defaults(self):
        e = Entity(name="GPT-4", type="product")
        assert e.confidence == 1.0
        assert e.description == ""
        assert e.source_id == ""

    def test_roundtrip(self):
        e = Entity(name="LLM", type="concept", description="Large language model", confidence=0.95, source_id="src1")
        e2 = Entity.from_dict(e.to_dict())
        assert e2.name == e.name
        assert e2.confidence == e.confidence
        assert e2.source_id == e.source_id

    def test_all_entity_types(self):
        for et in ENTITY_TYPES:
            e = Entity(name="test", type=et)
            assert e.type == et


class TestTriple:
    def test_defaults(self):
        t = Triple(subject="A", predicate="uses", object="B")
        assert t.confidence == 1.0
        assert t.source_id == ""

    def test_roundtrip(self):
        t = Triple(subject="X", predicate="is_a", object="Y", confidence=0.9, source_id="src")
        t2 = Triple.from_dict(t.to_dict())
        assert t2.subject == "X"
        assert t2.predicate == "is_a"
        assert t2.confidence == 0.9


class TestProcessedDoc:
    def test_roundtrip_full(self):
        doc = make_source_doc()
        entities = [
            Entity(name="LLM", type="concept", confidence=0.95),
            Entity(name="OpenAI", type="organization", confidence=0.99),
        ]
        triples = [
            Triple(subject="LLM", predicate="related_to", object="OpenAI", confidence=0.8),
        ]
        pdoc = ProcessedDoc(
            source=doc,
            summary="要約テキスト",
            entities=entities,
            triples=triples,
            score=92.5,
        )
        d = pdoc.to_dict()
        pdoc2 = ProcessedDoc.from_dict(d)

        assert pdoc2.source.source_id == "abc123"
        assert len(pdoc2.entities) == 2
        assert pdoc2.entities[0].name == "LLM"
        assert len(pdoc2.triples) == 1
        assert pdoc2.triples[0].predicate == "related_to"
        assert pdoc2.score == 92.5
        assert pdoc2.summary == "要約テキスト"

    def test_empty_entities_and_triples(self):
        pdoc = ProcessedDoc(source=make_source_doc())
        d = pdoc.to_dict()
        pdoc2 = ProcessedDoc.from_dict(d)
        assert pdoc2.entities == []
        assert pdoc2.triples == []
        assert pdoc2.score == 0.0
