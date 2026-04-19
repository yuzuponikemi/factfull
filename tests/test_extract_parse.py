"""factfull/extract/{entity,relation}.py の _parse 関数テスト（LLM 不要）。"""
import pytest
from factfull.core.types import Entity, Triple


class TestEntityParse:
    def _parse(self, raw, source_id="src"):
        from factfull.extract.entity import _parse
        return _parse(raw, source_id)

    def test_valid_json(self):
        raw = '{"entities":[{"name":"GPT-4","type":"product","description":"LLM","confidence":0.95}]}'
        entities = self._parse(raw)
        assert len(entities) == 1
        assert entities[0].name == "GPT-4"
        assert entities[0].type == "product"
        assert entities[0].confidence == 0.95

    def test_source_id_set(self):
        raw = '{"entities":[{"name":"X","type":"concept","description":"","confidence":0.8}]}'
        entities = self._parse(raw, source_id="episode_001")
        assert entities[0].source_id == "episode_001"

    def test_invalid_type_falls_back_to_concept(self):
        raw = '{"entities":[{"name":"Foo","type":"INVALID_TYPE","description":"","confidence":0.8}]}'
        entities = self._parse(raw)
        assert entities[0].type == "concept"

    def test_invalid_confidence_falls_back(self):
        raw = '{"entities":[{"name":"Bar","type":"person","description":"","confidence":"high"}]}'
        entities = self._parse(raw)
        assert entities[0].confidence == 0.8  # デフォルト

    def test_missing_name_skipped(self):
        raw = '{"entities":[{"name":"","type":"concept","description":"","confidence":0.9}]}'
        entities = self._parse(raw)
        assert entities == []

    def test_no_json_returns_empty(self):
        assert self._parse("Sorry, I cannot extract entities.") == []

    def test_json_embedded_in_text(self):
        raw = 'Sure! Here you go:\n{"entities":[{"name":"Neo4j","type":"product","description":"Graph DB","confidence":0.99}]}\nDone.'
        entities = self._parse(raw)
        assert len(entities) == 1
        assert entities[0].name == "Neo4j"

    def test_all_valid_entity_types(self):
        from factfull.core.types import ENTITY_TYPES
        for et in ENTITY_TYPES:
            raw = f'{{"entities":[{{"name":"test","type":"{et}","description":"","confidence":0.9}}]}}'
            entities = self._parse(raw)
            assert entities[0].type == et


class TestRelationParse:
    def _parse(self, raw, entity_names=None, source_id="src"):
        from factfull.extract.relation import _parse
        if entity_names is None:
            entity_names = ["LLM", "OpenAI", "GPT-4"]
        entities = [Entity(name=n, type="concept") for n in entity_names]
        return _parse(raw, entities, source_id)

    def test_valid_json(self):
        raw = '{"relationships":[{"from":"LLM","to":"GPT-4","type":"is_a","confidence":0.9}]}'
        triples = self._parse(raw)
        assert len(triples) == 1
        assert triples[0].subject == "LLM"
        assert triples[0].predicate == "is_a"
        assert triples[0].object == "GPT-4"

    def test_entity_not_in_list_skipped(self):
        raw = '{"relationships":[{"from":"Unknown","to":"GPT-4","type":"uses","confidence":0.9}]}'
        triples = self._parse(raw)
        assert triples == []

    def test_self_loop_skipped(self):
        raw = '{"relationships":[{"from":"LLM","to":"LLM","type":"related_to","confidence":0.9}]}'
        triples = self._parse(raw)
        assert triples == []

    def test_invalid_type_falls_back_to_related_to(self):
        raw = '{"relationships":[{"from":"LLM","to":"OpenAI","type":"INVALID","confidence":0.8}]}'
        triples = self._parse(raw)
        assert triples[0].predicate == "related_to"

    def test_no_json_returns_empty(self):
        assert self._parse("No relationships found.") == []

    def test_source_id_set(self):
        raw = '{"relationships":[{"from":"LLM","to":"OpenAI","type":"related_to","confidence":0.8}]}'
        triples = self._parse(raw, source_id="ep_123")
        assert triples[0].source_id == "ep_123"


class TestEntityDeduplication:
    """extract_entities の重複除去ロジックを _parse を使って検証。"""

    def test_higher_confidence_wins(self):
        from factfull.extract.entity import _parse

        e_low  = _parse('{"entities":[{"name":"LLM","type":"concept","description":"v1","confidence":0.7}]}', "src")
        e_high = _parse('{"entities":[{"name":"LLM","type":"concept","description":"v2","confidence":0.95}]}', "src")

        seen: dict[str, Entity] = {}
        for e in e_low + e_high:
            key = e.name.lower()
            if key not in seen or e.confidence > seen[key].confidence:
                seen[key] = e

        assert seen["llm"].confidence == 0.95
        assert seen["llm"].description == "v2"
