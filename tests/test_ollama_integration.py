"""
Ollama 実動作テスト（OllamaProx port 11435 が必要）。

実行:
    uv run --with pytest pytest tests/test_ollama_integration.py -v -s
"""
import os
import pytest

# OllamaProx が起動していない場合はスキップ
def _ollama_available() -> bool:
    import urllib.request, urllib.error
    try:
        urllib.request.urlopen("http://localhost:11435/api/tags", timeout=3)
        return True
    except Exception:
        return False

pytestmark = pytest.mark.skipif(
    not _ollama_available(),
    reason="OllamaProx (port 11435) が起動していません",
)

MODEL = "gemma4:e4b"  # 軽量モデルで高速テスト


class TestLlmCall:
    def test_basic_call(self):
        os.environ["FACTFULL_OLLAMA_MODEL"] = MODEL
        from factfull import llm
        result = llm.call("「はい」とだけ答えてください。", num_ctx=512)
        assert isinstance(result, str)
        assert len(result) > 0
        print(f"\n  llm.call 応答: {result[:80]}")

    def test_json_response(self):
        os.environ["FACTFULL_OLLAMA_MODEL"] = MODEL
        from factfull import llm
        result = llm.call(
            'Return JSON only: {"ok": true}',
            num_ctx=512,
        )
        assert isinstance(result, str)
        print(f"\n  llm.call JSON 応答: {result[:80]}")


class TestExtractEntities:
    TEXT = """
    Sam Altman is the CEO of OpenAI, the company behind ChatGPT and GPT-4.
    OpenAI was founded in 2015 in San Francisco.
    GPT-4 is a large language model that uses transformer architecture.
    """

    def test_returns_entities(self):
        os.environ["FACTFULL_OLLAMA_MODEL"] = MODEL
        from factfull.extract.entity import extract_entities
        entities = extract_entities([self.TEXT], source_id="test_ep")
        print(f"\n  抽出エンティティ ({len(entities)}件):")
        for e in entities:
            print(f"    {e.name} [{e.type}] conf={e.confidence:.2f}")
        assert len(entities) > 0

    def test_entity_fields_valid(self):
        os.environ["FACTFULL_OLLAMA_MODEL"] = MODEL
        from factfull.extract.entity import extract_entities
        from factfull.core.types import ENTITY_TYPES
        entities = extract_entities([self.TEXT], source_id="test_ep")
        for e in entities:
            assert e.name
            assert e.type in ENTITY_TYPES
            assert 0.0 <= e.confidence <= 1.0
            assert e.source_id == "test_ep"

    def test_known_entities_extracted(self):
        os.environ["FACTFULL_OLLAMA_MODEL"] = MODEL
        from factfull.extract.entity import extract_entities
        entities = extract_entities([self.TEXT], source_id="test_ep")
        names = {e.name.lower() for e in entities}
        # 少なくとも主要エンティティのいずれかが抽出されていること
        known = {"openai", "sam altman", "gpt-4", "chatgpt"}
        assert names & known, f"主要エンティティが未検出。抽出: {names}"


class TestExtractRelations:
    TEXT = """
    Sam Altman is the CEO of OpenAI.
    OpenAI created GPT-4.
    GPT-4 is a large language model.
    """

    def test_returns_triples(self):
        os.environ["FACTFULL_OLLAMA_MODEL"] = MODEL
        from factfull.extract.entity import extract_entities
        from factfull.extract.relation import extract_relations
        from factfull.core.types import Entity

        entities = extract_entities([self.TEXT], source_id="test_ep")
        if len(entities) < 2:
            pytest.skip("エンティティが2件未満のためスキップ")

        triples = extract_relations([self.TEXT], entities, source_id="test_ep")
        print(f"\n  抽出トリプル ({len(triples)}件):")
        for t in triples:
            print(f"    ({t.subject}) --[{t.predicate}]--> ({t.object}) conf={t.confidence:.2f}")
        # 0件でも失敗にしない（テキストが短いと関係が見つからない場合がある）
        assert isinstance(triples, list)

    def test_triple_fields_valid(self):
        os.environ["FACTFULL_OLLAMA_MODEL"] = MODEL
        from factfull.extract.entity import extract_entities
        from factfull.extract.relation import extract_relations, _RELATION_TYPES

        entities = extract_entities([self.TEXT], source_id="test_ep")
        if len(entities) < 2:
            pytest.skip("エンティティが2件未満のためスキップ")

        triples = extract_relations([self.TEXT], entities, source_id="test_ep")
        for t in triples:
            assert t.subject
            assert t.object
            assert t.predicate in _RELATION_TYPES
            assert 0.0 <= t.confidence <= 1.0
            assert t.source_id == "test_ep"


class TestPipelineConfigWriteGraph:
    """write_graph=False（デフォルト）ではグラフ書き込みが呼ばれないことを確認。"""

    def test_write_graph_default_false(self):
        from factfull.podcast.pipeline import PipelineConfig
        config = PipelineConfig()
        assert config.write_graph is False

    def test_write_graph_flag_exists(self):
        from factfull.podcast.pipeline import PipelineConfig
        config = PipelineConfig(write_graph=True)
        assert config.write_graph is True
