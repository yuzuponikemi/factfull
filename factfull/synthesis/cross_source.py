"""
factfull/synthesis/cross_source.py
====================================
複数ソースのナレッジグラフから横断的な統合論考を生成する。

フロー:
  1. 話者別 claim（具体的な主張文）を取得
  2. 共通エンティティ・クロスリレーションを取得
  3. LLM に「誰が何を主張したか」を渡して論文スタイルで統合
"""
from __future__ import annotations

from typing import Any

from factfull.graph.neo4j import Neo4jClient


# ── Cypher クエリ ──────────────────────────────────────────────────────────────

# description が "[Speaker Name]" で始まる全エンティティを話者帰属として取得
# entity type 名の揺れ（claim/theory/argument 等）に依存しない
_Q_SPEAKER_CLAIMS = """
MATCH (s:Source)-[:MENTIONS]->(c:Entity)
MATCH (s)-[:MENTIONS]->(p:Entity {type: 'person'})
WHERE c.description IS NOT NULL
  AND c.description STARTS WITH '[' + p.name + ']'
  AND NOT (c.type IN ['person', 'organization', 'place', 'event'])
RETURN p.name AS speaker,
       c.name AS claim,
       c.type AS entity_type,
       c.description AS description,
       s.title AS source_title,
       s.source_id AS source_id
ORDER BY p.name, c.confidence DESC
LIMIT $limit
"""

# concept/framework 系（description に [Speaker] があるもの）
_Q_SPEAKER_CONCEPTS = """
MATCH (s:Source)-[:MENTIONS]->(c:Entity)
MATCH (s)-[:MENTIONS]->(p:Entity {type: 'person'})
WHERE c.description IS NOT NULL
  AND c.description CONTAINS '[' + p.name + ']'
  AND NOT c.description STARTS WITH '[' + p.name + ']'
  AND NOT (c.type IN ['person', 'organization', 'place', 'event'])
RETURN p.name AS speaker,
       c.name AS concept,
       c.type AS type,
       c.description AS description,
       s.title AS source_title
ORDER BY p.name, c.confidence DESC
LIMIT $limit
"""

# 複数ソース共通エンティティ
_Q_SHARED_ENTITIES = """
MATCH (s:Source)-[:MENTIONS]->(e:Entity)
WITH e, collect(DISTINCT s) AS sources, count(DISTINCT s) AS num_sources
WHERE num_sources >= $min_sources
RETURN e.name AS name,
       e.type AS type,
       e.description AS description,
       num_sources,
       [src IN sources | src.title] AS source_titles
ORDER BY num_sources DESC, e.confidence DESC
LIMIT $limit
"""

_Q_ALL_SOURCES = """
MATCH (s:Source)
RETURN s.source_id AS source_id, s.title AS title, s.source_type AS source_type
ORDER BY s.created_at
"""


# ── 公開 API ───────────────────────────────────────────────────────────────────

def find_shared_entities(
    client: Neo4jClient,
    min_sources: int = 2,
    limit: int = 30,
) -> list[dict[str, Any]]:
    """min_sources 以上のソースに登場するエンティティを返す。"""
    return client.run_cypher(_Q_SHARED_ENTITIES, {"min_sources": min_sources, "limit": limit})


def synthesize(
    client: Neo4jClient,
    model: str,
    min_sources: int = 2,
    topic: str | None = None,
) -> str:
    """複数ソース横断の統合論考を生成して Markdown 文字列で返す。"""

    all_sources   = client.run_cypher(_Q_ALL_SOURCES)
    speaker_claims = client.run_cypher(_Q_SPEAKER_CLAIMS, {"limit": 120})
    speaker_concepts = client.run_cypher(_Q_SPEAKER_CONCEPTS, {"limit": 60})
    shared        = find_shared_entities(client, min_sources=min_sources)

    prompt = _build_prompt(all_sources, speaker_claims, speaker_concepts, shared, topic)

    from factfull import llm
    return llm.call(prompt, num_ctx=32768, num_predict=12000, timeout=3600, model=model)


# ── 内部ヘルパー ───────────────────────────────────────────────────────────────

def _build_prompt(
    all_sources: list[dict],
    speaker_claims: list[dict],
    speaker_concepts: list[dict],
    shared: list[dict],
    topic: str | None,
) -> str:
    # ソースを番号付きリストに
    numbered_sources = list(enumerate(all_sources, 1))
    src_index = {s["source_id"]: i for i, s in numbered_sources}
    src_lines = "\n".join(f"[{i}] {s['title']}" for i, s in numbered_sources)

    # 話者別 claim をグループ化
    claims_by_speaker: dict[str, list[dict]] = {}
    for row in speaker_claims:
        sp = row["speaker"]
        claims_by_speaker.setdefault(sp, []).append(row)

    claim_lines_parts = []
    for speaker, claims in claims_by_speaker.items():
        claim_lines_parts.append(f"\n**{speaker}**")
        for c in claims[:8]:
            desc = (c.get("description") or "")[:120]
            ref = f"[{src_index.get(c['source_id'], '?')}]"
            claim_lines_parts.append(f"  - {c['claim']} {ref}")
            if desc:
                claim_lines_parts.append(f"    （{desc}）")
    claim_lines = "\n".join(claim_lines_parts) or "  （なし）"

    # 話者別 concept/framework
    concepts_by_speaker: dict[str, list[dict]] = {}
    for row in speaker_concepts:
        concepts_by_speaker.setdefault(row["speaker"], []).append(row)

    concept_lines_parts = []
    for speaker, concepts in concepts_by_speaker.items():
        concept_lines_parts.append(f"\n**{speaker}**")
        for c in concepts[:5]:
            desc = (c.get("description") or "")[:80]
            concept_lines_parts.append(f"  - [{c['type']}] {c['concept']}" + (f"：{desc}" if desc else ""))
    concept_lines = "\n".join(concept_lines_parts) or "  （なし）"

    # 複数ソース共通エンティティ（上位15件）
    shared_lines = "\n".join(
        f"  - **{e['name']}** [{e['type']}] — {e['num_sources']}ソース共通"
        + (f"：{(e['description'] or '')[:60]}" if e.get("description") else "")
        for e in shared[:15]
    ) or "  （なし）"

    topic_line = f"テーマ: 「{topic}」\n\n" if topic else ""

    return f"""あなたは複数の情報源を統合して独自の洞察を生み出す知識人です。
以下のナレッジグラフから抽出した「話者別の具体的な主張」を素材に、学術論文スタイルの統合論考を日本語で書いてください。

{topic_line}---

## 参考ソース一覧
{src_lines}

---

## 話者別・具体的な主張（claim）
※ これが本論の一次素材です。各話者が何を主張したかを比較・対比・統合してください。
{claim_lines}

---

## 話者別・提唱した概念・フレームワーク
{concept_lines}

---

## 複数ソースに共通するキーエンティティ（横断的テーマの手がかり）
{shared_lines}

---

## 出力フォーマット（Markdown・論文スタイル）

以下の構成で、**6,000〜9,000文字**の論考を生成してください。

### 構成
1. `## アブストラクト` — 論考全体の主張を4〜6文で提示。何が明らかになるかを宣言する。
2. `## 本論`（4〜5セクション、`###` サブ見出し）
   - **各セクションで話者の主張を具体的に引用・比較すること**
   - 単純な紹介や列挙ではなく、主張間の矛盾・補強・対立を論じること
   - 複数のソースを横断して見えてくる「単一ソースでは見えなかった構造」を示すこと
3. `## 結論` — 統合的洞察と読者への問い

### 引用ルール
- 本文中の引用は `[番号]` のみ（タイトルを本文に埋め込まない）
- 末尾に `## 参考文献` セクションを設け `[番号] タイトル` 形式で列挙
- 一般論・背景説明には番号不要

### 文体・品質基準
- 固有名詞・概念名は太字（**名前**）
- 客観的・分析的トーン
- 「AはXと主張するが、BはYと反論する。この対立は〜を示唆する」のように論を展開する
- 話者の主張を要約するだけでなく、**なぜそれが重要か・何と矛盾するか**を論じること
"""
