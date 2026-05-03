"""
factfull/extract/relation.py
=============================
エンティティ間の関係を抽出してトリプルを生成する。

移植元: kg-builder/src/kg_builder/extractor/relation_extractor.py
変更点:
  - factfull.llm を使用
  - 返り値が factfull.core.types.Triple のリスト
  - 関係タイプに汎用型（says / mentions / created / about）を追加

使い方:
    from factfull.extract.relation import extract_relations
    from factfull.extract.entity import extract_entities

    entities = extract_entities(doc.chunks)
    triples = extract_relations(doc.chunks, entities, source_id=doc.source_id)
"""
from __future__ import annotations

import json
import re

from factfull.core.types import Entity, Triple
from factfull import llm

_RELATION_TYPES = (
    # 科学・技術（kg-builder 互換）
    "is_a", "part_of", "uses", "enables", "measures", "applies_to", "based_on",
    # 汎用
    "related_to", "created", "says", "mentions", "about", "works_at", "located_in",
)

_PROMPT = """\
あなたは知識グラフ構築の専門家です。以下のテキストと既知のエンティティリストを読み、
エンティティ間の関係（トリプル）を抽出してください。

## 既知のエンティティ
{entities}

## 関係タイプ
- is_a: X は Y の一種（階層関係）
- part_of: X は Y の一部
- uses: X は Y を使用・採用している
- enables: X は Y を可能にする・改善する
- measures: X は Y を測定・定量化する
- applies_to: X は Y に適用される
- based_on: X は Y に基づいている・派生している
- related_to: 上記に当てはまらない一般的な関連
- created: X は Y を作成・開発した
- says: X は Y について発言した（Podcast / 書籍向け）
- mentions: X は Y に言及した
- about: X は Y についての内容である
- works_at: X は Y に所属・勤務している
- located_in: X は Y に位置する

## ガイドライン
- テキストに明示的に述べられている関係のみ抽出する
- from / to はエンティティリストの名前と完全一致させる
- 自己ループ（from == to）は除外
- confidence: 関係の明確さを 0.0〜1.0 で評価

## 対象テキスト
{text}

## 出力形式（JSON のみ）
{{
  "relationships": [
    {{
      "from": "エンティティ名",
      "to": "エンティティ名",
      "type": "関係タイプ",
      "confidence": 0.90
    }}
  ]
}}
"""


def extract_relations(
    chunks: list[str],
    entities: list[Entity],
    source_id: str = "",
    model: str | None = None,
) -> list[Triple]:
    """テキストチャンクとエンティティリストから関係トリプルを抽出する。

    Args:
        chunks: テキストチャンクのリスト
        entities: extract_entities() が返したエンティティリスト
        source_id: どの SourceDoc から来たか
        model: Ollama モデル名

    Returns:
        Triple のリスト（重複除去済み、confidence 最大値を採用）
    """
    if not entities:
        return []

    entity_list = "\n".join(f"- {e.name} ({e.type})" for e in entities)
    seen: dict[str, Triple] = {}  # "from|type|to" → Triple

    for i, chunk in enumerate(chunks, 1):
        print(f"  [relation] チャンク {i}/{len(chunks)} 処理中...", flush=True)
        prompt = _PROMPT.format(entities=entity_list, text=chunk[:6000])
        raw = llm.call(prompt, num_ctx=8192, model=model)
        triples = _parse(raw, entities, source_id)

        for t in triples:
            key = f"{t.subject.lower()}|{t.predicate}|{t.object.lower()}"
            if key not in seen or t.confidence > seen[key].confidence:
                seen[key] = t

    result = list(seen.values())
    print(f"  [relation] 抽出完了: {len(result)} トリプル", flush=True)
    return result


def _parse(raw: str, entities: list[Entity], source_id: str) -> list[Triple]:
    """LLM 出力から Triple リストを取り出す。"""
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        return []
    try:
        data = json.loads(m.group())
    except json.JSONDecodeError:
        return []

    valid_names = {e.name.lower() for e in entities}
    triples: list[Triple] = []

    for item in data.get("relationships", []):
        frm = (item.get("from") or "").strip()
        to = (item.get("to") or "").strip()
        rel_type = (item.get("type") or "related_to").strip().lower()

        if not frm or not to:
            continue
        if frm.lower() not in valid_names or to.lower() not in valid_names:
            continue
        if frm.lower() == to.lower():
            continue
        if rel_type not in _RELATION_TYPES:
            rel_type = "related_to"

        conf = item.get("confidence", 0.8)
        if not isinstance(conf, (int, float)) or not (0.0 <= conf <= 1.0):
            conf = 0.8

        triples.append(Triple(
            subject=frm,
            predicate=rel_type,
            object=to,
            confidence=float(conf),
            source_id=source_id,
        ))

    return triples
