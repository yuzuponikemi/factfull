"""
factfull/extract/entity.py
===========================
テキストからエンティティを抽出する。

移植元: kg-builder/src/kg_builder/extractor/entity_extractor.py
変更点:
  - kg-builder の LLM クライアントの代わりに factfull.llm を使用
  - 論文特化型から汎用型に拡張（person / organization / concept 等を追加）
  - 返り値が factfull.core.types.Entity のリスト

使い方:
    from factfull.extract.entity import extract_entities
    from factfull.core.types import SourceDoc

    entities = extract_entities(doc.chunks[:5], source_id=doc.source_id)
"""
from __future__ import annotations

import json
import re

from factfull.core.types import Entity
from factfull import llm

_PROMPT = """\
あなたは知識抽出の専門家です。以下のテキストを読み、重要な固有名詞・概念を抽出してください。

## 抽出するエンティティ種別

### 汎用（Podcast・書籍・Web 記事向け）
- person: 人物（研究者・起業家・著者など）
- organization: 組織・企業・機関・大学
- place: 地名・施設・国
- event: イベント・会議・出来事
- concept: 抽象的概念・思想・理論（汎用）
- product: 製品・サービス・ソフトウェア
- work: 論文・書籍・映画などの作品

### 科学・技術（論文向け）
- method: 手法・アルゴリズム・アプローチ
- material: 材料・物質・化合物
- phenomenon: 現象・効果・プロセス
- theory: 理論的枠組み・モデル・原理
- measurement: 指標・特性・評価項目
- application: ユースケース・応用分野

## ガイドライン
- 文章に明示的に登場するエンティティのみ抽出する
- 固有名詞・専門用語を優先する（"data", "research" などの一般語は除外）
- 同じ概念の表記ゆれは統一する（例: "neural network" と "neural networks" → "neural networks"）
- confidence: テキスト内での重要度・明確さを 0.0〜1.0 で評価

## 対象テキスト
{text}

## 出力形式（JSON のみ。前置き不要）
{{
  "entities": [
    {{
      "name": "エンティティ名",
      "type": "person|organization|concept|method|...",
      "description": "1〜2文の説明",
      "confidence": 0.95
    }}
  ]
}}
"""


def extract_entities(
    chunks: list[str],
    source_id: str = "",
    model: str | None = None,
    max_per_chunk: int = 15,
) -> list[Entity]:
    """テキストチャンクのリストからエンティティを抽出する。

    Args:
        chunks: テキストチャンクのリスト
        source_id: どの SourceDoc から来たか（Entity.source_id に格納）
        model: Ollama モデル名（省略時は環境変数）
        max_per_chunk: チャンクごとの最大抽出数

    Returns:
        Entity のリスト（name をキーに重複除去済み、confidence 最大値を採用）
    """
    seen: dict[str, Entity] = {}  # name.lower() → Entity

    for i, chunk in enumerate(chunks, 1):
        print(f"  [entity] チャンク {i}/{len(chunks)} 処理中...", flush=True)
        prompt = _PROMPT.format(text=chunk[:6000])
        raw = llm.call(prompt, num_ctx=8192, model=model)
        entities = _parse(raw, source_id)

        for e in entities[:max_per_chunk]:
            key = e.name.lower()
            if key not in seen or e.confidence > seen[key].confidence:
                seen[key] = e

    result = list(seen.values())
    print(f"  [entity] 抽出完了: {len(result)} エンティティ", flush=True)
    return result


def _parse(raw: str, source_id: str) -> list[Entity]:
    """LLM 出力から Entity リストを取り出す。"""
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        return []
    try:
        data = json.loads(m.group())
    except json.JSONDecodeError:
        return []

    valid_types = {
        "person", "organization", "place", "event", "concept", "product", "work",
        "method", "material", "phenomenon", "theory", "measurement", "application",
    }
    entities: list[Entity] = []
    for item in data.get("entities", []):
        name = (item.get("name") or "").strip()
        etype = (item.get("type") or "concept").strip().lower()
        if not name:
            continue
        if etype not in valid_types:
            etype = "concept"
        conf = item.get("confidence", 0.8)
        if not isinstance(conf, (int, float)) or not (0.0 <= conf <= 1.0):
            conf = 0.8
        entities.append(Entity(
            name=name,
            type=etype,
            description=item.get("description", ""),
            confidence=float(conf),
            source_id=source_id,
        ))
    return entities
