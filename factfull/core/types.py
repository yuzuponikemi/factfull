"""
factfull/core/types.py
=======================
パイプライン全体で共有するデータ型。

すべての Ingestion モジュールは SourceDoc を返し、
Processing モジュールは ProcessedDoc を返す。
これにより L1〜L4 が疎結合になる。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# ── L1: Ingestion 出力 ────────────────────────────────────────────────────────

SOURCE_TYPES = {"podcast", "paper", "book", "web"}

@dataclass
class SourceDoc:
    """あらゆる取り込みソースの共通表現。"""
    source_type: str          # "podcast" | "paper" | "book" | "web"
    source_id: str            # video_id / DOI / ISBN / URL など
    title: str
    text: str                 # 原文テキスト（英語または原語）
    text_ja: str = ""         # 日本語テキスト（翻訳済み、なければ空）
    chunks: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_type": self.source_type,
            "source_id": self.source_id,
            "title": self.title,
            "text": self.text,
            "text_ja": self.text_ja,
            "chunks": self.chunks,
            "metadata": self.metadata,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SourceDoc:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ── L2: Processing 出力 ───────────────────────────────────────────────────────

ENTITY_TYPES = {
    # 論文・研究ドメイン（kg-builder 互換）
    "method", "material", "phenomenon", "theory", "measurement", "application",
    # 汎用ドメイン（podcast / book / web）
    "person", "organization", "place", "event", "concept", "product", "work",
}

@dataclass
class Entity:
    """テキストから抽出された固有名詞・概念。"""
    name: str
    type: str                 # ENTITY_TYPES のいずれか
    description: str = ""
    confidence: float = 1.0
    source_id: str = ""       # どの SourceDoc から来たか

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "type": self.type,
            "description": self.description,
            "confidence": self.confidence,
            "source_id": self.source_id,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Entity:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class Triple:
    """知識グラフの1エッジ: (subject, predicate, object)。"""
    subject: str
    predicate: str
    object: str
    confidence: float = 1.0
    source_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.object,
            "confidence": self.confidence,
            "source_id": self.source_id,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Triple:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class ProcessedDoc:
    """Processing 層の出力。SourceDoc + 生成された知識。"""
    source: SourceDoc
    summary: str = ""         # 日本語要約記事（Markdown）
    triples: list[Triple] = field(default_factory=list)
    entities: list[Entity] = field(default_factory=list)
    score: float = 0.0        # ファクトチェックスコア (0–100)

    # 記事ファイルパス（ディスクに保存されている場合）
    summary_path: Path | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source.to_dict(),
            "summary": self.summary,
            "triples": [t.to_dict() for t in self.triples],
            "entities": [e.to_dict() for e in self.entities],
            "score": self.score,
            "summary_path": str(self.summary_path) if self.summary_path else None,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ProcessedDoc:
        source = SourceDoc.from_dict(d["source"])
        triples = [Triple.from_dict(t) for t in d.get("triples", [])]
        entities = [Entity.from_dict(e) for e in d.get("entities", [])]
        sp = d.get("summary_path")
        return cls(
            source=source,
            summary=d.get("summary", ""),
            triples=triples,
            entities=entities,
            score=d.get("score", 0.0),
            summary_path=Path(sp) if sp else None,
        )
