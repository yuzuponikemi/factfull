"""
factfull/bilingual/types.py
============================
論文 英日対訳ドキュメントの共通データ型。

設計方針（factfull/core/types.py に準拠）:
  - plain dataclass（pydantic 不使用）
  - 各クラスに to_dict / from_dict（from_dict は未知キーを無視＝前方互換）
  - JSON 出力は json.dump(..., ensure_ascii=False, indent=2)

BilingualDoc は「入れ子なしの Block を読み順で並べた」フラット構造。
homupe 側は blocks を反復し、type / level だけでレイアウト・フォント比率・
図表配置を自由に決められる（Scholaread 風の英日対訳レイアウト）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

SCHEMA_VERSION = "1.0"

# ブロック種別
#   title / heading / abstract / paragraph : 翻訳対象テキスト
#   caption                                : 図表キャプション（翻訳対象）
#   reference                              : 参考文献（既定では未翻訳 or 除外）
#   figure / table                         : 図・表（画像として抽出、原文位置を保持）
BLOCK_TYPES = {
    "title", "heading", "abstract", "paragraph",
    "caption", "reference", "figure", "table",
}


@dataclass
class Block:
    """対訳ドキュメントの 1 ブロック（読み順で並ぶ独立レコード）。"""
    id: str                                   # "b0001"… 読み順のゼロ詰め連番（安定 ID）
    type: str                                 # BLOCK_TYPES のいずれか
    en: str = ""                              # 英語原文（figure/table は空）
    ja: str = ""                              # 日本語訳（未訳・非対象は空）
    level: int | None = None                  # 見出し深さ 1..N（見出し以外は None）
    section_path: list[str] = field(default_factory=list)  # 祖先見出し（EN）
    page: int | None = None                   # 出典ページ（1 始まり）
    bbox: list[float] | None = None           # 原文での矩形 [x0, y0, x1, y1]（図表位置）
    image_path: str = ""                      # figure/table の抽出画像（out_dir からの相対パス）
    label: str = ""                           # "Figure 1" / "Table 2" 等の図表ラベル
    skip_translate: bool = False              # 翻訳対象外で原文のまま載せる場合 True

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "en": self.en,
            "ja": self.ja,
            "level": self.level,
            "section_path": list(self.section_path),
            "page": self.page,
            "bbox": list(self.bbox) if self.bbox is not None else None,
            "image_path": self.image_path,
            "label": self.label,
            "skip_translate": self.skip_translate,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Block":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class BilingualDoc:
    """論文 1 本の英日対訳ドキュメント。JSON のルート。"""
    title_en: str
    title_ja: str
    authors: list[str]
    source_id: str
    arxiv_url: str
    source_type: str                          # "arxiv" | "pdf"
    model: str                                # 翻訳に使ったモデル
    translated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    schema_version: str = SCHEMA_VERSION
    abstract_en: str = ""
    abstract_ja: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)  # num_pages / published / categories…
    blocks: list[Block] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source_type": self.source_type,
            "source_id": self.source_id,
            "arxiv_url": self.arxiv_url,
            "title_en": self.title_en,
            "title_ja": self.title_ja,
            "authors": list(self.authors),
            "abstract_en": self.abstract_en,
            "abstract_ja": self.abstract_ja,
            "model": self.model,
            "translated_at": self.translated_at,
            "metadata": self.metadata,
            "blocks": [b.to_dict() for b in self.blocks],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "BilingualDoc":
        blocks = [Block.from_dict(b) for b in d.get("blocks", [])]
        scalar = {
            k: v for k, v in d.items()
            if k in cls.__dataclass_fields__ and k != "blocks"
        }
        return cls(blocks=blocks, **scalar)
