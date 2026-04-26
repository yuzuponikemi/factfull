"""
factfull/extract/podcast_extract.py
=====================================
話者分離済みポッドキャスト向けの高品質エンティティ・関係抽出。

通常の entity.py / relation.py との違い:
  - [Speaker] プレフィックスを活かして発言を話者に帰属させる
  - "claim"（主張）エンティティを追加 — 具体的な論点を構造化して保存
  - 関係タイプが意味的に豊富（argues_that / predicts / criticizes 等）
  - RELATED_TO をフォールバックとして使わない
  - 話者ターン境界でチャンクを分割して文脈を保持

使い方:
    from factfull.extract.podcast_extract import extract_podcast

    entities, triples = extract_podcast(diarized_text, source_id="n1E9IZfvGMA")
"""
from __future__ import annotations

import json
import re
from typing import Any

from factfull.core.types import Entity, Triple
from factfull import llm
from factfull.extract._prompts import ENTITY_PROMPT as _SUMMARY_ENTITY_PROMPT
from factfull.extract._prompts import RELATION_PROMPT as _SUMMARY_RELATION_PROMPT
from factfull.extract._postprocess import (
    make_speakers_block as _make_speakers_block,
    normalize_speakers as _normalize_speakers,
    fix_speaker_prefix_typos as _fix_speaker_prefix_typos,
    fix_triple_speaker_typos as _fix_triple_speaker_typos,
)

# ── エンティティ抽出プロンプト ─────────────────────────────────────────────

_ENTITY_PROMPT = """\
You are analyzing a podcast transcript. Speaker turns are marked as [Speaker Name].

Extract high-quality, specific entities. Focus on intellectual content, not generic terms.

## Entity types
- person      : Host, guest, or anyone discussed by name
- claim       : A specific argument, prediction, or position a speaker states
                (capture the core idea concisely, e.g. "AI scaling will plateau by 2027")
- concept     : A key idea, theory, or framework — especially if defined or introduced
- method      : A technique, algorithm, or approach
- work        : A paper, book, essay, or product mentioned by name
- organization: Company, lab, university, government body
- event       : A historical event, milestone, or release
- framework   : A named mental model or analytical lens (e.g. "Big Blob of Compute Hypothesis")

## Rules
- For [claim]: write as a short declarative sentence capturing the actual argument
- Skip generic words: "research", "data", "model", "system", "approach" alone
- Prefer specific named entities over vague references
- If the speaker explicitly introduces or names something, note it
- confidence: 0.9+ for central topics, 0.7 for briefly mentioned items

## Transcript excerpt
{text}

Output JSON only (no markdown):
{{
  "entities": [
    {{
      "name": "concise entity name",
      "type": "person|claim|concept|method|work|organization|event|framework",
      "description": "1-2 sentences of context from the transcript",
      "speaker": "Speaker Name if directly attributed, else null",
      "confidence": 0.90
    }}
  ]
}}
"""

# ── 関係抽出プロンプト ────────────────────────────────────────────────────

_RELATION_PROMPT = """\
You are building a knowledge graph from a podcast transcript.
Speaker turns are marked as [Speaker Name].

Given the entities and transcript, extract meaningful relationships.
Choose the MOST SPECIFIC type — never use "related_to".

## Relation types
- argues_that    : [person] argues/believes/claims [claim or concept]
- predicts       : [person] predicts [event or claim]
- criticizes     : [person/concept] challenges or criticizes [person/concept]
- distinguishes  : [concept] is explicitly contrasted with [concept]
- evidence_for   : [fact/concept] supports or is evidence for [claim]
- caused_by      : [outcome] is caused by [factor]
- builds_on      : [concept/work] extends or builds on [concept/work]
- created        : [person/org] created/wrote/developed [work/concept/product]
- works_at       : [person] works at or is affiliated with [organization]
- agrees_with    : [person] explicitly agrees with [person/claim]
- responds_to    : [claim/statement] directly responds to [claim/question]
- is_a           : [X] is a type or instance of [Y]
- part_of        : [X] is part of [Y]
- uses           : [X] uses or employs [Y]
- enables        : [X] makes [Y] possible

## Rules
- Only extract relationships clearly present in the text
- from/to must exactly match entity names from the list
- No self-loops (from == to)
- Do NOT use "related_to" under any circumstances

## Known entities
{entities}

## Transcript excerpt
{text}

Output JSON only (no markdown):
{{
  "relationships": [
    {{
      "from": "entity name",
      "to": "entity name",
      "type": "relation_type",
      "confidence": 0.85
    }}
  ]
}}
"""

# ── 話者ターン境界でチャンク分割 ──────────────────────────────────────────

def _split_by_speaker_turns(text: str, max_chars: int = 4000) -> list[str]:
    """[Speaker] 境界を尊重しながらテキストをチャンクに分割する。

    同一話者の連続発言はまとめ、max_chars を超えたら次のチャンクに移る。
    """
    # [Speaker Name] で始まる行を検出
    turn_pattern = re.compile(r"^\[([^\]]+)\]", re.MULTILINE)
    splits = list(turn_pattern.finditer(text))

    if not splits:
        # 話者ラベルなし → 文字数で単純分割
        return [text[i:i + max_chars] for i in range(0, len(text), max_chars)]

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for idx, m in enumerate(splits):
        start = m.start()
        end = splits[idx + 1].start() if idx + 1 < len(splits) else len(text)
        segment = text[start:end]

        if current_len + len(segment) > max_chars and current:
            chunks.append("".join(current))
            current = [segment]
            current_len = len(segment)
        else:
            current.append(segment)
            current_len += len(segment)

    if current:
        chunks.append("".join(current))

    return chunks


# ── エンティティ抽出 ──────────────────────────────────────────────────────

_VALID_TYPES = {
    "person", "claim", "concept", "method", "work",
    "organization", "event", "framework",
    # 後方互換
    "product", "place", "theory", "measurement", "application",
    "material", "phenomenon",
}


def _parse_entities(raw: str, source_id: str) -> list[Entity]:
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        return []
    try:
        data = json.loads(m.group())
    except json.JSONDecodeError:
        return []

    result: list[Entity] = []
    for item in data.get("entities", []):
        name = (item.get("name") or "").strip()
        if not name:
            continue
        etype = (item.get("type") or "concept").strip().lower()
        if etype not in _VALID_TYPES:
            etype = "concept"
        conf = item.get("confidence", 0.8)
        if not isinstance(conf, (int, float)) or not (0.0 <= conf <= 1.0):
            conf = 0.8
        desc = item.get("description", "")
        speaker = item.get("speaker") or ""
        # speaker情報をdescriptionに付加して保持
        if speaker:
            desc = f"[{speaker}] {desc}"
        result.append(Entity(
            name=name,
            type=etype,
            description=desc,
            confidence=float(conf),
            source_id=source_id,
        ))
    return result


def extract_entities_podcast(
    chunks: list[str],
    source_id: str = "",
    model: str | None = None,
    max_per_chunk: int = 20,
) -> list[Entity]:
    seen: dict[str, Entity] = {}

    for i, chunk in enumerate(chunks, 1):
        print(f"  [entity] チャンク {i}/{len(chunks)} 処理中...", flush=True)
        raw = llm.call(
            _ENTITY_PROMPT.format(text=chunk[:5000]),
            num_ctx=8192,
            model=model,
        )
        for e in _parse_entities(raw, source_id)[:max_per_chunk]:
            key = e.name.lower()
            if key not in seen or e.confidence > seen[key].confidence:
                seen[key] = e

    result = list(seen.values())
    print(f"  [entity] 抽出完了: {len(result)} エンティティ", flush=True)
    return result


# ── 関係抽出 ─────────────────────────────────────────────────────────────

_VALID_RELATIONS = {
    "argues_that", "predicts", "criticizes", "distinguishes",
    "evidence_for", "caused_by", "builds_on", "created",
    "works_at", "agrees_with", "responds_to",
    "is_a", "part_of", "uses", "enables",
    # 後方互換（entity.py / relation.py との互換）
    "says", "mentions", "about", "based_on", "applies_to",
    "measures", "located_in",
}


def _parse_triples(raw: str, valid_names: set[str], source_id: str) -> list[Triple]:
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        return []
    try:
        data = json.loads(m.group())
    except json.JSONDecodeError:
        return []

    result: list[Triple] = []
    for item in data.get("relationships", []):
        frm = (item.get("from") or "").strip()
        to = (item.get("to") or "").strip()
        rel = (item.get("type") or "").strip().lower().replace(" ", "_").replace("-", "_")
        if not frm or not to:
            continue
        if frm.lower() not in valid_names or to.lower() not in valid_names:
            continue
        if frm.lower() == to.lower():
            continue
        if rel == "related_to":
            continue  # 汎用すぎる関係は捨てる
        if rel not in _VALID_RELATIONS:
            rel = "is_a"  # 未知の述語はis_aに丸める（捨てるより情報を残す）
        conf = item.get("confidence", 0.8)
        if not isinstance(conf, (int, float)) or not (0.0 <= conf <= 1.0):
            conf = 0.8
        result.append(Triple(
            subject=frm,
            predicate=rel,
            object=to,
            confidence=float(conf),
            source_id=source_id,
        ))
    return result


def extract_relations_podcast(
    chunks: list[str],
    entities: list[Entity],
    source_id: str = "",
    model: str | None = None,
) -> list[Triple]:
    if not entities:
        return []

    entity_list = "\n".join(f"- {e.name} ({e.type})" for e in entities)
    valid_names = {e.name.lower() for e in entities}
    seen: dict[str, Triple] = {}

    for i, chunk in enumerate(chunks, 1):
        print(f"  [relation] チャンク {i}/{len(chunks)} 処理中...", flush=True)
        raw = llm.call(
            _RELATION_PROMPT.format(entities=entity_list, text=chunk[:5000]),
            num_ctx=8192,
            model=model,
        )
        for t in _parse_triples(raw, valid_names, source_id):
            key = f"{t.subject.lower()}|{t.predicate}|{t.object.lower()}"
            if key not in seen or t.confidence > seen[key].confidence:
                seen[key] = t

    result = list(seen.values())
    print(f"  [relation] 抽出完了: {len(result)} トリプル", flush=True)
    return result


# ── トランスクリプトから話者名を強制登録 ────────────────────────────────

def _extract_speaker_labels(diarized_text: str) -> list[str]:
    """[Speaker Name] ラベルをテキストから全件抽出してユニーク化する。"""
    return list(dict.fromkeys(
        m.group(1).strip()
        for m in re.finditer(r"^\[([^\]]+)\]", diarized_text, re.MULTILINE)
    ))


def _ensure_speaker_entities(
    speaker_names: list[str],
    entities: list[Entity],
    source_id: str,
) -> list[Entity]:
    """話者名を person エンティティとして保証する。

    LLMが名前を省略形で抽出した場合（"Dario" vs "Dario Amodei"）でも
    フルネームを正規エンティティとして登録する。
    """
    existing_lower = {e.name.lower() for e in entities}
    new_entities = list(entities)
    for name in speaker_names:
        if name.lower() not in existing_lower:
            new_entities.append(Entity(
                name=name,
                type="person",
                description=f"Podcast speaker: {name}",
                confidence=1.0,
                source_id=source_id,
            ))
            print(f"  [podcast_extract] 話者エンティティ追加: {name}", flush=True)
    return new_entities


# ── 話者帰属トリプルの自動生成 ───────────────────────────────────────────

def _infer_speaker_triples(
    entities: list[Entity],
    speaker_names: list[str],
    source_id: str,
) -> list[Triple]:
    """エンティティの description に埋め込まれた [Speaker] prefix から
    person --ARGUES_THAT--> claim/concept/framework トリプルを自動生成する。

    speaker_names: トランスクリプトから直接抽出したフルネームリスト。
    これを照合に使うことで "Dario" vs "Dario Amodei" の不一致を回避する。
    """
    # フルネームで照合テーブルを作成（部分一致対応）
    name_lookup: list[str] = speaker_names  # 登録順で優先度あり
    triples: list[Triple] = []

    for e in entities:
        if e.type not in ("claim", "framework", "concept"):
            continue
        m = re.match(r"^\[([^\]]+)\]", e.description or "")
        if not m:
            continue
        speaker_raw = m.group(1).strip()

        # トランスクリプトのフルネームと照合（完全一致 → 部分一致の順）
        matched_name = speaker_raw  # デフォルトはそのまま使う
        for full_name in name_lookup:
            if speaker_raw.lower() == full_name.lower():
                matched_name = full_name
                break
            if speaker_raw.lower() in full_name.lower() or full_name.lower() in speaker_raw.lower():
                matched_name = full_name
                break

        triples.append(Triple(
            subject=matched_name,
            predicate="argues_that",
            object=e.name,
            confidence=0.85,
            source_id=source_id,
        ))

    print(f"  [podcast_extract] 話者帰属トリプル: {len(triples)} 件", flush=True)
    return triples


# ── ワンショット API ──────────────────────────────────────────────────────

def extract_podcast(
    diarized_text: str,
    source_id: str = "",
    model: str | None = None,
    max_chunks: int = 10,
) -> tuple[list[Entity], list[Triple]]:
    """話者分離済みテキストからエンティティとトリプルを抽出する。

    Args:
        diarized_text: [Speaker] prefix 付きのトランスクリプト
        source_id: SourceDoc の source_id
        model: Ollama モデル名
        max_chunks: 処理するチャンク上限

    Returns:
        (entities, triples)
    """
    # トランスクリプトから話者フルネームを確定
    speaker_names = _extract_speaker_labels(diarized_text)
    print(f"  [podcast_extract] 話者: {speaker_names}", flush=True)

    chunks = _split_by_speaker_turns(diarized_text, max_chars=4000)[:max_chunks]
    print(f"  [podcast_extract] {len(chunks)} チャンクで処理", flush=True)

    # エンティティ抽出 → 話者をフルネームで強制登録
    entities = extract_entities_podcast(chunks, source_id=source_id, model=model)
    entities = _ensure_speaker_entities(speaker_names, entities, source_id)

    # LLM による関係抽出
    triples = extract_relations_podcast(chunks, entities, source_id=source_id, model=model)

    # 話者帰属トリプルをフルネームで自動補完
    speaker_triples = _infer_speaker_triples(entities, speaker_names, source_id=source_id)
    existing_keys = {f"{t.subject.lower()}|{t.predicate}|{t.object.lower()}" for t in triples}
    for t in speaker_triples:
        key = f"{t.subject.lower()}|{t.predicate}|{t.object.lower()}"
        if key not in existing_keys:
            triples.append(t)

    return entities, triples


# ── サマリーからの高品質抽出 ─────────────────────────────────────────────
# プロンプトは factfull/extract/_prompts.py
# 後処理は factfull/extract/_postprocess.py


def extract_from_summary(
    summary_text: str,
    source_id: str = "",
    model: str | None = None,
    chunk_size: int = 6000,
    canonical_speakers: list[str] | None = None,
) -> tuple[list[Entity], list[Triple]]:
    """高品質な日本語サマリーからエンティティとトリプルを抽出する。

    diarized transcript より遥かに密度が高く、論点が構造化されているため
    少ないチャンク数で高品質な抽出が得られる。

    Args:
        summary_text: summary_ja.md の内容
        source_id: SourceDoc の source_id
        model: Ollama モデル名（大きいモデル推奨）
        chunk_size: チャンク文字数（サマリーは密度が高いので大きめに）

    Returns:
        (entities, triples)
    """
    # サマリーをチャンク分割（見出し境界で分割）
    chunks = _split_summary(summary_text, chunk_size=chunk_size)
    print(f"  [summary_extract] {len(chunks)} チャンクで処理 (model={model})", flush=True)

    # 正規スピーカー名: 引数 > サマリーから自動抽出
    if canonical_speakers is None:
        canonical_speakers = _extract_speakers_from_summary(summary_text)
    speakers_block = _make_speakers_block(canonical_speakers)

    # エンティティ抽出
    seen_entities: dict[str, Entity] = {}
    for i, chunk in enumerate(chunks, 1):
        print(f"  [entity] チャンク {i}/{len(chunks)} 処理中...", flush=True)
        raw = llm.call(
            _SUMMARY_ENTITY_PROMPT.format(text=chunk[:6000], speakers_block=speakers_block),
            num_ctx=10000,
            timeout=3600,
            model=model,
        )
        for e in _parse_entities(raw, source_id):
            key = e.name.lower()
            if key not in seen_entities or e.confidence > seen_entities[key].confidence:
                seen_entities[key] = e

    entities = list(seen_entities.values())
    print(f"  [entity] 抽出完了: {len(entities)} エンティティ", flush=True)

    # 話者をフルネームで確保（canonical_speakers を再利用）
    speaker_names = canonical_speakers
    entities = _ensure_speaker_entities(speaker_names, entities, source_id)

    # 関係抽出
    entity_list = "\n".join(f"- {e.name} ({e.type})" for e in entities)
    valid_names = {e.name.lower() for e in entities}
    seen_triples: dict[str, Triple] = {}

    for i, chunk in enumerate(chunks, 1):
        print(f"  [relation] チャンク {i}/{len(chunks)} 処理中...", flush=True)
        raw = llm.call(
            _SUMMARY_RELATION_PROMPT.format(entities=entity_list, text=chunk[:6000]),
            num_ctx=10000,
            timeout=3600,
            model=model,
        )
        for t in _parse_triples(raw, valid_names, source_id):
            key = f"{t.subject.lower()}|{t.predicate}|{t.object.lower()}"
            if key not in seen_triples or t.confidence > seen_triples[key].confidence:
                seen_triples[key] = t

    triples = list(seen_triples.values())

    # 話者帰属トリプルを自動補完
    speaker_triples = _infer_speaker_triples(entities, speaker_names, source_id=source_id)
    existing_keys = {f"{t.subject.lower()}|{t.predicate}|{t.object.lower()}" for t in triples}
    for t in speaker_triples:
        key = f"{t.subject.lower()}|{t.predicate}|{t.object.lower()}"
        if key not in existing_keys:
            triples.append(t)

    # タイポ補正 → generic置換（_postprocess.py）
    entities, triples = _normalize_speakers(entities, triples, speaker_names)

    print(f"  [relation] 抽出完了: {len(triples)} トリプル", flush=True)
    return entities, triples


def _split_summary(text: str, chunk_size: int = 6000) -> list[str]:
    """サマリーを見出し（####）境界で分割する。"""
    parts = re.split(r"(?=^####)", text, flags=re.MULTILINE)
    chunks: list[str] = []
    current = ""
    for part in parts:
        if len(current) + len(part) > chunk_size and current:
            chunks.append(current)
            current = part
        else:
            current += part
    if current:
        chunks.append(current)
    return chunks or [text]


def _extract_speakers_from_summary(text: str) -> list[str]:
    """サマリー中の「―― **Name**」形式から話者名を抽出する。
    日本語文字を含む役職説明（例: Anthropic元メンバー）は除外する。
    """
    _JP = re.compile(r"[\u3040-\u309f\u30a0-\u30ff\u4e00-\u9fff]")
    names = []
    for m in re.finditer(r"――\s+\*\*([^*]+)\*\*", text):
        name = m.group(1).strip()
        if _JP.search(name):
            continue
        if name not in names:
            names.append(name)
    return names
