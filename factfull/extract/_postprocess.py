"""factfull/extract/_postprocess.py — 話者名の正規化・タイポ補正"""
from __future__ import annotations

import difflib
import re

_DESC_PREFIX_RE = re.compile(r"^\[([^\]]+)\]")
_FUZZY_THRESHOLD = 0.80

_GENERIC_SPEAKER_NAMES = {
    "guest", "the guest", "a guest",
    "speaker", "the speaker", "a speaker",
    "host", "the host",
    "interviewee", "the interviewee",
    "narrator", "the narrator",
}


def make_speakers_block(speakers: list[str]) -> str:
    """正規スピーカー名をプロンプトに注入するブロックを生成する。"""
    if not speakers:
        return ""
    names = ", ".join(f'"{s}"' for s in speakers)
    forbidden = ', '.join(['"The Guest"', '"The Speaker"', '"Host"', '"Interviewee"', '"Speaker"'])
    return (
        f"## CANONICAL SPEAKER NAMES — use these EXACT strings, no variations\n"
        f"Known speakers: {names}\n"
        f"CRITICAL: Never use generic placeholders like {forbidden}.\n"
        f"If a statement is attributed to a speaker, use their EXACT canonical name above.\n"
        f"If unsure which speaker, use the first canonical name: \"{speakers[0]}\".\n\n"
    )


def closest_speaker(name: str, canonical: list[str]) -> str | None:
    """name に最も近い正規スピーカー名を返す。閾値未満なら None。"""
    best_ratio, best_name = 0.0, None
    for canon in canonical:
        ratio = difflib.SequenceMatcher(None, name.lower(), canon.lower()).ratio()
        if ratio > best_ratio:
            best_ratio, best_name = ratio, canon
    return best_name if best_ratio >= _FUZZY_THRESHOLD else None


def fix_speaker_prefix_typos(entities: list, canonical: list[str]) -> list:
    """entity.description の [Name] prefix をfuzzy補正する。"""
    for e in entities:
        if not e.description:
            continue
        m = _DESC_PREFIX_RE.match(e.description)
        if not m:
            continue
        prefix_name = m.group(1)
        if prefix_name in canonical:
            continue
        fixed = closest_speaker(prefix_name, canonical)
        if fixed:
            e.description = f"[{fixed}]" + e.description[m.end():]
    return entities


def fix_triple_speaker_typos(triples: list, canonical: list[str]) -> list:
    """triple の subject/object をfuzzy補正する。"""
    for t in triples:
        if t.subject not in canonical:
            fixed = closest_speaker(t.subject, canonical)
            if fixed:
                t.subject = fixed
        if t.object not in canonical:
            fixed = closest_speaker(t.object, canonical)
            if fixed:
                t.object = fixed
    return triples


def resolve_generic_speakers(entities: list, triples: list, canonical: list[str]) -> tuple[list, list]:
    """[The Guest] / [Speaker] 等のジェネリック名を正規名に置換する。"""
    if not canonical:
        return entities, triples

    guest_name = canonical[0]

    def _is_generic(name: str) -> bool:
        return name.strip().lower() in _GENERIC_SPEAKER_NAMES

    for e in entities:
        if _is_generic(e.name):
            e.name = guest_name
        if e.description:
            for placeholder in _GENERIC_SPEAKER_NAMES:
                pattern = re.compile(
                    r"^\[" + re.escape(placeholder) + r"\]",
                    re.IGNORECASE,
                )
                if pattern.match(e.description):
                    e.description = pattern.sub(f"[{guest_name}]", e.description, count=1)
                    break

    for t in triples:
        if _is_generic(t.subject):
            t.subject = guest_name
        if _is_generic(t.object):
            t.object = guest_name

    return entities, triples


def normalize_speakers(
    entities: list,
    triples: list,
    canonical: list[str],
) -> tuple[list, list]:
    """タイポ補正 → generic置換を順に適用する。"""
    if not canonical:
        return entities, triples
    entities = fix_speaker_prefix_typos(entities, canonical)
    triples = fix_triple_speaker_typos(triples, canonical)
    entities, triples = resolve_generic_speakers(entities, triples, canonical)
    return entities, triples
