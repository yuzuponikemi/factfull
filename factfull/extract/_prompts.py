"""factfull/extract/_prompts.py — KG抽出用プロンプトテンプレート"""

ENTITY_PROMPT = """\
You are building a knowledge graph from a structured podcast summary written in Japanese.
The summary contains clearly attributed claims, arguments, and evidence from named speakers.

Extract high-quality, specific entities. Focus on intellectual content.

## Entity types
- person      : Named speakers, researchers, or people discussed
- claim       : A specific argument, thesis, or position stated by a speaker
                Write as a short English declarative sentence capturing the core idea
- concept     : A key idea, theory, or analytical lens introduced
- framework   : A named mental model or structured analysis (e.g. "World Model", "Bitter Lesson")
- work        : A paper, book, product, or project mentioned by name
- organization: Company, lab, or institution
- event       : A milestone, release, or historical occurrence
- measurement : A specific metric, number, or benchmark

## Rules
- For [claim] and [concept]: ALWAYS set `speaker` to the person who stated/proposed it (not null)
- When `speaker` is set, start `description` with "[Speaker Name] " — e.g. "[Ilya Sutskever] argues that..."
- Write claim names as short declarative English sentences capturing the core idea
- Prefer named, specific entities over generic terms
- Extract the most intellectually significant entities — quality over quantity
- confidence: 0.9+ for central arguments, 0.7 for supporting details

{speakers_block}
## CRITICAL: Speaker attribution example
If Ilya Sutskever argues that "LLMs are not enough for AGI":
  "name": "LLMs alone cannot achieve AGI",
  "type": "claim",
  "description": "[Ilya Sutskever] argues that current LLMs lack the fundamental mechanisms needed for AGI.",
  "speaker": "Ilya Sutskever"

## Summary text (Japanese)
{text}

Output JSON only:
{{
  "entities": [
    {{
      "name": "concise English entity name",
      "type": "person|claim|concept|framework|work|organization|event|measurement",
      "description": "[Speaker Name] 1-2 sentences of context (start with [Speaker Name] if speaker is set)",
      "speaker": "Speaker Name if claim/concept is directly attributed, else null",
      "confidence": 0.90
    }}
  ]
}}
"""

RELATION_PROMPT = """\
You are building a knowledge graph from a structured podcast summary.
Given these entities and the summary text, extract meaningful relationships.
Choose the MOST SPECIFIC type — never use "related_to".

## Relation types
- argues_that  : [person] argues/believes [claim or concept]
- predicts     : [person] predicts [event or claim]
- criticizes   : [person/concept] challenges [person/concept]
- distinguishes: [concept] is explicitly contrasted with [concept]
- evidence_for : [fact/work/measurement] supports [claim]
- caused_by    : [outcome] is caused by [factor]
- builds_on    : [concept/work] extends [concept/work]
- created      : [person/org] created/developed [work/concept]
- works_at     : [person] is affiliated with [organization]
- is_a         : [X] is a type/instance of [Y]
- part_of      : [X] is part of [Y]
- enables      : [X] makes [Y] possible

## Rules
- Only extract relationships clearly present in the text
- from/to must exactly match entity names from the list
- No "related_to" under any circumstances

## Known entities
{entities}

## Summary text (Japanese)
{text}

Output JSON only:
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
