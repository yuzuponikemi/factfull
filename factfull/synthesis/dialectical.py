"""
factfull/synthesis/dialectical.py
====================================
弁証法的仮説検証サイクルによる統合論考生成。

ループ構造:
  1. 定立 (Thesis)     : 選択ソースの概念・主張から初期仮説を生成
  2. 反証収集          : 他ソースの共通エンティティから対抗エビデンスを収集
  3. 検証 (Antithesis) : 仮説 vs 反証をLLMが評価
  4. 止揚 (Aufhebung)  : 仮説を精緻化・再定式化
       └── 2→3→4 を n ループ
  5. 統合論考          : ループ履歴全体から ~10,000字の論考を生成
"""
from __future__ import annotations

import dataclasses
from typing import Any

from factfull.graph.neo4j import Neo4jClient
from factfull import llm


# ── Cypher クエリ ──────────────────────────────────────────────────────────────

_Q_SOURCE_CONCEPTS = """
MATCH (s:Source)-[:MENTIONS]->(e:Entity)
WHERE s.source_id IN $source_ids
  AND e.type IN ['concept', 'framework', 'theory', 'method', 'phenomenon', 'claim']
  AND e.description IS NOT NULL
RETURN s.source_id, s.title, e.name, e.type, e.description, e.confidence
ORDER BY e.confidence DESC
LIMIT $limit
"""

_Q_SOURCE_PERSONS = """
MATCH (s:Source)-[:MENTIONS]->(p:Entity {type: 'person'})
WHERE s.source_id IN $source_ids
RETURN DISTINCT p.name AS name, s.title AS source_title
LIMIT 20
"""

_Q_ANTITHESIS_OTHER_SOURCES = """
MATCH (s:Source)-[:MENTIONS]->(e:Entity)
WHERE NOT s.source_id IN $primary_ids
  AND e.type IN ['concept', 'framework', 'theory', 'method', 'claim', 'phenomenon']
  AND e.description IS NOT NULL
  AND e.confidence >= 0.5
WITH s, e
ORDER BY e.confidence DESC
RETURN s.source_id, s.title AS source_title,
       e.name AS name, e.type AS type, e.description AS description
LIMIT $limit
"""


# ── データクラス ───────────────────────────────────────────────────────────────

@dataclasses.dataclass
class DialecticalStep:
    loop: int
    thesis: str
    antithesis_entities: list[dict[str, Any]]
    evaluation: str
    refined_thesis: str


# ── 内部ヘルパー ───────────────────────────────────────────────────────────────

def _fetch_primary_material(
    client: Neo4jClient,
    source_ids: list[str],
    limit: int = 80,
) -> tuple[list[dict], list[dict]]:
    """主要ソースの概念群と関与人物を取得する。"""
    concepts = client.run_cypher(_Q_SOURCE_CONCEPTS, {"source_ids": source_ids, "limit": limit})
    persons = client.run_cypher(_Q_SOURCE_PERSONS, {"source_ids": source_ids})
    return concepts, persons


def _fetch_antithesis_material(
    client: Neo4jClient,
    source_ids: list[str],
    primary_concepts: list[dict] | None = None,
    limit: int = 60,
) -> list[dict]:
    """他ソースの概念群から対抗エビデンス候補を取得する。

    キーワード重複やエンティティ名共有ではなく、他ソース全体からサンプリングし、
    LLM が弁証法的緊張を発見する設計。
    """
    rows = client.run_cypher(
        _Q_ANTITHESIS_OTHER_SOURCES,
        {"primary_ids": source_ids, "limit": limit},
    )

    # 主要概念のキーワードと重複するものを優先的に選別（Python側フィルタ）
    if primary_concepts and rows:
        primary_terms: set[str] = set()
        for c in primary_concepts[:30]:
            name = (c.get("e.name") or c.get("name") or "").lower()
            for word in name.split():
                if len(word) > 4:
                    primary_terms.add(word)

        scored: list[tuple[int, dict]] = []
        for r in rows:
            r_name = (r.get("name") or "").lower()
            r_desc = (r.get("description") or "").lower()
            score = sum(1 for t in primary_terms if t in r_name or t in r_desc)
            scored.append((score, r))
        scored.sort(key=lambda x: -x[0])
        rows = [r for _, r in scored]

    # 名前でデデュプ（最初の出現を残す）
    seen: set[str] = set()
    deduped: list[dict] = []
    for r in rows:
        key = (r.get("name") or "").lower()
        if key not in seen:
            seen.add(key)
            r["other_perspectives"] = []
            deduped.append(r)
    return deduped[:30]


def _format_concepts(concepts: list[dict], persons: list[dict]) -> str:
    lines: list[str] = []
    if persons:
        names = ", ".join(p["name"] for p in persons[:8])
        lines.append(f"**関与する思想家・論者:** {names}\n")
    lines.append("**主要概念・主張:**")
    for c in concepts[:50]:
        src = (c.get("s.source_id") or c.get("source_id") or "")[:30]
        name = c.get("e.name") or c.get("name") or ""
        etype = c.get("e.type") or c.get("type") or ""
        desc = c.get("e.description") or c.get("description") or ""
        lines.append(f"- [{etype}] **{name}** ({src})")
        if desc:
            lines.append(f"  {desc[:150]}")
    return "\n".join(lines)


def _format_antithesis(shared: list[dict]) -> str:
    if not shared:
        return "（対抗エビデンスなし）"
    lines = ["**他ソースから見た共通エンティティ（対抗・補完視点）:**"]
    for r in shared[:20]:
        name = r.get("name") or ""
        etype = r.get("type") or ""
        desc = r.get("description") or ""
        lines.append(f"\n- **{name}** [{etype}]")
        lines.append(f"  主要ソースでの記述: {desc[:120]}")
        for op in r.get("other_perspectives", [])[:2]:
            lines.append(f"  他ソース視点: {op}")
    return "\n".join(lines)


# ── LLM プロンプト ─────────────────────────────────────────────────────────────

_THESIS_PROMPT = """\
あなたは哲学・思想の統合研究者です。以下の素材から、**初期仮説（定立）**を生成してください。

## テーマ
{topic}

## 参照ソース
{source_list}

## 概念・主張の素材
{material}

---

**指示:**
以下の形式で初期仮説を日本語で生成してください。500〜700字で。

### 初期仮説（定立）

（現代における問題意識から始め、上記概念・主張を統合した1つの鮮明な主張を述べる。
 「〜である」調。反証の余地を残す断定的な形で書く。末尾に「この仮説の弱点は〜である」と1文加える。）
"""

_ANTITHESIS_PROMPT = """\
あなたは批判的分析者です。以下の**仮説**に対し、提示された**対抗エビデンス**を用いて評価してください。

## 現在の仮説
{thesis}

## 対抗エビデンス（他ソースからの反証候補）
{antithesis_material}

---

**指示:**
以下の形式で評価を日本語で書いてください。400〜600字。

### 検証評価（第{loop}ループ）

**支持される点:** （仮説のうち対抗エビデンスでも裏付けられる核心部分）

**反証される・修正が必要な点:** （対抗エビデンスが仮説の何を覆すか、具体的に）

**見落とされていた視点:** （対抗エビデンスが加える新しい次元）
"""

_REFINEMENT_PROMPT = """\
あなたは弁証法的思考を行う哲学者です。以下の定立・評価を踏まえ、**より精緻な仮説（止揚）**を生成してください。

## 元の仮説
{thesis}

## 検証評価
{evaluation}

---

**指示:**
止揚（Aufhebung）として、元の仮説を解体・統合・発展させた精緻化仮説を500〜700字で書いてください。
単なる折衷ではなく、矛盾を内包しつつより高次の視座から統合すること。「〜である」調。
"""

_COMMON_HEADER = """\
## 文体参照（このエッセイ全体で統一）
{style_sample}

## テーマ
{topic}

## 参照ソース
{source_list}

## 弁証法的思考の履歴（要約）
{dialectical_history}

## 最終仮説（到達点）
{final_thesis}

---
**品質基準:** 文体は「である・だ」調（論考・学術エッセイ風）。「です・ます」調は使わない。
具体的な概念名・思想家名を積極的に引用すること。Markdownのみで出力。指示文を出力しないこと。
"""

_ABSTRACT_PROMPT = """\
あなたは哲学・思想の統合研究者です。以下の弁証法的思考の全履歴を踏まえ、
統合論考の **アブストラクト** を日本語で執筆してください。

{common_header}

**指示:**
`## アブストラクト` の見出しで始め、600〜800字で論考全体の主張を提示する。
弁証法的過程（定立→反証→止揚）を経て何が明らかになったかを宣言する。
末尾に本論考が開く「新たな問い」を1文で示す。
"""

_SECTION_PROMPTS = [
    (
        "第一部：問題の地平と初期定立",
        """\
あなたは哲学・思想の統合研究者です。統合論考の **第一部** を日本語で執筆してください。

{common_header}

**指示:**
`## 第一部：問題の地平と初期定立` の見出しで始め、1,800〜2,200字で執筆する。
- 現代における問題意識（なぜ今この問いか）を100〜200字で開く
- 参照ソースの主要概念を具体的に引用しながら初期仮説（定立）を論じる
- `###` サブ見出しを2〜3個使い、論を展開する
- 末尾で「しかし、この定立には〜という問題が潜んでいる」と次部への橋渡しをする
""",
    ),
    (
        "第二部：反証と批判的検証",
        """\
あなたは哲学・思想の統合研究者です。統合論考の **第二部** を日本語で執筆してください。

{common_header}

**指示:**
`## 第二部：反証と批判的検証` の見出しで始め、2,000〜2,500字で執筆する。
- 第一部の定立に対し、弁証法的思考の履歴に記録された対抗エビデンスを用いて批判的に検証する
- 「AはXを主張するが、Bの視点からはYという反証が成立する。この対立は〜を示唆する」の形で論を展開
- `###` サブ見出しを2〜3個使い、何が崩れ・何が残ったかを丁寧に論じる
- 末尾で「しかしこの反証もまた〜という限界を持つ」と止揚への問いを立てる
""",
    ),
    (
        "第三部：止揚と精緻化",
        """\
あなたは弁証法的思考を行う哲学・思想の統合研究者です。統合論考の **第三部** を日本語で執筆してください。

{common_header}

**指示:**
`## 第三部：止揚と精緻化` の見出しで始め、2,000〜2,500字で執筆する。
- 各ループで仮説がどう進化したかを示す（単なる折衷ではなく、矛盾を内包しつつ高次に統合）
- ヘーゲル的な止揚（Aufhebung）の過程を、具体的な概念の変容として追跡する
- `###` サブ見出しを2〜3個使い、最終仮説の到達点を論じる
- 末尾で「この精緻化が開く地平は〜である」と第四部への橋渡しをする
""",
    ),
    (
        "第四部：統合的洞察と現代的含意",
        """\
あなたは哲学・思想の統合研究者です。統合論考の **第四部** を日本語で執筆してください。

{common_header}

**指示:**
`## 第四部：統合的洞察と現代的含意` の見出しで始め、2,000〜2,500字で執筆する。
- 最終仮説から見える現代社会・テクノロジー・人間への具体的示唆を論じる
- 抽象論に留まらず、現代の具体的事象（AI・テクノロジー・社会変化など）に適用して考察する
- `###` サブ見出しを2〜3個使う
- 読者が「明日から何を問い直すべきか」が見えるような示唆で締める
""",
    ),
]

_CONCLUSION_PROMPT = """\
あなたは哲学・思想の統合研究者です。

## テーマ
{topic}

## 最終仮説（到達点）
{final_thesis}

**指示:**
`## 結論：問いの継承` の見出しで始め、700〜900字で執筆する。
- 文体は「である・だ」調（論考・学術エッセイ風）。「です・ます」調は使わない
- この論考全体が到達した知的地点を200字以内で総括する
- この論考が「閉じる問い」ではなく「開く問い」として何を継承するかを500字程度で論じる
- 読者への問いかけで締めくくる
- 指示文を出力しないこと。Markdownのみ
"""


# ── メイン関数 ────────────────────────────────────────────────────────────────

def run_dialectical(
    client: Neo4jClient,
    source_ids: list[str],
    topic: str,
    model: str = "gemma4:26b",
    loops: int = 2,
    num_ctx: int = 32768,
) -> tuple[str, list[DialecticalStep]]:
    """弁証法的仮説検証ループを実行し、統合論考と履歴を返す。

    Args:
        source_ids: 主要ソースID（書籍 + podcast の組み合わせ）
        topic:      論考テーマ（自由記述）
        model:      使用する Ollama モデル
        loops:      仮説精緻化ループ数（推奨: 2〜3）

    Returns:
        (final_essay_markdown, steps_history)
    """
    # ── ソースタイトル取得 ──────────────────────────────────────────────────────
    src_rows = client.run_cypher(
        "MATCH (s:Source) WHERE s.source_id IN $ids RETURN s.source_id, s.title",
        {"ids": source_ids},
    )
    source_list = "\n".join(
        f"- [{r['s.source_id']}] {r['s.title'] or r['s.source_id']}"
        for r in src_rows
    )

    # ── ① 素材収集 ────────────────────────────────────────────────────────────
    print(f"\n[dialectical] 素材収集中... source_ids={source_ids}", flush=True)
    concepts, persons = _fetch_primary_material(client, source_ids)
    print(f"  concepts={len(concepts)}  persons={len(persons)}", flush=True)

    material = _format_concepts(concepts, persons)

    # ── ② 初期仮説（定立）生成 ────────────────────────────────────────────────
    print(f"[dialectical] 初期仮説（定立）を生成中...", flush=True)
    thesis = llm.call(
        _THESIS_PROMPT.format(topic=topic, source_list=source_list, material=material),
        num_ctx=num_ctx, num_predict=2000, model=model,
    )
    print(f"  thesis={len(thesis)}字", flush=True)

    steps: list[DialecticalStep] = []

    # ── ③ 弁証法ループ ────────────────────────────────────────────────────────
    for i in range(loops):
        loop_num = i + 1
        print(f"[dialectical] ループ {loop_num}/{loops}: 反証収集中...", flush=True)

        antithesis_rows = _fetch_antithesis_material(client, source_ids, concepts)
        print(f"  antithesis_entities={len(antithesis_rows)}", flush=True)
        antithesis_text = _format_antithesis(antithesis_rows)

        print(f"[dialectical] ループ {loop_num}/{loops}: 検証評価中...", flush=True)
        evaluation = llm.call(
            _ANTITHESIS_PROMPT.format(
                thesis=thesis,
                antithesis_material=antithesis_text,
                loop=loop_num,
            ),
            num_ctx=num_ctx, num_predict=2000, model=model,
        )
        print(f"  evaluation={len(evaluation)}字", flush=True)

        print(f"[dialectical] ループ {loop_num}/{loops}: 止揚（精緻化）中...", flush=True)
        refined = llm.call(
            _REFINEMENT_PROMPT.format(thesis=thesis, evaluation=evaluation),
            num_ctx=num_ctx, num_predict=2000, model=model,
        )
        print(f"  refined_thesis={len(refined)}字", flush=True)

        steps.append(DialecticalStep(
            loop=loop_num,
            thesis=thesis,
            antithesis_entities=antithesis_rows,
            evaluation=evaluation,
            refined_thesis=refined,
        ))
        thesis = refined  # 次ループの入力

    # ── ④ 弁証法履歴のフォーマット ────────────────────────────────────────────
    history_parts: list[str] = []
    for step in steps:
        history_parts.append(f"### ループ {step.loop} — 定立\n{step.thesis}")
        history_parts.append(f"### ループ {step.loop} — 検証評価\n{step.evaluation}")
        history_parts.append(f"### ループ {step.loop} — 止揚\n{step.refined_thesis}")
    dialectical_history = "\n\n".join(history_parts)

    # ── ⑤ 統合論考の生成（セクション別） ────────────────────────────────────────
    common_kwargs = dict(
        topic=topic,
        source_list=source_list,
        dialectical_history=dialectical_history[:3000],
        final_thesis=thesis,
    )

    # アブストラクト（スタイルサンプル抽出元）
    print(f"[dialectical] アブストラクト生成中...", flush=True)
    abstract_md = llm.call(
        _ABSTRACT_PROMPT.format(
            common_header=_COMMON_HEADER.format(style_sample="（最初のセクションなので参照なし）", **common_kwargs)
        ),
        num_ctx=num_ctx, num_predict=2000, model=model,
    )
    print(f"  abstract={len(abstract_md)}字", flush=True)

    # スタイルサンプルをアブストラクトから抽出
    style_lines = [l for l in abstract_md.splitlines() if l.strip() and not l.startswith("#")]
    style_sample = " ".join(style_lines)[:400]

    common_header = _COMMON_HEADER.format(style_sample=style_sample, **common_kwargs)

    # 本文セクション
    section_parts: list[str] = []
    for sec_title, sec_prompt_tpl in _SECTION_PROMPTS:
        print(f"[dialectical] {sec_title} 生成中...", flush=True)
        sec_md = llm.call(
            sec_prompt_tpl.format(common_header=common_header),
            num_ctx=num_ctx, num_predict=4000, model=model,
        )
        print(f"  {sec_title}={len(sec_md)}字", flush=True)
        section_parts.append(sec_md)

    # 結論（コンテキスト節約のため topic + final_thesis のみ渡す）
    print(f"[dialectical] 結論 生成中...", flush=True)
    conclusion_md = ""
    for _attempt in range(3):
        conclusion_md = llm.call(
            _CONCLUSION_PROMPT.format(topic=topic, final_thesis=thesis),
            num_ctx=16384, num_predict=2000, model=model,
        )
        if conclusion_md.strip():
            break
        print(f"  [warn] 結論が空 (attempt {_attempt + 1}/3)、リトライ中...", flush=True)
    print(f"  conclusion={len(conclusion_md)}字", flush=True)

    # 参考ソース
    sources_md = "## 参考ソース\n\n" + source_list

    # 組み立て
    parts = [abstract_md] + section_parts + [conclusion_md, sources_md]
    essay = "\n\n---\n\n".join(parts)

    total = len(essay)
    print(f"[dialectical] 完了: essay={total:,}字, sections={len(parts)}, steps={len(steps)}", flush=True)

    return essay, steps


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    import argparse
    import os
    from datetime import date
    from pathlib import Path

    parser = argparse.ArgumentParser(
        description="弁証法的仮説検証サイクルで統合論考を生成する"
    )
    parser.add_argument("--topic", required=True, help="論考テーマ（自由記述）")
    parser.add_argument(
        "--source-ids", required=True,
        help="主要ソースID（カンマ区切り、例: 'book_wittgenstein_...,book_kaiwa_...'）",
    )
    parser.add_argument("--model", default="gemma4:26b")
    parser.add_argument("--loops", type=int, default=2, help="仮説精緻化ループ数")
    parser.add_argument("--num-ctx", type=int, default=32768)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--publish", action="store_true", help="homupe へ投稿")
    args = parser.parse_args()

    os.environ.setdefault("NEO4J_URI", "bolt://localhost:7687")
    os.environ.setdefault("NEO4J_USER", "neo4j")
    os.environ.setdefault("NEO4J_PASSWORD", "factfull123")
    os.environ.setdefault("FACTFULL_LLM_BACKEND", "ollama")

    source_ids = [s.strip() for s in args.source_ids.split(",")]
    out_path = args.out or Path(f"/tmp/dialectical_{date.today().isoformat()}.md")

    with Neo4jClient() as client:
        essay, steps = run_dialectical(
            client,
            source_ids=source_ids,
            topic=args.topic,
            model=args.model,
            loops=args.loops,
            num_ctx=args.num_ctx,
        )

    out_path.write_text(essay, encoding="utf-8")
    print(f"\n[dialectical] → {out_path}  ({len(essay):,}字)", flush=True)

    if args.publish:
        from factfull.synthesis.cross_source import _publish_to_homupe  # type: ignore
        _publish_to_homupe(essay, args.topic, "Synthesis")


if __name__ == "__main__":
    main()
