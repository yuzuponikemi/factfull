"""
factfull/synthesis/kg_analysis.py
===================================
ナレッジグラフの構造解析 + LLM によるインサイト生成。

分析の柱:
  1. ブリッジエンティティ — 3人以上の話者が独自の claim を持つ概念
  2. クロス話者 claim ペア — 同一トピックに対する対立・収束する主張
  3. LLM インサイト — 上記データから人間が気づきにくい構造を言語化

使い方:
    python -m factfull.synthesis.kg_analysis
    python -m factfull.synthesis.kg_analysis --out /tmp/insights.md
    python -m factfull.synthesis.kg_analysis --publish   # homupe へ投稿
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from factfull.graph.neo4j import Neo4jClient
from factfull.publishers import ALLOWED_CATEGORIES

assert "Synthesis" in ALLOWED_CATEGORIES


# ── データクラス ───────────────────────────────────────────────────────────────

@dataclass
class BridgeEntity:
    name: str
    entity_type: str
    speaker_count: int
    speakers: list[str]
    sample_claims: list[dict]   # {speaker, claim, description}


@dataclass
class ClaimPair:
    entity: str                 # 共通トピック
    speaker1: str
    claim1: str
    desc1: str
    speaker2: str
    claim2: str
    desc2: str
    relation: str               # "opposing" | "converging" | "complementary"


# ── Cypher クエリ ──────────────────────────────────────────────────────────────

_Q_BRIDGE = """
MATCH (p:Entity {type: 'person'})-[:ARGUES_THAT|SAYS]->(concept:Entity)
WHERE concept.type IN ['concept', 'framework', 'method', 'theory', 'product']
WITH concept,
     collect(DISTINCT {speaker: p.name, claim: concept.name, desc: concept.description}) AS claim_data,
     collect(DISTINCT p.name) AS speakers,
     count(DISTINCT p.name) AS n
WHERE n >= $min_speakers
RETURN concept.name AS entity,
       concept.type AS type,
       n AS speaker_count,
       speakers,
       claim_data
ORDER BY n DESC
LIMIT $limit
"""

_Q_CROSS_CLAIMS = """
MATCH (p1:Entity {type: 'person'})-[:ARGUES_THAT|SAYS]->(c1:Entity)
MATCH (p2:Entity {type: 'person'})-[:ARGUES_THAT|SAYS]->(c2:Entity)
WHERE p1.name < p2.name
  AND c1 <> c2
  AND c1.type IN ['concept', 'framework', 'method', 'theory', 'product', 'claim']
  AND c2.type IN ['concept', 'framework', 'method', 'theory', 'product', 'claim']
  AND (
    (c1)-[:RELATED_TO|PART_OF|IS_A|BASED_ON|CRITICIZES]-(c2)
  )
WITH p1, c1, p2, c2
RETURN p1.name  AS speaker1,
       c1.name  AS claim1,
       c1.description AS desc1,
       p2.name  AS speaker2,
       c2.name  AS claim2,
       c2.description AS desc2
LIMIT $limit
"""

_Q_SHARED_CLAIM_TOPIC = """
MATCH (p1:Entity {type: 'person'})-[:ARGUES_THAT]->(c1:Entity)
MATCH (p2:Entity {type: 'person'})-[:ARGUES_THAT]->(c2:Entity)
MATCH (e:Entity)
WHERE p1.name < p2.name
  AND (c1)-[:RELATED_TO|ABOUT|APPLIES_TO]-(e)
  AND (c2)-[:RELATED_TO|ABOUT|APPLIES_TO]-(e)
  AND c1 <> c2
  AND e.type IN ['concept', 'framework', 'theory', 'product']
RETURN p1.name AS speaker1, c1.name AS claim1, c1.description AS desc1,
       p2.name AS speaker2, c2.name AS claim2, c2.description AS desc2,
       e.name  AS shared_topic
ORDER BY e.confidence DESC
LIMIT $limit
"""


# ── 分析関数 ───────────────────────────────────────────────────────────────────

def find_bridge_entities(
    client: Neo4jClient,
    min_speakers: int = 3,
    limit: int = 15,
) -> list[BridgeEntity]:
    """3人以上の話者が独自の claim を持つエンティティ（ブリッジ）を返す。"""
    rows = client.run_cypher(_Q_BRIDGE, {"min_speakers": min_speakers, "limit": limit})
    result = []
    for r in rows:
        # claim_data から speaker ごとに最初の claim を抽出
        seen: set[str] = set()
        sample_claims = []
        for cd in r["claim_data"]:
            spk = cd.get("speaker", "")
            if spk and spk not in seen:
                seen.add(spk)
                sample_claims.append(cd)
            if len(sample_claims) >= 5:
                break
        result.append(BridgeEntity(
            name=r["entity"],
            entity_type=r["type"],
            speaker_count=r["speaker_count"],
            speakers=r["speakers"][:8],
            sample_claims=sample_claims,
        ))
    return result


def find_cross_speaker_pairs(
    client: Neo4jClient,
    limit: int = 20,
) -> list[dict[str, Any]]:
    """RELATED_TO で繋がる claim を持つ 2 話者ペアを返す。"""
    return client.run_cypher(_Q_CROSS_CLAIMS, {"limit": limit})


def find_shared_topic_pairs(
    client: Neo4jClient,
    limit: int = 20,
) -> list[dict[str, Any]]:
    """共通トピックに対して異なる claim を持つ話者ペアを返す。"""
    return client.run_cypher(_Q_SHARED_CLAIM_TOPIC, {"limit": limit})


# ── プロンプト構築 ─────────────────────────────────────────────────────────────

def _build_insight_prompt(
    bridges: list[BridgeEntity],
    cross_pairs: list[dict],
    shared_pairs: list[dict],
) -> str:
    # ブリッジエンティティ
    bridge_lines = []
    for b in bridges[:10]:
        spk_str = " / ".join(b.speakers[:6])
        bridge_lines.append(f"\n**{b.name}** [{b.entity_type}] — {b.speaker_count}人の話者")
        bridge_lines.append(f"  話者: {spk_str}")
        for c in b.sample_claims[:3]:
            desc = (c.get("desc") or "")[:100]
            bridge_lines.append(f"  - [{c['speaker']}] {desc}")

    # クロス話者 claim ペア
    cross_lines = []
    for r in cross_pairs[:12]:
        d1 = (r.get("desc1") or r.get("claim1", ""))[:90]
        d2 = (r.get("desc2") or r.get("claim2", ""))[:90]
        cross_lines.append(
            f"  {r['speaker1']}: 「{d1}」\n"
            f"  {r['speaker2']}: 「{d2}」"
        )

    # 共通トピックペア
    topic_lines = []
    seen_topics: set[str] = set()
    for r in shared_pairs[:12]:
        topic = r.get("shared_topic", "")
        if topic in seen_topics:
            continue
        seen_topics.add(topic)
        d1 = (r.get("desc1") or r.get("claim1", ""))[:90]
        d2 = (r.get("desc2") or r.get("claim2", ""))[:90]
        topic_lines.append(
            f"  共通テーマ: **{topic}**\n"
            f"  {r['speaker1']}: 「{d1}」\n"
            f"  {r['speaker2']}: 「{d2}」"
        )

    bridge_text = "\n".join(bridge_lines) or "（なし）"
    cross_text = "\n\n".join(cross_lines) or "（なし）"
    topic_text = "\n\n".join(topic_lines) or "（なし）"

    return f"""あなたは複数のポッドキャストのナレッジグラフを解析する知識人です。
以下のデータから、単一エピソードを聞くだけでは気づけない「横断的な洞察」を日本語で生成してください。

---

## 1. ブリッジエンティティ（複数話者が独自の claim を持つ概念）
これらは「知的交差点」です。複数の専門家が全く異なる文脈から同じ概念に辿り着いています。
{bridge_text}

---

## 2. 関連する claim を持つ話者ペア
異なるエピソードで、似た関係にある主張をした話者の組み合わせです。
{cross_text}

---

## 3. 共通テーマへの異なるアプローチ
同じ概念に対して異なる角度からアプローチした話者のペアです。
{topic_text}

---

## 出力フォーマット（Markdown）

以下の構成で **3,000〜5,000文字** のインサイトレポートを生成してください:

### 構成
1. `## エグゼクティブサマリー` — 最も重要な発見を3〜5箇条
2. `## ブリッジ概念の解説`（上位3〜4件）
   - なぜこの概念が知的交差点になっているのか
   - 異なる話者がこの概念をどう位置づけているか（比較）
   - 単独では見えなかった何が見えるか
3. `## 話者間の対話`（注目すべきクロス話者ペア3〜4件）
   - 直接対話していないが、主張が「呼応」または「対立」している箇所
   - この対話から引き出せる問い
4. `## 結論：知識グラフが示すパターン`
   - データ全体から浮かぶ構造的なパターン
   - 今後の探索に値する仮説

### スタイル
- 固有名詞は **太字**
- 具体的な主張を引用してから分析する
- 「AとBの主張は表面上異なるが、実は〜という共通の前提を持っている」のような形で論じる
"""


# ── 公開 API ───────────────────────────────────────────────────────────────────

def analyze_and_generate(
    client: Neo4jClient,
    model: str = "gemma4:26b",
    min_bridge_speakers: int = 3,
) -> str:
    """KG を解析して LLM インサイトレポートを生成する。"""
    print("  ブリッジエンティティ解析中...", flush=True)
    bridges = find_bridge_entities(client, min_speakers=min_bridge_speakers)
    print(f"  → {len(bridges)} 件", flush=True)

    print("  クロス話者 claim ペア解析中...", flush=True)
    cross_pairs = find_cross_speaker_pairs(client)
    print(f"  → {len(cross_pairs)} 件", flush=True)

    print("  共通トピックペア解析中...", flush=True)
    shared_pairs = find_shared_topic_pairs(client)
    print(f"  → {len(shared_pairs)} 件", flush=True)

    prompt = _build_insight_prompt(bridges, cross_pairs, shared_pairs)
    print(f"  プロンプト: {len(prompt)} 文字", flush=True)

    from factfull import llm
    return llm.call(prompt, num_ctx=16384, num_predict=8192, timeout=3600, model=model)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    import argparse
    import os
    from datetime import date
    from pathlib import Path

    parser = argparse.ArgumentParser(description="KG 構造解析 + LLM インサイト生成")
    parser.add_argument("--model", default="gemma4:26b")
    parser.add_argument("--min-speakers", type=int, default=2)
    parser.add_argument("--out", type=Path, default=Path("/tmp/kg_insights.md"))
    parser.add_argument("--publish", action="store_true", help="homupe へ投稿")
    parser.add_argument("--dry-run", action="store_true", help="データ解析のみ（LLM なし）")
    args = parser.parse_args()

    os.environ.setdefault("NEO4J_URI", "bolt://localhost:7687")
    os.environ.setdefault("NEO4J_USER", "neo4j")
    os.environ.setdefault("NEO4J_PASSWORD", "factfull123")
    os.environ.setdefault("FACTFULL_LLM_BACKEND", "ollama")
    os.environ.setdefault("FACTFULL_OLLAMA_MODEL", args.model)

    with Neo4jClient() as client:
        stats = client.get_statistics()
        print(f"[neo4j] sources={stats['sources']}  entities={stats['entities']}", flush=True)

        if args.dry_run:
            bridges = find_bridge_entities(client, min_speakers=args.min_speakers)
            print(f"\n=== Bridge entities ({len(bridges)}) ===")
            for b in bridges:
                print(f"  {b.speaker_count}spk  [{b.entity_type}] {b.name}")
                for c in b.sample_claims[:2]:
                    print(f"    [{c['speaker']}] {(c.get('desc') or '')[:80]}")
            cross = find_cross_speaker_pairs(client)
            print(f"\n=== Cross pairs ({len(cross)}) ===")
            for r in cross[:5]:
                print(f"  {r['speaker1']} ↔ {r['speaker2']}")
            return

        print("\n[analysis] インサイト生成中...", flush=True)
        report = analyze_and_generate(client, model=args.model, min_bridge_speakers=args.min_speakers)

    print(f"\n[analysis] 生成完了: {len(report)} 文字", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report, encoding="utf-8")
    print(f"[analysis] 保存: {args.out}", flush=True)

    if args.publish:
        from factfull.publishers.homupe import default_blog_dir
        today = date.today()
        blog_dir = default_blog_dir()
        blog_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{today.isoformat()}-kg-insights.md"
        post_path = blog_dir / filename

        frontmatter = f"""---
date: {today.isoformat()}
categories:
  - Synthesis
tags:
  - AI
  - ナレッジグラフ
  - 合成記事
  - KG解析
---

# KG インサイト：{today.isoformat()}

{stats['sources']} エピソードのナレッジグラフから抽出した横断的洞察。ブリッジエンティティと話者間の隠れた対話を分析する。

<!-- more -->

"""
        post_path.write_text(frontmatter + report, encoding="utf-8")
        print(f"[publish] 投稿: {post_path}", flush=True)

        import subprocess
        repo_root = post_path.parents[4]
        subprocess.run(["git", "-C", str(repo_root), "add", str(post_path)], check=True)
        subprocess.run(
            ["git", "-C", str(repo_root), "commit", "-m", f"feat: KG insights — {today.isoformat()}"],
            check=True,
        )
        print("[publish] git commit 完了", flush=True)

    print("\n── 冒頭 ──")
    print(report[:500])


if __name__ == "__main__":
    main()
