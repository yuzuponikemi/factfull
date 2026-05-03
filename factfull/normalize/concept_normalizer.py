"""
factfull/normalize/concept_normalizer.py
==========================================
Neo4j のコンセプト系エンティティを重複マージで正規化する。

Phase 1 — 文字列正規化マージ:
  括弧内テキスト除去 + 小文字化で同一視できるノードを統合する。
  例: "LLM", "LLM (Large Language Model)", "LLMs" → 最長の名前を canonical に選択

Phase 2 — 埋め込みクラスタリング（オプション）:
  Ollama embeddings + コサイン類似度で意味的重複を検出。
  --embed フラグで有効化。

使い方:
    python -m factfull.normalize.concept_normalizer --dry-run
    python -m factfull.normalize.concept_normalizer --types concept,framework,method,product
    python -m factfull.normalize.concept_normalizer --embed --threshold 0.93
"""
from __future__ import annotations

import argparse
import re
from collections import defaultdict
from typing import Sequence

from factfull.graph.neo4j import Neo4jClient
from factfull.normalize.entity_normalizer import _rename_entity


_DEFAULT_TYPES = ("concept", "framework", "method", "product", "theory")
_EMBED_THRESHOLD = 0.93   # コサイン類似度の閾値


# ── Phase 1: 文字列正規化 ─────────────────────────────────────────────────────

def _strip_parens(s: str) -> str:
    """括弧・角括弧内テキストを除去して小文字に正規化する。"""
    s = re.sub(r"\s*[\(\[].*?[\)\]]", "", s)
    return s.strip().lower()


def _pick_canonical(names: list[str]) -> str:
    """重複グループから canonical 名を選ぶ。

    優先順位:
      1. 括弧付きで最長（情報量が多い）
      2. 先頭大文字のもの
      3. 単純に最長
    """
    with_parens = [n for n in names if re.search(r"[\(\[]", n)]
    if with_parens:
        return max(with_parens, key=len)
    titled = [n for n in names if n and n[0].isupper()]
    if titled:
        return max(titled, key=len)
    return max(names, key=len)


def _strip_plural(s: str) -> str:
    """末尾の複数形 's' を正規化する（簡易版）。"""
    if s.endswith("s") and len(s) > 3:
        return s[:-1]
    return s


def find_acronym_duplicates(
    client: Neo4jClient,
    types: Sequence[str] = _DEFAULT_TYPES,
) -> list[tuple[str, list[str]]]:
    """括弧内の略語を展開してクロスグループの重複を検出する。

    例:
      'Large Language Model (LLM)'  → 略語 'llm'
      'LLM (Large Language Models)' → 略語 'llm'
      'LLMs (Large Language Models)'→ 略語 'llms' → singular 'llm'
    これらは同じ略語キーなのでグループ化してマージする。

    Returns:
        list of (canonical_name, [alias1, alias2, ...])
    """
    rows = client.run_cypher(
        "MATCH (e:Entity) WHERE e.type IN $types RETURN e.name AS name ORDER BY e.name",
        {"types": list(types)},
    )
    names = [r["name"] for r in rows]

    # 各名前から「略語キー」を抽出する
    # パターン A: 'Full Name (ABBREV)' → 略語 abbrev
    # パターン B: 'ABBREV (Full Name)' → 略語 abbrev
    abbrev_re = re.compile(r"[\(\[]([A-Z][A-Z0-9\-]{1,12})[\)\]]")

    def extract_abbrev_key(name: str) -> str | None:
        m = abbrev_re.search(name)
        if m:
            abbrev = m.group(1).lower()
            return _strip_plural(abbrev)
        # 名前自体が大文字略語の場合 (例: 'LLM', 'LLMs', 'RL')
        stripped = _strip_parens(name)
        if re.match(r"^[a-z]{2,8}$", stripped) and name == name.upper().replace(" ", ""):
            return _strip_plural(stripped)
        return None

    # 略語キーでグループ化
    groups: dict[str, list[str]] = defaultdict(list)
    for name in names:
        key = extract_abbrev_key(name)
        if key:
            groups[key].append(name)

    # 裸の略語（括弧なし・展開形なし）を持つグループのみ安全にマージ
    # 例: 'RLHF' は裸の略語 → 'Reinforcement Learning from Human Feedback (RLHF)' と統合可
    # 例: 'ImageCast Evolution (ICE)' と 'Internal Combustion Engine (ICE)' はどちらも展開形あり → スキップ
    def is_bare_abbrev(name: str) -> bool:
        return not re.search(r"[\(\[]", name)

    result = []
    for key, group_names in groups.items():
        if len(group_names) < 2:
            continue
        bare = [n for n in group_names if is_bare_abbrev(n)]
        expanded = [n for n in group_names if not is_bare_abbrev(n)]
        if not bare:
            continue  # 全員が展開形を持つ場合は略語衝突の可能性があるのでスキップ
        if not expanded:
            continue  # 裸の略語しかいない場合もスキップ
        # 裸の略語 → 最長の展開形 に統合
        canonical = _pick_canonical(expanded)
        aliases = bare + [n for n in expanded if n != canonical]
        result.append((canonical, aliases))

    return sorted(result, key=lambda x: x[0].lower())


def merge_acronym_duplicates(
    client: Neo4jClient,
    types: Sequence[str] = _DEFAULT_TYPES,
    dry_run: bool = False,
) -> dict[str, int]:
    """略語ベースのクロスグループ重複をマージする。"""
    groups = find_acronym_duplicates(client, types)
    stats = {"groups": len(groups), "merged": 0, "failed": 0}

    print(f"  重複グループ (略語): {len(groups)} 件", flush=True)
    for canonical, aliases in groups:
        for alias in aliases:
            if dry_run:
                print(f"  [DRY] MERGE  '{alias}' → '{canonical}'", flush=True)
                stats["merged"] += 1
                continue
            try:
                _rename_entity(client, alias, canonical)
                print(f"  MERGE  '{alias}' → '{canonical}'", flush=True)
                stats["merged"] += 1
            except Exception as e:
                print(f"  FAIL   '{alias}' → '{canonical}': {e}", flush=True)
                stats["failed"] += 1

    return stats


def find_string_duplicates(
    client: Neo4jClient,
    types: Sequence[str] = _DEFAULT_TYPES,
) -> list[tuple[str, list[str]]]:
    """括弧除去+小文字化で重複するエンティティグループを返す。

    Returns:
        list of (canonical_name, [alias1, alias2, ...])
        aliases には canonical 以外の名前が入る
    """
    rows = client.run_cypher(
        """
        MATCH (e:Entity)
        WHERE e.type IN $types
        RETURN e.name AS name
        ORDER BY e.name
        """,
        {"types": list(types)},
    )

    groups: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        key = _strip_parens(row["name"])
        groups[key].append(row["name"])

    result = []
    for names in groups.values():
        if len(names) < 2:
            continue
        canonical = _pick_canonical(names)
        aliases = [n for n in names if n != canonical]
        result.append((canonical, aliases))

    return sorted(result, key=lambda x: x[0].lower())


def merge_string_duplicates(
    client: Neo4jClient,
    types: Sequence[str] = _DEFAULT_TYPES,
    dry_run: bool = False,
) -> dict[str, int]:
    """文字列ベースの重複を検出してマージする。"""
    groups = find_string_duplicates(client, types)
    stats = {"groups": len(groups), "merged": 0, "failed": 0}

    print(f"  重複グループ (文字列): {len(groups)} 件", flush=True)

    for canonical, aliases in groups:
        for alias in aliases:
            if dry_run:
                print(f"  [DRY] MERGE  '{alias}' → '{canonical}'", flush=True)
                stats["merged"] += 1
                continue
            try:
                _rename_entity(client, alias, canonical)
                print(f"  MERGE  '{alias}' → '{canonical}'", flush=True)
                stats["merged"] += 1
            except Exception as e:
                print(f"  FAIL   '{alias}' → '{canonical}': {e}", flush=True)
                stats["failed"] += 1

    return stats


# ── Phase 2: 埋め込みクラスタリング ──────────────────────────────────────────

def _get_embedding(text: str, model: str = "nomic-embed-text") -> list[float] | None:
    """Ollama REST API でテキストの埋め込みベクトルを取得する。"""
    import json
    import urllib.request

    payload = json.dumps({"model": model, "prompt": text}).encode()
    try:
        req = urllib.request.Request(
            "http://localhost:11434/api/embeddings",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
            return data.get("embedding")
    except Exception:
        return None


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _name_token_overlap(a: str, b: str) -> float:
    """名前の単語トークン重複率（Jaccard）。0 なら共通語なし。"""
    def tokens(s: str) -> set[str]:
        return {t.lower() for t in re.split(r"[\s\-_/]+", s) if len(t) > 2}
    ta, tb = tokens(a), tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def find_embedding_duplicates(
    client: Neo4jClient,
    types: Sequence[str] = _DEFAULT_TYPES,
    threshold: float = _EMBED_THRESHOLD,
    embed_model: str = "mxbai-embed-large",
    limit: int = 500,
    min_name_overlap: float = 0.0,
) -> list[tuple[str, str, float]]:
    """埋め込みベクトルのコサイン類似度で意味的重複ペアを検出する。

    embed テキストは "name: description" — 文脈を含めることで短い名前のノイズを減らす。
    min_name_overlap > 0 を設定すると名前トークン重複がない誤検知を除外できる。

    Returns:
        list of (name_a, name_b, similarity) — 閾値以上のペア
    """
    rows = client.run_cypher(
        """
        MATCH (e:Entity)
        WHERE e.type IN $types
        RETURN e.name AS name, e.description AS description
        ORDER BY e.confidence DESC
        LIMIT $limit
        """,
        {"types": list(types), "limit": limit},
    )
    entries = [(r["name"], (r.get("description") or "")[:200]) for r in rows]
    total = len(entries)
    print(f"  埋め込み計算中: {total} 件 (model={embed_model})...", flush=True)

    embeddings: dict[str, list[float]] = {}
    descriptions: dict[str, str] = {}
    for i, (name, desc) in enumerate(entries, 1):
        if i % 50 == 0:
            print(f"    {i}/{total}", flush=True, end="\r")
        embed_text = f"{name}: {desc}" if desc else name
        vec = _get_embedding(embed_text, model=embed_model)
        if vec:
            embeddings[name] = vec
            descriptions[name] = desc

    print(f"    {total}/{total} 完了", flush=True)

    pairs = []
    names_with_emb = list(embeddings.keys())
    for i in range(len(names_with_emb)):
        for j in range(i + 1, len(names_with_emb)):
            na, nb = names_with_emb[i], names_with_emb[j]
            if min_name_overlap > 0 and _name_token_overlap(na, nb) < min_name_overlap:
                continue
            sim = _cosine_similarity(embeddings[na], embeddings[nb])
            if sim >= threshold:
                pairs.append((na, nb, sim))

    return sorted(pairs, key=lambda x: -x[2])


def merge_embedding_duplicates(
    client: Neo4jClient,
    types: Sequence[str] = _DEFAULT_TYPES,
    threshold: float = _EMBED_THRESHOLD,
    embed_model: str = "mxbai-embed-large",
    dry_run: bool = True,
    limit: int = 500,
    min_name_overlap: float = 0.0,
) -> dict[str, int]:
    """埋め込みベースの意味的重複を検出してマージする。

    dry_run=True（デフォルト）では候補を表示するのみ。
    実際にマージするには --no-dry-run を明示的に指定する。
    """
    pairs = find_embedding_duplicates(
        client, types, threshold, embed_model, limit, min_name_overlap
    )
    stats = {"candidates": len(pairs), "merged": 0, "failed": 0, "skipped": 0}

    print(f"\n  意味的重複候補: {len(pairs)} ペア (threshold={threshold}, model={embed_model})", flush=True)
    if dry_run:
        print("  [DRY-RUN] 変更なし — 確認後 --no-dry-run で実行\n", flush=True)

    for na, nb, sim in pairs:
        canonical = _pick_canonical([na, nb])
        alias = nb if canonical == na else na
        overlap = _name_token_overlap(na, nb)
        print(f"  sim={sim:.3f}  overlap={overlap:.2f}  '{alias}' → '{canonical}'", flush=True)

        if dry_run:
            stats["skipped"] += 1
            continue
        try:
            _rename_entity(client, alias, canonical)
            stats["merged"] += 1
        except Exception as e:
            print(f"    FAIL: {e}", flush=True)
            stats["failed"] += 1

    return stats


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="コンセプト系エンティティの重複マージ")
    parser.add_argument(
        "--types",
        default=",".join(_DEFAULT_TYPES),
        help="対象エンティティタイプ（カンマ区切り）",
    )
    parser.add_argument("--dry-run", action="store_true", help="Phase 1/1b を dry-run（変更なし）")
    parser.add_argument("--embed", action="store_true", help="Phase 2: 埋め込みクラスタリングも実行")
    parser.add_argument(
        "--threshold", type=float, default=_EMBED_THRESHOLD,
        help=f"埋め込み類似度の閾値（デフォルト: {_EMBED_THRESHOLD}）",
    )
    parser.add_argument("--embed-model", default="mxbai-embed-large", help="Ollama 埋め込みモデル")
    parser.add_argument(
        "--embed-merge", action="store_true",
        help="Phase 2 で実際にマージを実行（デフォルトは候補表示のみ）",
    )
    parser.add_argument("--min-name-overlap", type=float, default=0.0, help="名前トークン重複率の下限（0=無効）")
    parser.add_argument("--limit", type=int, default=500, help="埋め込みフェーズの処理件数上限")
    args = parser.parse_args()

    types = [t.strip() for t in args.types.split(",")]

    with Neo4jClient() as client:
        print("\n=== Phase 1: 文字列正規化マージ ===")
        stats1 = merge_string_duplicates(client, types=types, dry_run=args.dry_run)
        print(f"  結果: {stats1}")

        print("\n=== Phase 1b: 略語展開マッチング ===")
        stats1b = merge_acronym_duplicates(client, types=types, dry_run=args.dry_run)
        print(f"  結果: {stats1b}")

        if args.embed:
            # Phase 2 は --embed-merge を明示しない限り dry-run（候補表示のみ）
            embed_dry_run = not args.embed_merge
            print("\n=== Phase 2: 埋め込みクラスタリング ===")
            if embed_dry_run:
                print("  [DRY-RUN] 候補表示のみ。実際にマージするには --embed-merge を追加してください。")
            stats2 = merge_embedding_duplicates(
                client,
                types=types,
                threshold=args.threshold,
                embed_model=args.embed_model,
                dry_run=embed_dry_run,
                limit=args.limit,
                min_name_overlap=args.min_name_overlap,
            )
            print(f"  結果: {stats2}")


if __name__ == "__main__":
    main()
