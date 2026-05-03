"""
factfull/normalize/wiki_linker.py
==================================
エンティティ名 → Wikipedia 正規名 + Wikidata QID へのリンキング。

人物・組織: Wikidata Search API（ネット必須、キャッシュあり）
概念・フレームワーク: localsearch-mcp の Wikipedia BM25 インデックス（ローカル）

使い方:
    from factfull.normalize.wiki_linker import WikiLinker

    with WikiLinker() as wl:
        r = wl.link("Dario Amodeary", entity_type="person")
        print(r.canonical_name)  # "Dario Amodei"
        print(r.wikidata_qid)    # "Q56017089"

        r2 = wl.link("Large Language Model", entity_type="concept")
        print(r2.canonical_name) # "Large language model"
"""
from __future__ import annotations

import json
import re
import sqlite3
import subprocess
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# ── データクラス ──────────────────────────────────────────────────────────────

@dataclass
class WikiLinkResult:
    query: str
    canonical_name: str
    wikipedia_url: str
    wikidata_qid: Optional[str]
    confidence: float          # 0.0–1.0
    is_disambiguation: bool = False
    found: bool = True
    method: str = ""           # "wikidata" | "local_wiki" | "none"


def _not_found(q: str) -> WikiLinkResult:
    return WikiLinkResult(
        query=q, canonical_name=q, wikipedia_url="",
        wikidata_qid=None, confidence=0.0, found=False, method="none",
    )


# ── 文字列類似度 ──────────────────────────────────────────────────────────────

def _title_confidence(query: str, title: str) -> float:
    q, t = query.lower().strip(), title.lower().strip()
    if q == t:
        return 1.0
    # t が q で始まる場合は高スコア — ただし数字サフィックスは除外
    # 例: "machine learning (ML)" OK, "Mercury 1" NG
    def _starts_with_qualifier(base: str, full: str) -> bool:
        if not full.startswith(base):
            return False
        raw_suffix = full[len(base):]
        if not raw_suffix.strip():
            return True
        # 括弧 or カンマ は qualifier（OK）
        if raw_suffix.lstrip()[0] in "([,":
            return True
        # スペース + 単語 は qualifier（OK）— ただし数字で始まる単語は除外
        if raw_suffix.startswith(" "):
            next_word = raw_suffix.strip().split()[0]
            return not next_word[0].isdigit()
        return False

    if _starts_with_qualifier(q, t) or _starts_with_qualifier(t, q):
        return 0.85
    qw = set(re.split(r"\W+", q)) - {""}
    tw = set(re.split(r"\W+", t)) - {""}
    if not qw or not tw:
        return 0.0
    return round(len(qw & tw) / max(len(qw), len(tw)), 3)


# ── SQLite キャッシュ ─────────────────────────────────────────────────────────

_DEFAULT_CACHE = Path.home() / ".factfull" / "wiki_link_cache.db"
_SENTINEL = "__NULL__"


class _LinkCache:
    def __init__(self, path: Path = _DEFAULT_CACHE):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path))
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS link_cache ("
            "  query TEXT PRIMARY KEY,"
            "  canonical_name TEXT,"
            "  wikipedia_url TEXT,"
            "  wikidata_qid TEXT,"
            "  confidence REAL,"
            "  method TEXT,"
            "  fetched_at TEXT DEFAULT CURRENT_TIMESTAMP"
            ")"
        )
        self._conn.commit()

    def get(self, query: str) -> Optional[WikiLinkResult]:
        row = self._conn.execute(
            "SELECT canonical_name, wikipedia_url, wikidata_qid, confidence, method "
            "FROM link_cache WHERE query = ?", (query,)
        ).fetchone()
        if row is None:
            return None
        canonical, url, qid_raw, conf, method = row
        return WikiLinkResult(
            query=query,
            canonical_name=canonical,
            wikipedia_url=url or "",
            wikidata_qid=None if qid_raw == _SENTINEL else qid_raw,
            confidence=conf,
            found=conf > 0,
            method=method or "",
        )

    def set(self, result: WikiLinkResult) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO link_cache "
            "(query, canonical_name, wikipedia_url, wikidata_qid, confidence, method) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                result.query,
                result.canonical_name,
                result.wikipedia_url,
                result.wikidata_qid if result.wikidata_qid is not None else _SENTINEL,
                result.confidence,
                result.method,
            ),
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()


# ── Method A: Wikidata Search API（人物・組織向け） ───────────────────────────

_WIKIDATA_SEARCH = (
    "https://www.wikidata.org/w/api.php"
    "?action=wbsearchentities&search={query}&language=en&type=item"
    "&limit=3&format=json"
)
_WIKIDATA_SITELINK = (
    "https://www.wikidata.org/w/api.php"
    "?action=wbgetentities&ids={qid}&props=sitelinks/urls&sitefilter=enwiki&format=json"
)


def _link_via_wikidata(query: str, timeout: int = 8) -> Optional[WikiLinkResult]:
    """Wikidata Search API でエンティティを検索して正規名と QID を取得する。"""
    url = _WIKIDATA_SEARCH.format(query=urllib.parse.quote(query))
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "factfull/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode())
        hits = data.get("search", [])
        if not hits:
            return None

        best = hits[0]
        qid = best.get("id", "")
        label = best.get("label", "")
        description = best.get("description", "")
        conf = _title_confidence(query, label)

        # Wikipedia URL を取得
        wiki_url = ""
        try:
            sl_url = _WIKIDATA_SITELINK.format(qid=qid)
            req2 = urllib.request.Request(sl_url, headers={"User-Agent": "factfull/1.0"})
            with urllib.request.urlopen(req2, timeout=timeout) as resp2:
                sl_data = json.loads(resp2.read().decode())
            enwiki = (
                sl_data.get("entities", {}).get(qid, {})
                .get("sitelinks", {}).get("enwiki", {})
            )
            wiki_url = enwiki.get("url", "")
        except Exception:
            pass

        return WikiLinkResult(
            query=query,
            canonical_name=label,
            wikipedia_url=wiki_url,
            wikidata_qid=qid,
            confidence=conf,
            found=True,
            method="wikidata",
        )
    except Exception:
        return None


# ── Method B: localsearch-mcp（概念向け） ────────────────────────────────────

_LOCALSEARCH_PATH = Path(__file__).parents[3] / "localsearch-mcp"

_SEARCH_SCRIPT = """\
import sys, json
sys.path.insert(0, "src")
from indexer import WikiIndexer
idx = WikiIndexer()
idx.load_or_build()
results = idx.hybrid_search(sys.argv[1], top_k=int(sys.argv[2]))
print(json.dumps(results))
"""


def _link_via_local_wiki(
    query: str, top_k: int = 3,
    localsearch_path: Path = _LOCALSEARCH_PATH,
) -> Optional[WikiLinkResult]:
    """localsearch-mcp の BM25+vector インデックスで概念を検索する。"""
    script_path = localsearch_path / "_factfull_search.py"
    script_path.write_text(_SEARCH_SCRIPT, encoding="utf-8")
    try:
        result = subprocess.run(
            ["uv", "run", "python", str(script_path), query, str(top_k)],
            cwd=str(localsearch_path),
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode != 0:
            return None
        docs = json.loads(result.stdout.strip())
    except Exception:
        return None
    finally:
        script_path.unlink(missing_ok=True)

    if not docs:
        return None

    chosen = next((d for d in docs if not d.get("is_disambiguation")), docs[0])
    title = chosen.get("title", "")
    url = chosen.get("url", "")
    conf = _title_confidence(query, title)

    if conf < 0.4:
        return None

    return WikiLinkResult(
        query=query,
        canonical_name=title,
        wikipedia_url=url,
        wikidata_qid=None,
        confidence=conf,
        is_disambiguation=chosen.get("is_disambiguation", False),
        found=True,
        method="local_wiki",
    )


# ── WikiLinker 本体 ───────────────────────────────────────────────────────────

_PERSON_ORG_TYPES = {"person", "organization"}
_CONCEPT_TYPES = {"concept", "framework", "theory"}


class WikiLinker:
    """
    エンティティ型に応じてリンキング方法を切り替える。
    - person / organization → Wikidata Search API
    - concept / framework → localsearch-mcp ローカル検索
    """

    def __init__(
        self,
        localsearch_path: Path = _LOCALSEARCH_PATH,
        min_confidence: float = 0.6,
        cache_path: Path = _DEFAULT_CACHE,
    ) -> None:
        self.min_confidence = min_confidence
        self._cache = _LinkCache(cache_path)
        self._lspath = Path(localsearch_path)

    def link(self, entity_name: str, entity_type: str = "person") -> WikiLinkResult:
        # キャッシュ確認
        cached = self._cache.get(entity_name)
        if cached is not None:
            return cached

        result: Optional[WikiLinkResult] = None

        if entity_type in _PERSON_ORG_TYPES:
            result = _link_via_wikidata(entity_name)
        elif entity_type in _CONCEPT_TYPES:
            result = _link_via_local_wiki(entity_name, localsearch_path=self._lspath)

        if result is None or result.confidence < self.min_confidence:
            out = _not_found(entity_name)
        else:
            out = result

        self._cache.set(out)
        return out

    def close(self) -> None:
        self._cache.close()

    def __enter__(self) -> WikiLinker:
        return self

    def __exit__(self, *_) -> None:
        self.close()
