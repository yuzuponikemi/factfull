"""
factfull/book/pipeline.py
==========================
book-research の run_dir を受け取り、book_guide.md をファクトチェックして
ファクトチェック済み book_guide.md を返す。

フロー:
  1. run_dir から concept_graph.json を読み込んで書籍メタデータを取得
  2. 真実ソースを構築:
     - Route A (01_chunks.json あり): チャンクテキストを book_source.txt に展開
     - Route B (web_researcher): concept_graph の concepts/relations を book_source.txt に展開
  3. book_guide.md に対してファクトチェックループを実行
  4. BookPipelineResult を返す
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


# ── 結果 ─────────────────────────────────────────────────────────────────────

@dataclass
class BookPipelineResult:
    run_dir: Path
    book_id: str
    book_title: str
    author: str
    subject: str
    book_guide_path: Path
    score: float
    source_mode: str   # "gutenberg" | "web_researcher" | "unknown"


# ── 設定 ─────────────────────────────────────────────────────────────────────

@dataclass
class BookPipelineConfig:
    factcheck_model: str = "gemma4:e4b"
    threshold: float = 95.0
    max_iter: int = 5
    max_claims: int = 50
    top_k: int = 5
    critique: bool = False
    editorial: bool = False
    editorial_model: str | None = None


# ── エントリポイント ───────────────────────────────────────────────────────────

def run_book_pipeline(config: BookPipelineConfig, run_dir: Path) -> BookPipelineResult:
    """
    book-research の run_dir を受け取り、book_guide.md をファクトチェックして
    BookPipelineResult を返す。

    Args:
        config: BookPipelineConfig
        run_dir: book-research の run_YYYYMMDD_HHMMSS ディレクトリ

    Returns:
        BookPipelineResult
    """
    run_dir = Path(run_dir)
    book_guide_path = run_dir / "book_guide.md"
    if not book_guide_path.exists():
        raise FileNotFoundError(f"book_guide.md が見つかりません: {book_guide_path}")

    # メタデータ取得
    meta = _parse_metadata(run_dir)

    print(f"\n📚 書籍: {meta['author']} 『{meta['book_title']}』")
    print(f"   ソースモード: {meta['source_mode']}")
    print(f"   ガイドサイズ: {len(book_guide_path.read_text(encoding='utf-8')):,} 文字")

    # 真実ソース構築
    truth_path = _build_truth_source(run_dir, meta["source_mode"])

    # ファクトチェックループ
    from factfull.podcast.steps.factcheck import run_factcheck_on_file

    score = run_factcheck_on_file(config, book_guide_path, truth_path, run_dir)

    return BookPipelineResult(
        run_dir=run_dir,
        book_id=meta["book_id"],
        book_title=meta["book_title"],
        author=meta["author"],
        subject=meta["subject"],
        book_guide_path=book_guide_path,
        score=score,
        source_mode=meta["source_mode"],
    )


# ── 内部ヘルパー ──────────────────────────────────────────────────────────────

def _parse_metadata(run_dir: Path) -> dict:
    """concept_graph.json と events.log から書籍メタデータを抽出する。"""
    meta = {
        "book_id": "unknown",
        "book_title": "Unknown",
        "author": "Unknown",
        "subject": "",
        "source_mode": "unknown",
    }

    # events.log から book_id と source_mode を取得
    events_path = run_dir / "events.log"
    if events_path.exists():
        for line in events_path.read_text(encoding="utf-8").splitlines():
            if "run started" in line and "book=" in line:
                import re
                m = re.search(r"book=(\S+)", line)
                if m:
                    meta["book_id"] = m.group(1)
            if "web_research" in line and "→ node:" in line:
                meta["source_mode"] = "web_researcher"
            if "ingest" in line and "→ node:" in line and meta["source_mode"] == "unknown":
                meta["source_mode"] = "gutenberg"

    # concept_graph.json から詳細情報を取得
    cg_path = run_dir / "03_concept_graph.json"
    if cg_path.exists():
        cg = json.loads(cg_path.read_text(encoding="utf-8"))
        meta["subject"] = cg.get("subject", "")
        if meta["source_mode"] == "unknown":
            meta["source_mode"] = cg.get("source_mode", "unknown")

        # subject から著者・タイトルを推定
        author, title = _extract_author_title(meta["book_id"], meta["subject"])
        meta["author"] = author
        meta["book_title"] = title

    return meta


def _extract_author_title(book_id: str, subject: str) -> tuple[str, str]:
    """book_id と subject から著者・タイトルを推定する。"""
    known = {
        "plato_republic": ("Plato", "国家"),
        "nietzsche_zarathustra": ("Nietzsche", "ツァラトゥストラはこう言った"),
        "spinoza_ethics": ("Spinoza", "エチカ"),
        "descartes_discourse": ("Descartes", "方法序説"),
        "arendt_human_condition": ("Arendt", "人間の条件"),
        "nishida_zen_no_kenkyu": ("西田幾多郎", "善の研究"),
        "watsuji_fudo": ("和辻哲郎", "風土"),
        "wittgenstein_investigations": ("Wittgenstein", "哲学探究"),
        "nihon_no_shiso": ("丸山眞男", "日本の思想"),
        "namerakana_shakai": ("鈴木健", "なめらかな社会とその敵"),
        "benkyo_no_tetsugaku": ("千葉雅也", "勉強の哲学 来たるべきバカのために"),
        "kaiwa_0_2_byou": ("水野", "会話の0.2秒を言語学する"),
        "kaze_no_tani": ("安宅和人", "「風の谷」という希望"),
        "plurality": ("Tang / Weyl", "⿻数位（Plurality）"),
    }
    if book_id in known:
        return known[book_id]

    # フォールバック: subject の最初の文から推定
    if subject:
        first_sentence = subject.split(".")[0].strip()
        return ("Unknown", first_sentence[:40])
    return ("Unknown", book_id.replace("_", " ").title())


def _build_truth_source(run_dir: Path, source_mode: str) -> Path:
    """
    ファクトチェックの真実ソースとなる book_source.txt を構築する。

    - Route A (gutenberg): 01_chunks.json のチャンクテキストを結合
    - Route B (web_researcher): 03_concept_graph.json の概念・関係・アポリアを結合
    """
    truth_path = run_dir / "book_source.txt"
    if truth_path.exists():
        print(f"   [truth] 既存の book_source.txt を使用 ({truth_path.stat().st_size:,} bytes)")
        return truth_path

    chunks_path = run_dir / "01_chunks.json"
    if chunks_path.exists():
        truth_text = _chunks_to_text(chunks_path)
        print(f"   [truth] 01_chunks.json から book_source.txt を生成 ({len(truth_text):,} 文字)")
    else:
        cg_path = run_dir / "03_concept_graph.json"
        if not cg_path.exists():
            raise FileNotFoundError(f"真実ソースが見つかりません: {run_dir}")
        truth_text = _concept_graph_to_text(cg_path)
        print(f"   [truth] 03_concept_graph.json から book_source.txt を生成 ({len(truth_text):,} 文字)")

    truth_path.write_text(truth_text, encoding="utf-8")
    return truth_path


def _chunks_to_text(chunks_path: Path) -> str:
    """01_chunks.json のチャンクテキストを単一テキストに結合する。"""
    data = json.loads(chunks_path.read_text(encoding="utf-8"))
    chunks = data.get("chunks", [])
    parts = []
    for chunk in chunks:
        heading = chunk.get("heading", "")
        text = chunk.get("text", "")
        if heading:
            parts.append(f"## {heading}\n\n{text}")
        else:
            parts.append(text)
    return "\n\n---\n\n".join(parts)


def _concept_graph_to_text(cg_path: Path) -> str:
    """03_concept_graph.json の内容を検索可能なテキストに変換する。"""
    cg = json.loads(cg_path.read_text(encoding="utf-8"))
    parts = []

    subject = cg.get("subject", "")
    if subject:
        parts.append(f"# Overview\n\n{subject}")

    for concept in cg.get("concepts", []):
        name = concept.get("name", "")
        description = concept.get("description", "")
        significance = concept.get("significance", "")
        if name:
            block = f"## Concept: {name}\n\n{description}"
            if significance:
                block += f"\n\nSignificance: {significance}"
            parts.append(block)

    for relation in cg.get("relations", []):
        src = relation.get("source", "")
        tgt = relation.get("target", "")
        rel_type = relation.get("relation_type", "")
        description = relation.get("description", "")
        if src and tgt:
            parts.append(f"## Relation: {src} → {tgt} ({rel_type})\n\n{description}")

    for aporia in cg.get("aporias", []):
        name = aporia.get("name", "")
        description = aporia.get("description", "")
        if name:
            parts.append(f"## Aporia: {name}\n\n{description}")

    logic_flow = cg.get("logic_flow", [])
    if logic_flow:
        flow_text = "\n".join(f"- {step}" for step in logic_flow if isinstance(step, str))
        if flow_text:
            parts.append(f"## Logic Flow\n\n{flow_text}")

    core_frustration = cg.get("core_frustration", "")
    if core_frustration:
        parts.append(f"## Core Frustration\n\n{core_frustration}")

    return "\n\n---\n\n".join(parts)
