"""
factfull/process/summarizer.py
================================
SourceDoc → 日本語要約記事（Map-Reduce 2パス）

podcast/archiver.py の汎用版。タイムスタンプ・発言者などの podcast 固有処理を除き、
論文・書籍・Web 記事にも使える形に整理した。

パス構成:
  Pass 1 (Map)    : チャンクごとに要点を箇条書き抽出（日本語）
  Pass 2 (Reduce) : 全チャンクの要点を統合して構造化記事を生成
"""
from __future__ import annotations

from factfull.core.types import SourceDoc
from factfull import llm

# ── Pass 1 プロンプト ─────────────────────────────────────────────────────────

_PASS1_PROMPT = """\
以下のテキストを読み、重要な情報を日本語で箇条書きにしてください。

## ルール
- 事実・主張・数値・固有名詞を優先する
- 1点につき1〜2文。冗長にしない
- 5〜10点を目安に抽出する
- 出力は箇条書きのみ（見出し・前置き不要）

## テキスト
{chunk}
"""

# ── Pass 2 プロンプト ─────────────────────────────────────────────────────────

_PASS2_PROMPT = """\
あなたは情報整理の専門家です。
以下の「要点メモ群」はひとつのソース（論文・書籍・記事など）を
チャンクに分割して抽出した要点です。これらを統合して、
日本語の読者向けに構造化された解説記事を書いてください。

## ソース情報
- タイトル: {title}
- 種別: {source_type}

## 要点メモ群
{notes}

## 記事の構成（Markdown で出力）
### 概要
（200字程度でソース全体の主題と結論を要約）

### 主要な論点・発見
（要点メモから重要な論点を3〜7項目にまとめる。各項目に短い説明を添える）

### キーワード
（重要な固有名詞・概念を ` タグで列挙。例: `AI` / `機械学習`）

### 注目すべき点
（特に印象的・意外・重要だと感じた内容を1〜3点、自由に述べる）

## 注意
- 要点メモにない情報を追加しない
- 見出しの番号は不要
- 合計 800〜1500字を目安に
"""


# ── 公開 API ──────────────────────────────────────────────────────────────────

def summarize(
    doc: SourceDoc,
    model: str | None = None,
    max_chunks: int = 20,
) -> str:
    """SourceDoc を日本語要約記事（Markdown）に変換する。

    Args:
        doc: 取り込み済みの SourceDoc
        model: Ollama モデル名（省略時は環境変数）
        max_chunks: Pass 1 で処理するチャンク上限

    Returns:
        Markdown 形式の日本語記事
    """
    chunks = doc.chunks or [doc.text]
    chunks = chunks[:max_chunks]

    # ── Pass 1: Map ──────────────────────────────────────────────────────────
    print(f"  [summarize] Pass 1: {len(chunks)} チャンクを処理中...", flush=True)
    notes: list[str] = []
    for i, chunk in enumerate(chunks, 1):
        print(f"    チャンク {i}/{len(chunks)}", flush=True)
        prompt = _PASS1_PROMPT.format(chunk=chunk[:4000])
        note = llm.call(prompt, num_ctx=8192, model=model).strip()
        if note:
            notes.append(f"[チャンク {i}]\n{note}")

    if not notes:
        return f"# {doc.title}\n\n要点を抽出できませんでした。"

    # ── Pass 2: Reduce ───────────────────────────────────────────────────────
    print(f"  [summarize] Pass 2: 統合記事を生成中...", flush=True)
    notes_text = "\n\n".join(notes)
    # 長すぎる場合は前半優先で切る
    if len(notes_text) > 14000:
        notes_text = notes_text[:14000] + "\n\n[... 以降省略 ...]"

    prompt = _PASS2_PROMPT.format(
        title=doc.title,
        source_type=_source_type_label(doc.source_type),
        notes=notes_text,
    )
    article = llm.call(prompt, num_ctx=16384, model=model).strip()
    print(f"  [summarize] 完了: {len(article)} 字", flush=True)
    return article


def _source_type_label(source_type: str) -> str:
    return {
        "podcast": "ポッドキャスト",
        "paper": "論文",
        "book": "書籍",
        "web": "Web 記事",
    }.get(source_type, source_type)
