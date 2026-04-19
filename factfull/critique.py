"""
factfull/critique.py
======================
ファクトチェック済み記事に「批評的読み」セクションを生成・追加する。

editorial.py（個人的な感想・問いかけ）とは役割が異なり、
こちらは構造化された批評的分析:
  1. 欠けている視点
  2. 日本の文脈で読む
  3. 発言者の前提とバイアス

パイプライン上の位置:
  [factcheck loop] → generate_critique() → generate_editorial_note() → publish
"""
from __future__ import annotations
from . import llm

_PROMPT = """\
あなたは批評的思考を専門とするメディア・アナリストです。
以下の「ファクトチェック済み記事」を読み、3つの観点から批評的分析を行ってください。

---

## ファクトチェック済み記事
{article}

---

## 出力形式（Markdownでそのまま出力）

### 欠けている視点
この対話・発言で「議論されていない」または「軽視されている」観点を2〜3点指摘する。
反論・対立仮説・見落とされたデータ・声なき当事者など。
各点を1〜2文で簡潔に。

### 日本の文脈で読む
この内容を日本の読者が受け取る際に注意すべき文脈差・背景差を具体的に書く。
「米国では〇〇だが日本では△△」のような対比が有効。150〜250字。

### 前提とバイアス
発言者が「自明」としている前提や、立場から来るバイアスを1〜2点指摘する。
批判ではなく、読み解きのための補助線として書く。各点を1〜2文で。

---

注意:
- 見出し（###）はそのまま出力する
- 各セクションの間に空行を1行入れる
- 合計で500〜800字程度
- 事実の要約は書かない。分析のみ
"""


def generate_critique(article: str, model: str | None = None) -> str:
    """
    検証済み記事を読んで批評的分析を生成し、文字列で返す。

    Returns:
        "## 批評的読み\n\n..." 形式の Markdown 文字列
    """
    article_for_prompt = article[:12000] if len(article) > 12000 else article
    prompt = _PROMPT.format(article=article_for_prompt)
    body = llm.call(prompt, num_ctx=16384, model=model).strip()
    return f"\n\n## 批評的読み\n\n{body}\n"


def append_critique(document: str, model: str | None = None) -> str:
    """
    既存の document に批評セクションがなければ生成して末尾に追加する。
    既にある場合はそのまま返す。
    """
    if "## 批評的読み" in document:
        print("  [critique] 批評セクションは既に存在します。スキップ。", flush=True)
        return document

    print("  [critique] 批評的読みを生成中...", flush=True)
    section = generate_critique(document, model=model)
    print(f"  [critique] 完了: {len(section)}字", flush=True)
    return document.rstrip() + section
