"""
factfull/editorial.py
======================
ファクトチェック済みの記事に「編集後記」を生成・追加する。

パイプライン上の位置:
  generate_summary() → [factcheck loop] → generate_editorial_note() → publish

編集後記はトランスクリプトの事実に縛られない自由な考察ゾーン。
事実ベースのレポートが確定した「後」に書くことで、
ファクトチェックの対象から完全に分離できる。
"""
from __future__ import annotations
from . import llm

_PROMPT = """\
あなたはポッドキャスト記事の編集後記を書くエディター AI です。

以下の「ファクトチェック済み記事」を読んだうえで、400〜600字の編集後記を書いてください。

## 編集後記とは
- 事実の要約ではない。記事本文に書いたことを繰り返さない
- このポッドキャストを「読んで」何を思ったか、何が引っかかったか
- 発言者の主張の中で面白いと感じた論理・前提・盲点
- この対話が問いかけていること、まだ答えが出ていないこと
- 読者が自分の考察を書くための「踏み台」になるような問いかけで締める

## 書き方のルール
- 自分の解釈・感想を率直に書く（「〜ではないか」「〜が気になる」など）
- 見出しなしの連続した文章（箇条書き不可）
- 400〜600字

## ファクトチェック済み記事
{article}
"""


def generate_editorial_note(article: str) -> str:
    """
    検証済み記事を読んで編集後記を生成し、文字列で返す。

    Args:
        article: ファクトチェック・修正済みの summary_ja.md 全文

    Returns:
        "## 編集後記\n\n..." 形式の Markdown 文字列
    """
    # 記事が長すぎる場合は末尾を切って概要＋論点部分だけ渡す
    article_for_prompt = article[:12000] if len(article) > 12000 else article

    prompt = _PROMPT.format(article=article_for_prompt)
    note = llm.call(prompt, num_ctx=16384).strip()

    return f"\n\n## 編集後記\n\n{note}\n"


def append_editorial_note(document: str) -> str:
    """
    既存の document に編集後記セクションがなければ生成して末尾に追加する。
    既にある場合はそのまま返す。

    Returns:
        編集後記が追加された document
    """
    if "## 編集後記" in document:
        print("  [editorial] 編集後記は既に存在します。スキップ。", flush=True)
        return document

    print("  [editorial] 編集後記を生成中...", flush=True)
    note = generate_editorial_note(document)
    print(f"  [editorial] 完了: {len(note)}字", flush=True)
    return document.rstrip() + note
