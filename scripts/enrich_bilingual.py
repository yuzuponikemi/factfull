#!/usr/bin/env python3
"""
bilingual.json を「理解しやすく」する付加情報で拡張する
========================================================
既存の対訳 JSON に、homupe の対訳記事で使う以下を LLM 生成して追記する：

  - summary_ja : {"tldr": str, "points": [str, ...]}  記事冒頭の要点サマリー
  - 各 figure/table ブロックの explanation_ja : str     図表の一言解説

翻訳はやり直さない（既存 JSON にフィールドを足すだけ）。固定プロンプトの
単発 LLM 呼び出しなので factfull/llm.py の call() を使う（CLAUDE.md の方針どおり）。

使い方:
    python scripts/enrich_bilingual.py ~/papers/bilingual/1706.03762/bilingual.json
    # 既定モデル: 要約 qwen3.6:35b-a3b / 図表解説 glm-4.7-flash:latest
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from factfull.llm import call

# 要約・図表解説とも非 thinking の gemma4 を使う（qwen3.6 は thinking 型で
# 出力が空になったり形式が崩れたりして不安定だった。gemma4 は形式が安定・高速）。
# ML 論文なので gemma の安全チューニングは無関係。身体・性に触れる人文系を扱う
# 場合は要再検討（gemma は学術的でも拒否することがある）。
SUMMARY_MODEL = "gemma4:latest"
FIGURE_MODEL = "gemma4:e4b"


def _headings(doc: dict) -> list[str]:
    return [
        (b.get("ja") or b.get("en") or "").strip()
        for b in doc.get("blocks", [])
        if b.get("type") == "heading"
    ]


def gen_summary(doc: dict, model: str) -> dict:
    """タイトル・アブストラクト・見出しから要点サマリーを生成する。"""
    title = doc.get("title_ja") or doc.get("title_en") or ""
    abstract = (doc.get("abstract_ja") or "").strip()
    heads = "、".join(h for h in _headings(doc)[:18] if h)
    prompt = (
        "あなたは技術ライターです。次の論文の要点を日本語で簡潔にまとめてください。\n"
        "出力は次の形式を厳守し、前置き・解説・マークダウン装飾は付けないこと：\n\n"
        "TLDR: <この論文を1〜2文で。専門用語は噛み砕く>\n"
        "POINTS:\n"
        "- <重要ポイント1>\n"
        "- <重要ポイント2>\n"
        "- <重要ポイント3>\n"
        "（POINTS は3〜5個。各行は1文・60字程度まで）\n\n"
        f"# タイトル\n{title}\n\n# アブストラクト\n{abstract}\n\n# 章立て\n{heads}\n"
    )
    out = call(prompt, model=model, num_ctx=8192, num_predict=1024).strip()
    return _parse_summary(out)


def _parse_summary(text: str) -> dict:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    tldr = ""
    points: list[str] = []
    for line in text.splitlines():
        s = line.strip()
        m = re.match(r"^TL;?DR\s*[:：]\s*(.+)$", s, re.IGNORECASE)
        if m:
            tldr = m.group(1).strip()
            continue
        m = re.match(r"^[-・*]\s*(.+)$", s)
        if m and not s.upper().startswith("POINTS"):
            pt = m.group(1).strip()
            # 「要点1: …」「ポイント2：…」のような接頭辞を除去
            pt = re.sub(r"^(要点|ポイント|point)\s*\d+\s*[:：.]\s*", "", pt, flags=re.IGNORECASE)
            if pt and pt not in points:
                points.append(pt)
    if not tldr:
        # 形式崩れ時は先頭の非空行を TLDR に
        for line in text.splitlines():
            if line.strip() and not line.strip().upper().startswith(("POINTS", "TLDR")):
                tldr = line.strip()
                break
    return {"tldr": tldr, "points": points[:5]}


def _caption_for(doc: dict, fig: dict) -> dict | None:
    """figure/table ブロックに対応する caption ブロックを返す（label or 近傍）。"""
    blocks = doc.get("blocks", [])
    label = fig.get("label")
    if label:
        for b in blocks:
            if b.get("type") == "caption" and b.get("label") == label:
                return b
    # ラベルが無ければ同ページの最近傍 caption
    i = blocks.index(fig)
    for off in range(1, 5):
        for j in (i + off, i - off):
            if 0 <= j < len(blocks):
                b = blocks[j]
                if b.get("type") == "caption" and b.get("page") == fig.get("page"):
                    return b
    return None


def gen_figure_explanations(doc: dict, model: str) -> int:
    """各 figure/table に explanation_ja を付与。付与した件数を返す。"""
    n = 0
    for b in doc.get("blocks", []):
        if b.get("type") not in ("figure", "table"):
            continue
        cap = _caption_for(doc, b)
        if not cap:
            continue  # 文脈が無いものは無理に作らない
        label = b.get("label") or ("表" if b["type"] == "table" else "図")
        section = " > ".join(b.get("section_path") or [])
        prompt = (
            "次は学術論文の図表のキャプションと文脈です。この図表が何を示しているかを、"
            "日本語で1文（最大60字程度）で説明してください。キャプションに忠実に、"
            "推測や誇張はしないこと。『解説：』などの接頭辞や引用符は付けないこと。\n\n"
            f"ラベル: {label}\n"
            f"キャプション(英): {(cap.get('en') or '').strip()}\n"
            f"キャプション(訳): {(cap.get('ja') or '').strip()}\n"
            f"セクション: {section}\n"
        )
        try:
            exp = call(prompt, model=model, num_ctx=4096, num_predict=512).strip()
        except Exception:
            continue
        exp = re.sub(r"<think>.*?</think>", "", exp, flags=re.DOTALL).strip()
        exp = exp.strip("「」\"' 　").splitlines()[0].strip() if exp else ""
        if exp:
            b["explanation_ja"] = exp
            n += 1
    return n


def main() -> None:
    p = argparse.ArgumentParser(description="bilingual.json を要点・図表解説で拡張")
    p.add_argument("json", type=Path, help="bilingual.json のパス")
    p.add_argument("--summary-model", default=SUMMARY_MODEL)
    p.add_argument("--figure-model", default=FIGURE_MODEL)
    p.add_argument("--skip-figures", action="store_true")
    args = p.parse_args()

    doc = json.loads(args.json.read_text(encoding="utf-8"))
    print(f"📝 要約生成（{args.summary_model}）: {doc.get('title_ja','')[:30]}", flush=True)
    doc["summary_ja"] = gen_summary(doc, args.summary_model)
    print(f"   TLDR: {doc['summary_ja']['tldr'][:50]}… / 要点 {len(doc['summary_ja']['points'])} 個",
          flush=True)

    n_fig = 0
    if not args.skip_figures:
        print(f"🖼  図表解説生成（{args.figure_model}）...", flush=True)
        n_fig = gen_figure_explanations(doc, args.figure_model)

    args.json.write_text(
        json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"✅ 拡張完了: 要約 + 図表解説 {n_fig} 件 → {args.json}", flush=True)


if __name__ == "__main__":
    main()
