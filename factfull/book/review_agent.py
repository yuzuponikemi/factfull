"""
factfull/book/review_agent.py
==============================
パイプライン末尾に置くレビューエージェント。
生成された book_guide.md を評価し、公開 / レビュー待ち / やり直し を判定する。
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import requests


@dataclass
class ReviewDecision:
    action: Literal["publish", "review", "redo"]
    reason: str
    concerns: list[str] = field(default_factory=list)

    def is_blocked(self) -> bool:
        return self.action in ("review", "redo")

    def emoji(self) -> str:
        return {"publish": "✅", "review": "⏸️ ", "redo": "❌"}[self.action]


class ReviewAgent:
    """
    LLM を使って book_guide.md の品質を評価し、公開可否を判定する。

    action の意味:
      publish — 品質十分、即公開可能
      review  — 軽微な懸念あり、人間の確認を推奨（--force-publish で上書き可）
      redo    — 重大な問題あり、やり直し必要
    """

    REDO_THRESHOLD = 70.0
    PUBLISH_THRESHOLD = 88.0

    _PROMPT_TEMPLATE = """\
以下のブックガイド記事を評価してください。

書籍: 『{book_title}』  著者: {author}
ファクトチェックスコア: {score:.0f}/100

--- 記事本文（先頭 6000 文字） ---
{guide_excerpt}
--- ここまで ---

次の観点で評価し、JSON のみで回答してください（説明文不要）:
1. 完成度 — 序論・各章解説・結論が揃っているか
2. 内容の深さ — 表面的な要約でなく、論点・思想的意義が書かれているか
3. 誤情報リスク — 明らかな事実誤認や矛盾がないか

回答フォーマット（JSON のみ、コードブロック不要）:
{{"action": "publish" | "review" | "redo", "reason": "判断理由（1〜2文）", "concerns": ["懸念点1", "懸念点2"]}}

判断基準:
- publish: 高品質で即公開可能
- review:  軽微な問題あり、人間の確認推奨
- redo:    重大な問題あり（内容が薄い・誤情報の疑い・未完成など）、やり直し必要"""

    def __init__(
        self,
        model: str = "gemma4:e4b",
        ollama_url: str = "http://localhost:11434",
        redo_threshold: float = REDO_THRESHOLD,
        publish_threshold: float = PUBLISH_THRESHOLD,
    ):
        self.model = model
        self.ollama_url = ollama_url.rstrip("/")
        self.redo_threshold = redo_threshold
        self.publish_threshold = publish_threshold

    def evaluate(self, result) -> ReviewDecision:
        """BookPipelineResult を受け取り ReviewDecision を返す。"""
        # スコア不足は即 redo
        if result.score < self.redo_threshold:
            return ReviewDecision(
                action="redo",
                reason=f"ファクトチェックスコアが基準を下回っています ({result.score:.0f}/100 < {self.redo_threshold:.0f})",
                concerns=["スコア不足 — ファクトチェックのやり直しを推奨"],
            )

        guide_text = Path(result.book_guide_path).read_text(encoding="utf-8")
        guide_excerpt = guide_text[:6000] + ("\n...[省略]..." if len(guide_text) > 6000 else "")

        prompt = self._PROMPT_TEMPLATE.format(
            book_title=result.book_title,
            author=result.author,
            score=result.score,
            guide_excerpt=guide_excerpt,
        )

        try:
            resp = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"num_predict": 500, "temperature": 0.1},
                },
                timeout=120,
            )
            resp.raise_for_status()
            raw = resp.json().get("response", "")
            return self._parse_response(raw)
        except Exception as e:
            # LLM 失敗時はスコアで判定
            return self._score_fallback(result.score, str(e))

    def _parse_response(self, text: str) -> ReviewDecision:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            return ReviewDecision(action="review", reason="LLM レスポンスのパース失敗 — 人間レビュー推奨", concerns=[])
        try:
            data = json.loads(m.group())
            action = data.get("action", "review")
            if action not in ("publish", "review", "redo"):
                action = "review"
            return ReviewDecision(
                action=action,
                reason=data.get("reason", "LLM 評価完了"),
                concerns=data.get("concerns", []),
            )
        except json.JSONDecodeError:
            return ReviewDecision(action="review", reason="LLM レスポンスの JSON パース失敗", concerns=[])

    def _score_fallback(self, score: float, error: str) -> ReviewDecision:
        if score >= self.publish_threshold:
            return ReviewDecision(
                action="publish",
                reason=f"LLM 評価失敗のためスコアで判定 ({score:.0f}/100 ≥ {self.publish_threshold:.0f}) — {error}",
            )
        return ReviewDecision(
            action="review",
            reason=f"LLM 評価失敗のためスコアで判定 ({score:.0f}/100) — 人間レビュー推奨",
            concerns=[f"LLM エラー: {error}"],
        )
