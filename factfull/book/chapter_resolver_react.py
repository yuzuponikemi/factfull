"""
factfull/book/chapter_resolver_react.py
========================================
書籍の章立てを test-smith の ReAct エージェントで解決する代替実装。

レイヤー方針:
    factfull (Knowledge Layer) は固定パイプラインとプリミティブを提供し、
    多段判断ループは test-smith (Agent Layer) に委譲する。本モジュールは
    既存の ``factfull.book.chapter_resolver.ChapterResolver`` (ルール
    ベース・multi-source 取得) と同じ出力型 (``ChapterStructure``) を
    返すので、パイプライン側はバックエンド選択フラグで切り替えられる。

依存方向: factfull → test_smith のみ。
"""

from __future__ import annotations

import os

from langchain_ollama.chat_models import ChatOllama

from test_smith.agents.react import (
    FinalAnswerTool,
    ReActAgent,
    WebFetchTool,
    WebSearchTool,
)
from test_smith.models import _get_model_for_role, set_quality_profile

from factfull.book.chapter_resolver import ChapterStructure
from factfull.tools.chapter import SubmitChaptersTool

# Role names from test_smith.src.models (re-imported via the façade would
# duplicate constants; keep them as local literals to avoid a new export).
_ROLE_QUALITY = "quality"

# Direct-model defaults for when the user's local Ollama doesn't ship the
# models in test-smith's QualityProfile presets (command-r, qwen3-next,
# etc.). gemma4 family is fine for most books but is overly safety-tuned
# for academic books that touch the body / sexuality — for those, prefer
# a less-restrictive model like Qwen 3.6.
_DEFAULT_OLLAMA_CTX = 32768


class ReActChapterResolver:
    """ReAct-driven chapter resolver.

    The agent has three tools: ``web_search`` (broad discovery),
    ``web_fetch`` (primary-source reading), and ``submit_chapters``
    (structured terminal). ``submit_chapters`` is what produces the
    typed ``ChapterStructure`` — its result is read back from the
    tool instance after ``agent.run()`` returns.

    Args:
        quality_profile: test-smith QualityProfile name (``"gemma4"``
            recommended for multilingual book research; ``None`` keeps
            whatever MODEL_QUALITY env var is set to).
        max_iterations: hard cap on ReAct loop length.
        temperature: LLM temperature.
        verbose: stream the agent's Thought/Action/Observation log to
            stdout for debugging.
    """

    def __init__(
        self,
        quality_profile: str | None = "gemma4",
        ollama_model: str | None = None,
        ollama_num_ctx: int = _DEFAULT_OLLAMA_CTX,
        max_iterations: int = 14,
        temperature: float = 0.2,
        verbose: bool = True,
    ):
        """Construct a ReAct chapter resolver.

        Args:
            quality_profile: test-smith QualityProfile name. Used when
                ``ollama_model`` is None.
            ollama_model: If set, bypass test-smith profiles entirely and
                instantiate ChatOllama with this exact model name. Useful
                when the local Ollama instance doesn't carry the models
                expected by test-smith's presets (e.g. command-r,
                qwen3-next), or when a specific model is needed to dodge
                a safety-tuned base (gemma4 trips on academic books that
                discuss the body / sexuality).
            ollama_num_ctx: Context window for the direct-model path.
        """
        self._quality_profile = quality_profile
        self._ollama_model = ollama_model
        self._ollama_num_ctx = ollama_num_ctx
        self._max_iterations = max_iterations
        self._temperature = temperature
        self._verbose = verbose

    def resolve(
        self,
        book_title: str,
        author: str,
        extra_urls: list[str] | None = None,
        translate_to: str | None = None,
    ) -> ChapterStructure:
        """Return a ``ChapterStructure`` for the given book.

        Args:
            book_title:   Book title (any language).
            author:       Author name.
            extra_urls:   Hint URLs surfaced to the agent in the prompt
                          (e.g. publisher TOC page, GitHub repo for
                          open-source books).
            translate_to: ISO language code (``"ja"``, ``"en"``,
                          ``"fr"``…). If set, the agent is instructed to
                          render chapter titles in that language while
                          keeping the original-language title visible.
        """
        if self._ollama_model:
            llm = ChatOllama(
                model=self._ollama_model,
                temperature=self._temperature,
                num_ctx=self._ollama_num_ctx,
            )
        else:
            if self._quality_profile:
                set_quality_profile(self._quality_profile)
            llm = _get_model_for_role(_ROLE_QUALITY, temperature=self._temperature)
        submit_tool = SubmitChaptersTool(book_title=book_title, author=author)
        tools = [
            WebSearchTool(),
            WebFetchTool(),
            submit_tool,
            FinalAnswerTool(),  # safety net if the agent can't structure its output
        ]
        agent = ReActAgent(
            llm=llm,
            tools=tools,
            max_iterations=self._max_iterations,
            verbose=self._verbose,
        )

        prompt = self._build_prompt(book_title, author, extra_urls, translate_to)
        agent_result = agent.run(prompt)

        if submit_tool.result is not None:
            return submit_tool.result

        # Fallback: the agent finished via FinalAnswerTool (markdown text)
        # without producing structured output. Return a zero-confidence
        # structure carrying the raw answer in notes so the caller can
        # inspect / retry.
        raw = (agent_result.answer or "").strip()
        return ChapterStructure(
            book_title=book_title,
            author=author,
            chapters=[],
            confidence=0.0,
            sources_consulted=[],
            sources_agreed=0,
            notes=(
                "ReAct agent did not call submit_chapters; "
                f"raw final answer follows.\n---\n{raw}"
            ),
        )

    def _build_prompt(
        self,
        book_title: str,
        author: str,
        extra_urls: list[str] | None,
        translate_to: str | None,
    ) -> str:
        lines: list[str] = []
        lines.append(
            f"書籍『{book_title}』（著者: {author}）の正確な章立て（目次）を調べて、"
            "submit_chapters ツールに渡して終了してください。"
        )
        lines.append("")
        lines.append("【最低限の要件】")
        lines.append("- 何部・何章構成か。")
        lines.append("- 各章の正式タイトル。原語と日本語訳の両方を併記すること。")
        lines.append("- 章番号が飛ぶ場合（1,2,3,5,6...のように）は、欠けている章を")
        lines.append("  必ず web_fetch で原典・出版社ページ・GitHub・図書館目録等から確認すること。")
        lines.append("")
        lines.append("【信頼性ルール】")
        lines.append("- web_search のスニペットだけで結論を出さないこと。")
        lines.append("  必ず少なくとも 1 つの一次資料（出版社ページ、図書館目録、")
        lines.append("  原著の公開リポジトリ、原語版 Wikipedia 等）に web_fetch すること。")
        lines.append("- 信頼度 (confidence) は 0.0〜1.0 の実数で評価し、")
        lines.append("  根拠が弱い場合は素直に低い値を返すこと。")
        lines.append("- sources には実際に参照した URL を全て列挙すること（複数ドメイン推奨）。")
        lines.append("")
        lines.append("【取得が失敗した時の戦略】")
        lines.append("- web_fetch が HTTP 403 / 404 等で失敗した場合、同じ URL を繰り返さず、")
        lines.append("  別のドメイン（出版社、archive.org、academia.edu、")
        lines.append("  monoskop.org の PDF など）を試すこと。")
        lines.append("- **PDF は web_fetch で本文抽出可能**（pypdf 経由）。")
        lines.append("  特に古典哲学・社会学書はオンラインPDF（Monoskop, Gallica, Internet Archive 等）")
        lines.append("  に完全な目次があることが多い。")
        lines.append("- 同じ web_search を2回試して同じ結果なら、別キーワードに切り替えるか、")
        lines.append("  この時点での最善情報で submit_chapters を呼ぶこと。検索ループに陥らないこと。")
        if translate_to:
            lines.append("")
            lines.append(
                f"【出力言語】章タイトルは {translate_to} 言語で記載し、"
                "括弧内に原語タイトルを併記してください。"
            )
        if extra_urls:
            lines.append("")
            lines.append("【ヒント URL】以下は出発点として有用な可能性があります:")
            for url in extra_urls:
                lines.append(f"  - {url}")
        lines.append("")
        lines.append(
            "結果は submit_chapters ツールに渡してください。final_answer は"
            "submit_chapters が使えない場合のフォールバックです。"
        )
        return "\n".join(lines)


__all__ = ["ReActChapterResolver"]


def _cli() -> int:  # pragma: no cover — manual smoke test
    """Tiny CLI: ``uv run python -m factfull.book.chapter_resolver_react <title> <author>``."""
    import argparse

    parser = argparse.ArgumentParser(description="ReAct-backed chapter resolver.")
    parser.add_argument("book_title")
    parser.add_argument("author")
    parser.add_argument("--quality", default="gemma4")
    parser.add_argument(
        "--model",
        default=None,
        help="Override the LLM model (bypass --quality). Example: qwen3.6:35b-a3b",
    )
    parser.add_argument("--num-ctx", type=int, default=_DEFAULT_OLLAMA_CTX)
    parser.add_argument("--max-iterations", type=int, default=14)
    parser.add_argument("--extra-url", action="append", default=[])
    parser.add_argument("--translate-to", default=None)
    args = parser.parse_args()

    # Load test-smith's .env so TAVILY_API_KEY etc. are available.
    try:
        from dotenv import load_dotenv

        test_smith_env = os.path.expanduser("~/source/personal/test-smith/.env")
        if os.path.exists(test_smith_env):
            load_dotenv(test_smith_env)
        load_dotenv()
    except ImportError:
        pass

    resolver = ReActChapterResolver(
        quality_profile=args.quality,
        ollama_model=args.model,
        ollama_num_ctx=args.num_ctx,
        max_iterations=args.max_iterations,
    )
    result = resolver.resolve(
        book_title=args.book_title,
        author=args.author,
        extra_urls=args.extra_url or None,
        translate_to=args.translate_to,
    )
    print()
    print("=" * 70)
    print(f"  Confidence: {result.confidence:.2f}")
    print(f"  Sources agreed: {result.sources_agreed}")
    print(f"  Chapters: {len(result.chapters)}")
    print("=" * 70)
    for ch in result.chapters:
        print(f"  {ch}")
    if result.notes:
        print()
        print("Notes:")
        print(result.notes)
    return 0 if result.is_reliable() else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_cli())
