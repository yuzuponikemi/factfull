"""SubmitChaptersTool: terminal tool the ReAct agent calls when it has
finished resolving a book's table of contents.

The tool's input is a structured chapter list, so the agent's output
lands directly in a ChapterStructure — no markdown parsing step.
"""

from __future__ import annotations

from typing import Any

from test_smith.agents.react import Tool, ToolError, ToolResult

from factfull.book.chapter_resolver import ChapterEntry, ChapterStructure


class SubmitChaptersTool(Tool):
    """Agent-facing terminal tool that yields a ChapterStructure.

    The tool is parameterised with the book metadata (title, author) so
    the structure can be assembled without the agent having to repeat
    that information back to us.
    """

    def __init__(self, book_title: str, author: str):
        self._book_title = book_title
        self._author = author
        # Populated when the agent calls this tool. The caller reads it
        # back after agent.run() to get the typed ChapterStructure.
        self.result: ChapterStructure | None = None

    @property
    def name(self) -> str:
        return "submit_chapters"

    @property
    def is_terminal_intent(self) -> bool:
        return True

    @property
    def description(self) -> str:
        return (
            "Submit the final, verified table of contents for the book "
            "and stop the loop. Use this once you have cross-checked the "
            "chapter list against at least one primary source (publisher "
            "page, library catalogue, original-language Wikipedia, or "
            "the book's own GitHub repo for open-source books)."
        )

    @property
    def input_schema(self) -> str:
        return (
            '{"chapters": [{"num": "1", "title": "...", "part": "(optional)"}, ...], '
            '"confidence": 0.0-1.0, '
            '"sources": ["url1", "url2", ...], '
            '"notes": "(optional, e.g. edition differences, missing chapters)"}'
        )

    def run(self, args: dict[str, Any]) -> ToolResult:
        chapters_raw = args.get("chapters")
        if not isinstance(chapters_raw, list) or not chapters_raw:
            raise ToolError(
                "submit_chapters requires a non-empty 'chapters' list."
            )

        chapters: list[ChapterEntry] = []
        for i, item in enumerate(chapters_raw, 1):
            if not isinstance(item, dict):
                raise ToolError(
                    f"chapters[{i - 1}] must be an object with title/num/part keys."
                )
            title = (item.get("title") or "").strip()
            if not title:
                raise ToolError(f"chapters[{i - 1}] is missing 'title'.")
            num = str(item.get("num") or i).strip()
            part = (item.get("part") or "").strip()
            chapters.append(ChapterEntry(title=title, num=num, part=part))

        confidence = float(args.get("confidence") or 0.0)
        confidence = max(0.0, min(1.0, confidence))

        sources_raw = args.get("sources") or []
        sources = [str(s).strip() for s in sources_raw if str(s).strip()]
        # Distinct domain count is a rough proxy for "independent agreement".
        sources_agreed = len({_domain_of(s) for s in sources})

        notes = (args.get("notes") or "").strip()

        structure = ChapterStructure(
            book_title=self._book_title,
            author=self._author,
            chapters=chapters,
            confidence=confidence,
            sources_consulted=sources,
            sources_agreed=sources_agreed,
            notes=notes,
        )
        self.result = structure
        return ToolResult(
            observation=(
                f"Submitted {len(chapters)} chapters "
                f"(confidence={confidence:.2f}, sources_agreed={sources_agreed})."
            ),
            is_terminal=True,
            final_payload=structure,
        )


def _domain_of(url: str) -> str:
    """Cheap domain extractor — enough for distinct-source counting."""
    if "://" not in url:
        return url
    host = url.split("://", 1)[1].split("/", 1)[0]
    # Strip leading "www."
    return host.removeprefix("www.").lower()
