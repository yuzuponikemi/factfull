"""Fact-check primitives exposed as test-smith Tools.

These wrappers let the test-smith ReAct agent call factfull's existing
single-shot LLM primitives (``factfull.claim_extractor.extract``,
``factfull.verifier.verify``) as named tools.

Layer rule: the ReAct loop lives in test-smith. factfull stays a
provider of primitives. These tool classes are the bridge — they expose
factfull's functions through the test-smith Tool interface without
relocating any of the underlying logic.

BM25 evidence retrieval is intentionally not exposed here yet. It is
stateful (requires a pre-built index over truth sources) and the right
shape — passing the index at construction time vs lazily loading from
disk vs querying a remote MCP server — is still under discussion. A
``Bm25EvidenceSearchTool`` will land in a follow-up once that shape is
settled.
"""

from __future__ import annotations

import json
from typing import Any

from test_smith.agents.react import Tool, ToolError, ToolResult

from factfull import claim_extractor, verifier
from factfull.ingest.chunker import Chunk


class ExtractClaimsTool(Tool):
    """Wraps ``factfull.claim_extractor.extract``.

    The agent passes a document body; the tool returns a JSON array of
    atomic claims that are candidates for verification.
    """

    def __init__(self, default_max_claims: int = 30):
        self._default_max_claims = default_max_claims

    @property
    def name(self) -> str:
        return "extract_claims"

    @property
    def description(self) -> str:
        return (
            "Extract atomic, fact-checkable claims from a document. Each "
            "claim is a single sentence that can be verified against "
            "evidence. Opinions and speculation are filtered out."
        )

    @property
    def input_schema(self) -> str:
        return (
            '{"document": "<full document text>", '
            f'"max_claims": {self._default_max_claims}}}'
        )

    def run(self, args: dict[str, Any]) -> ToolResult:
        document = (args.get("document") or "").strip()
        if not document:
            raise ToolError("extract_claims requires a non-empty 'document'.")
        max_claims = int(args.get("max_claims") or self._default_max_claims)

        claims = claim_extractor.extract(document, max_claims=max_claims)
        observation = json.dumps(
            {"count": len(claims), "claims": claims},
            ensure_ascii=False,
            indent=2,
        )
        return ToolResult(observation=observation)


class VerifyClaimTool(Tool):
    """Wraps ``factfull.verifier.verify``.

    The agent passes a single claim plus a list of evidence passages it
    has already gathered (via web_fetch, BM25 search, etc.). The tool
    returns the verdict (``supported`` / ``contradicted`` / ``partial``
    / ``unverifiable``) and a short reason.
    """

    @property
    def name(self) -> str:
        return "verify_claim"

    @property
    def description(self) -> str:
        return (
            "Verify a single claim against a provided list of evidence "
            "passages. Returns a verdict (supported / contradicted / "
            "partial / unverifiable) and a one-sentence reason. Evidence "
            "must be gathered first — this tool does not retrieve sources."
        )

    @property
    def input_schema(self) -> str:
        return (
            '{"claim": "<a single factual claim>", '
            '"evidence": ["<passage 1>", "<passage 2>", ...], '
            '"sources": ["<optional source label per passage>"]}'
        )

    def run(self, args: dict[str, Any]) -> ToolResult:
        claim = (args.get("claim") or "").strip()
        if not claim:
            raise ToolError("verify_claim requires a non-empty 'claim'.")

        evidence_raw = args.get("evidence") or []
        if not isinstance(evidence_raw, list) or not evidence_raw:
            raise ToolError(
                "verify_claim requires at least one evidence passage in 'evidence'."
            )

        sources_raw = args.get("sources") or []
        if not isinstance(sources_raw, list):
            sources_raw = []

        chunks: list[Chunk] = []
        for i, passage in enumerate(evidence_raw):
            text = str(passage).strip()
            if not text:
                continue
            source = (
                str(sources_raw[i]).strip()
                if i < len(sources_raw) and sources_raw[i]
                else f"evidence-{i + 1}"
            )
            chunks.append(Chunk(text=text, source=source, offset=0))

        if not chunks:
            raise ToolError("All provided evidence passages were empty.")

        result = verifier.verify(claim, chunks)
        observation = json.dumps(
            {
                "verdict": result.verdict.value,
                "reason": result.reason,
                "evidence_count": len(result.evidence),
            },
            ensure_ascii=False,
            indent=2,
        )
        return ToolResult(observation=observation)


__all__ = ["ExtractClaimsTool", "VerifyClaimTool"]
