## GBrain Configuration (configured by /setup-gbrain)
- Mode: local-stdio
- Engine: pglite
- Config file: ~/.gbrain/config.json (mode 0600)
- Setup date: 2026-05-13
- MCP registered: yes (user scope, /Users/ikmx/.bun/bin/gbrain serve)
- Artifacts sync: full → https://github.com/yuzuponikemi/gstack-artifacts-ikmx
- Current repo policy: read-write (github.com/yuzuponikemi/factfull)
- Transcript ingest: incremental

## GBrain Search Guidance (configured by /sync-gbrain)
<!-- gstack-gbrain-search-guidance:start -->

GBrain is set up and synced on this machine. The agent should prefer gbrain
over Grep when the question is semantic or when you don't know the exact
identifier yet.

**This worktree is pinned to a worktree-scoped code source** via the
`.gbrain-source` file in the repo root. Source ID: gstack-code-full-d0907b79-4d43f9.

Currently indexed: 71 pages (65 Claude Code session transcripts across all repos,
5 concepts, 1 timeline entry). Keyword search (tsvector) is active.
Semantic/vector search requires OPENAI_API_KEY env var.
Code-symbol search (code-def, code-refs) requires gbrain v0.20.0+ (not yet released).

Prefer gbrain when:
- "What did we do in this repo last session?" / cross-session history:
    `gbrain search "<terms>"` (keyword)
- "What was discussed/decided about X?" / past transcript search:
    `gbrain search "<terms>"` or `gbrain query "<question>"` (if OPENAI_API_KEY set)

Grep is still right for code symbol lookup, regex, multiline patterns, and
file globs. Run `/sync-gbrain` to refresh transcripts after new sessions.

<!-- gstack-gbrain-search-guidance:end -->

## Layer architecture (workspace position)

factfull sits in the **Knowledge Layer**. Multi-step research /
verification loops live in the **Agent Layer** (test-smith). The
workspace-level CLAUDE.md (`../CLAUDE.md`) holds the canonical stack
diagram; the dependency direction is one-way:

    factfull (Knowledge)  →  test-smith (Agent)  →  infra tools

factfull imports `test_smith.agents.react.ReActAgent` when it needs an
autonomous loop. test-smith never imports factfull. To let an agent use
a factfull primitive, expose it as a `Tool` subclass under
`factfull/tools/` (see existing wrappers there).

## LLM call policy — when to use what

| Case | Use | Example |
| --- | --- | --- |
| Single-shot, fixed prompt | `factfull/llm.py` `call()` | claim extraction, summarisation, metadata, KG triple extraction |
| Multi-step judgement / external source discovery | `from test_smith.agents.react import ReActAgent` | book chapter resolution, deep fact-check across multiple sources |
| factfull primitive callable by an agent | `factfull/tools/` (subclass `test_smith.agents.react.Tool`) | `ExtractClaimsTool`, `VerifyClaimTool`, `SubmitChaptersTool` |

**Rule of thumb**: if you are about to write a loop where an LLM decides
*what to do next*, stop and use the ReAct agent from test-smith. If the
loop is "fixed steps; call LLM at step N", keep it in factfull.

Reviewer checklist when touching factfull:

1. Is this a multi-step LLM-driven judgement loop? → belongs in test-smith
2. Is this a fixed pipeline that calls an LLM once or N times in order? → fine in factfull
3. Is this a one-shot LLM call? → fine in factfull
4. Do you need an agent to drive a factfull primitive? → wrap as a Tool in `factfull/tools/`, do **not** write a ReAct loop in factfull

## Local Ollama model availability

test-smith's `QualityProfile` presets reference models (`command-r`,
`qwen3-next`, `llama3`) that are not currently pulled on this machine.
Currently installed Ollama models on this machine:

- `gemma4:26b`, `gemma4:e4b`, `gemma4:latest` (multilingual; **but
  overly safety-tuned — refuses academic content that discusses the
  body or sexuality, and the scratchpad makes the refusal contagious
  for the rest of the run**)
- `qwen3.6:35b-a3b` (multilingual, less safety-tuned; preferred for
  philosophy / sociology books)
- `glm-4.7-flash:latest` (fast variant)
- `translategemma:12b` (translation-specialised)
- Embedding models: `qwen3-embedding:0.6b`, `nomic-embed-8k`,
  `nomic-embed-text`

For the ReAct chapter resolver, the working baseline is:

    ReActChapterResolver(ollama_model="qwen3.6:35b-a3b", ollama_num_ctx=32768)

Avoid `quality_profile="gemma4"` for any book that may touch the body /
sexuality (most modern philosophy and sociology).

## 長時間パイプライン

- パイプライン出力を `head` や `tail` でパイプしない（SIGPIPE でプロセスが死ぬ）
- 実行は `python -m pipeline.cli run ... 2>&1 | tee logs/run-$(date +%s).log` の形を使う
- Ollama の URL は `localhost`。Docker コンテナ内でない限り `host.docker.internal` は使わない
- バッチスクリプト実行前に `which python` で正しい venv を確認する
