"""
LLM バックエンド抽象化。
FACTFULL_LLM_BACKEND 環境変数で切り替え可能:
  - ollama (デフォルト): ローカル Ollama
  - anthropic: Anthropic API
"""
from __future__ import annotations
import json
import os
import urllib.request


BACKEND = os.environ.get("FACTFULL_LLM_BACKEND", "ollama")
OLLAMA_URL = os.environ.get("FACTFULL_OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.environ.get("FACTFULL_OLLAMA_MODEL", "glm-4.7-flash:latest")
ANTHROPIC_MODEL = os.environ.get("FACTFULL_ANTHROPIC_MODEL", "claude-sonnet-4-6")


def call(prompt: str, num_ctx: int = 8192, timeout: int = 600) -> str:
    if BACKEND == "anthropic":
        return _call_anthropic(prompt)
    return _call_ollama(prompt, num_ctx=num_ctx, timeout=timeout)


def _call_ollama(prompt: str, num_ctx: int = 8192, timeout: int = 600) -> str:
    payload = json.dumps({
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": True,
        "options": {"temperature": 0.1, "num_ctx": num_ctx},
    }).encode("utf-8")

    req = urllib.request.Request(
        OLLAMA_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    chunks: list[str] = []
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for line in resp:
            if not line.strip():
                continue
            data = json.loads(line.decode("utf-8"))
            chunks.append(data.get("response", ""))
            if data.get("done"):
                break
    return "".join(chunks).strip()


def _call_anthropic(prompt: str) -> str:
    import anthropic  # type: ignore
    client = anthropic.Anthropic()
    message = client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text.strip()
