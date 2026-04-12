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


def call(
    prompt: str,
    num_ctx: int = 8192,
    timeout: int = 900,
    max_retries: int = 6,
    retry_wait: int = 45,
) -> str:
    if BACKEND == "anthropic":
        return _call_anthropic(prompt)
    return _call_ollama(
        prompt, num_ctx=num_ctx, timeout=timeout,
        max_retries=max_retries, retry_wait=retry_wait,
    )


def _call_ollama(
    prompt: str,
    num_ctx: int = 8192,
    timeout: int = 900,
    max_retries: int = 6,
    retry_wait: int = 45,
) -> str:
    import time
    payload = json.dumps({
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": True,
        "options": {"temperature": 0.1, "num_ctx": num_ctx},
    }).encode("utf-8")

    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
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
                    try:
                        data = json.loads(line.decode("utf-8"))
                    except json.JSONDecodeError:
                        print(f"  [llm] JSON解析エラー: {line[:100]}", flush=True)
                        continue
                    if "error" in data:
                        raise RuntimeError(f"Ollama エラー: {data['error']}")
                    chunks.append(data.get("response", ""))
                    if data.get("done"):
                        break
            return "".join(chunks).strip()
        except Exception as e:
            import urllib.error as _ue
            print(f"  [llm] エラー: {type(e).__name__}: {e}", flush=True)
            retryable = (TimeoutError, OSError, _ue.HTTPError, _ue.URLError)
            if not isinstance(e, retryable):
                raise
            last_err = e
            if attempt < max_retries:
                print(
                    f"  [llm] タイムアウト (attempt {attempt}/{max_retries})、{retry_wait}秒後にリトライ... [{type(e).__name__}]",
                    flush=True,
                )
                time.sleep(retry_wait)
            else:
                raise RuntimeError(
                    f"Ollama が {max_retries} 回タイムアウトしました: {last_err}"
                ) from last_err
    raise RuntimeError("unreachable")


def _call_anthropic(prompt: str) -> str:
    import anthropic  # type: ignore
    client = anthropic.Anthropic()
    message = client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text.strip()
