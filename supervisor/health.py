#!/usr/bin/env python3
"""Preflight health checks for the nightly pipeline supervisor.

Run standalone to verify environment before launching the pipeline:
    python supervisor/health.py
"""

import json
import os
import subprocess
import sys
import urllib.request
import urllib.error


REQUIRED_ENV_VARS = [
    "PODCAST_OUTPUT_DIR",
    "OLLAMA_URL",
    "FACTFULL_LLM_BACKEND",
    "FACTFULL_OLLAMA_MODEL",
]


def check_venv() -> tuple[bool, str]:
    # The interpreter actually running this process is the real signal —
    # PATH's `python3` may differ when the venv binary is invoked directly
    # (e.g. under launchd, which does not "activate" the venv).
    python = sys.executable or ""
    if not python:
        return False, "could not determine running interpreter"
    if ".venv" not in python and "venv" not in python:
        return False, f"not running from a venv: {python}"
    return True, python


def check_ollama(url: str | None = None) -> tuple[bool, str]:
    if url is None:
        url = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
    payload = json.dumps({
        "model": os.environ.get("FACTFULL_OLLAMA_MODEL", "gemma4:e4b"),
        "prompt": "hi",
        "stream": False,
        "options": {"num_ctx": 128, "num_predict": 1},
    }).encode()
    try:
        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=20) as resp:
            body = json.loads(resp.read())
            if body.get("done"):
                return True, url
    except urllib.error.URLError as e:
        return False, f"Ollama unreachable at {url}: {e}"
    except Exception as e:
        return False, f"Ollama check failed: {e}"
    return False, f"Ollama response missing 'done:true' at {url}"


def check_env_vars() -> tuple[bool, str]:
    missing = [v for v in REQUIRED_ENV_VARS if not os.environ.get(v)]
    if missing:
        return False, f"Missing env vars: {', '.join(missing)}"
    return True, "all required env vars set"


def check_podcast_dir() -> tuple[bool, str]:
    d = os.environ.get("PODCAST_OUTPUT_DIR", "")
    if not d:
        return False, "PODCAST_OUTPUT_DIR not set"
    if not os.path.isdir(d):
        return False, f"PODCAST_OUTPUT_DIR does not exist: {d}"
    return True, d


def run_all() -> bool:
    checks = [
        ("venv", check_venv),
        ("env vars", check_env_vars),
        ("podcast dir", check_podcast_dir),
        ("ollama", check_ollama),
    ]
    all_ok = True
    for name, fn in checks:
        ok, detail = fn()
        status = "✅" if ok else "❌"
        print(f"  {status} {name}: {detail}")
        if not ok:
            all_ok = False
    return all_ok


if __name__ == "__main__":
    print("=== Preflight checks ===")
    ok = run_all()
    sys.exit(0 if ok else 1)
