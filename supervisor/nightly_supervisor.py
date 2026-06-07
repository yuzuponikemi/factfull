#!/usr/bin/env python3
"""Self-healing nightly pipeline supervisor for factfull.

This is a *wrapper* around the existing ``nightly_pipeline.py`` detection
flow (arXiv + RSS auto-detection → article generation → homupe post). The
supervisor adds three things on top:

  1. Preflight (``health.py``) — refuse to start on a broken environment.
  2. Self-healing — classify subprocess failures and recover automatically
     (Ollama restart/retry, ``--regen`` retry) before giving up.
  3. Publish gate + email — run ``mkdocs build --strict`` before pushing
     homupe, and send SMTP success / escalation mail.

It deliberately does NOT re-implement episode detection: that lives in
``nightly_pipeline.py`` (config/channels.yaml + registry.json). The
supervisor runs the pipeline *without* ``--push`` and performs the
strict-build → commit → push itself, because ``nightly_pipeline.py``'s own
push phase skips the strict build (see CLAUDE.md: push 前に strict 必須).

Usage:
    python supervisor/nightly_supervisor.py              # full nightly run
    python supervisor/nightly_supervisor.py --dry-run    # preflight + detection only
    python supervisor/nightly_supervisor.py --max 1      # cap episodes this run
    python supervisor/nightly_supervisor.py --channel lex_fridman
    python supervisor/nightly_supervisor.py --skip-arxiv

Environment variables:
    TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID — push notification (see notify.py)
    PODCAST_OUTPUT_DIR, OLLAMA_URL, FACTFULL_OLLAMA_MODEL, FACTFULL_LLM_BACKEND
    FACTFULL_DIR — path to factfull repo root (defaults to this script's parent)
    HOMUPE_DIR   — path to homupe repo root (for git push after publish)

Failure classification and recovery:
    ImportError / ModuleNotFound → escalate (check venv)
    JSONDecodeError              → retry whole run with --regen (once)
    Ollama timeout / SIGPIPE     → pkill ollama runner, wait, retry (up to 3×)
    HTTP 401 / auth error        → escalate immediately (cannot auto-fix)
    Rate-limit / usage limit     → stop cleanly, escalate
"""

import argparse
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

SUPERVISOR_DIR = Path(__file__).parent
FACTFULL_DIR = Path(os.environ.get("FACTFULL_DIR", SUPERVISOR_DIR.parent))
LOG_DIR = SUPERVISOR_DIR / "logs"

MAX_OLLAMA_RETRIES = 3

# nightly_pipeline.py runs from the factfull repo root.
PIPELINE_SCRIPT = "nightly_pipeline.py"


def _uv() -> str:
    """Absolute path to uv. PATH is unreliable under launchd (no ~/.local/bin)."""
    return shutil.which("uv") or os.path.expanduser("~/.local/bin/uv")

import logging  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("supervisor")


# ── Failure classification ───────────────────────────────────────────────────

class Escalate(Exception):
    """Raised when a failure cannot be auto-recovered."""


def _classify(combined_log: str) -> str:
    """Return a short failure-type string from the run log."""
    c = combined_log.lower()
    if "401" in c or "unauthorized" in c or "authentication" in c:
        return "auth_error"
    if "rate limit" in c or "usage limit" in c:
        return "rate_limit"
    if "importerror" in c or "modulenotfounderror" in c:
        return "import_error"
    if "jsondecode" in c or "json.decoder" in c:
        return "json_error"
    if "sigpipe" in c or "broken pipe" in c:
        return "sigpipe"
    if "timeout" in c or "connection refused" in c:
        return "ollama_timeout"
    return "unknown"


# ── Ollama recovery ──────────────────────────────────────────────────────────

def _restart_ollama() -> None:
    log.info("Restarting Ollama runner...")
    subprocess.run(["pkill", "-f", "ollama runner"], capture_output=True)
    time.sleep(8)


def _ollama_healthy() -> bool:
    sys.path.insert(0, str(SUPERVISOR_DIR))
    from health import check_ollama
    ok, msg = check_ollama()
    if not ok:
        log.warning("Ollama not healthy: %s", msg)
    return ok


# ── Pipeline runner (wraps nightly_pipeline.py) ──────────────────────────────

def _run_pipeline(
    pipeline_args: list[str],
    log_path: Path,
    regen: bool = False,
    dry_run: bool = False,
) -> None:
    """Run nightly_pipeline.py under self-healing supervision.

    Returns on success; raises Escalate on unrecoverable failure.
    """
    # Run nightly_pipeline.py with the same venv interpreter that runs the
    # supervisor (sys.executable = factfull/.venv/bin/python). This avoids
    # depending on `uv` being on PATH, which it is not under launchd.
    cmd = [sys.executable, PIPELINE_SCRIPT, *pipeline_args]
    if dry_run:
        cmd.append("--dry-run")
    if regen:
        cmd.append("--regen")

    ollama_retries = 0
    while True:
        log.info("Running: %s", " ".join(cmd))
        with open(log_path, "a") as lf:
            lf.write(f"\n=== {datetime.now().isoformat()} :: {' '.join(cmd)} ===\n")
            lf.flush()
            result = subprocess.run(
                cmd,
                cwd=FACTFULL_DIR,
                stdout=lf,
                stderr=subprocess.STDOUT,
                text=True,
            )

        log_text = log_path.read_text(errors="replace")
        tail = "\n".join(log_text.splitlines()[-80:])

        if result.returncode == 0:
            return

        failure = _classify(log_text)
        log.warning("Pipeline failed: %s (rc=%d)", failure, result.returncode)

        if failure == "auth_error":
            raise Escalate(f"Auth error — re-login required.\n\n{tail}")
        if failure == "rate_limit":
            raise Escalate(f"Rate limit hit — stopping cleanly.\n\n{tail}")
        if failure == "import_error":
            raise Escalate(f"ImportError — check venv.\n\n{tail}")

        if failure == "json_error" and not regen:
            log.info("JSONDecodeError detected — retrying whole run with --regen")
            _run_pipeline(pipeline_args, log_path, regen=True, dry_run=dry_run)
            return

        if failure in ("ollama_timeout", "sigpipe"):
            if ollama_retries >= MAX_OLLAMA_RETRIES:
                raise Escalate(
                    f"Ollama unrecoverable after {MAX_OLLAMA_RETRIES} restarts.\n\n{tail}"
                )
            _restart_ollama()
            if not _ollama_healthy():
                raise Escalate(f"Ollama still down after restart.\n\n{tail}")
            ollama_retries += 1
            log.info("Retrying pipeline (Ollama retry %d)", ollama_retries)
            continue

        raise Escalate(f"Unrecoverable failure ({failure}).\n\n{tail}")


# ── Publish gate (strict build → commit → push) ──────────────────────────────

PUBLISH_PATHS = ["docs/blog/posts/", "docs/data/kg/", "docs/data/synthesis/"]


def _homupe_dir() -> Path:
    return Path(os.environ.get("HOMUPE_DIR", FACTFULL_DIR.parent / "homupe"))


def _changed_posts(homupe_dir: Path) -> list[str]:
    """Filenames of new/modified blog posts in the homupe working tree."""
    out = subprocess.run(
        ["git", "status", "--porcelain", "docs/blog/posts/"],
        cwd=homupe_dir,
        capture_output=True,
        text=True,
    ).stdout
    posts = []
    for line in out.splitlines():
        path = line[3:].strip()
        if path.endswith(".md"):
            posts.append(Path(path).name)
    return posts


def _git(homupe_dir: Path, *args: str) -> subprocess.CompletedProcess:
    """Run a git command in homupe, raising Escalate on failure (clean notify)."""
    r = subprocess.run(["git", *args], cwd=homupe_dir, capture_output=True, text=True)
    if r.returncode != 0:
        raise Escalate(
            f"git {' '.join(args)} failed:\n{(r.stdout + r.stderr).strip()[-800:]}"
        )
    return r


def _publish(homupe_dir: Path) -> list[str]:
    """Strict-build homupe, then commit+push to main. Returns published posts.

    Raises Escalate (→ notify) on strict-build failure or any git failure, so
    a problem surfaces as a notification instead of an uncaught traceback.
    """
    posts = _changed_posts(homupe_dir)
    # untracked new posts also show here (git status --porcelain lists `??`)
    dirty = subprocess.run(
        ["git", "status", "--porcelain", *PUBLISH_PATHS],
        cwd=homupe_dir,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if not dirty:
        log.info("No homupe changes to publish")
        return []

    # Nightly posts belong on main. New post files are untracked, so switching
    # branches carries them along. If a tracked file would conflict, _git
    # escalates rather than committing to the wrong branch (which strands the
    # post — exactly the failure this guards against).
    current = _git(homupe_dir, "rev-parse", "--abbrev-ref", "HEAD").stdout.strip()
    if current != "main":
        log.info("homupe is on %r — switching to main to publish", current)
        _git(homupe_dir, "checkout", "main")

    log.info("Strict mkdocs build before push...")
    build = subprocess.run(
        [_uv(), "run", "mkdocs", "build", "--strict"],
        cwd=homupe_dir,
        capture_output=True,
        text=True,
    )
    if build.returncode != 0:
        raise Escalate(
            "mkdocs build --strict failed — NOT pushing.\n\n"
            + "\n".join((build.stdout + build.stderr).splitlines()[-40:])
        )

    label = f"{len(posts)} 件" if posts else "更新"
    _git(homupe_dir, "add", *PUBLISH_PATHS)
    _git(homupe_dir, "commit", "-m", f"post: nightly pipeline — {label}")
    # Integrate remote first so a moved origin/main doesn't reject the push.
    _git(homupe_dir, "pull", "--rebase", "origin", "main")
    _git(homupe_dir, "push", "origin", "main")
    log.info("Pushed homupe (%s)", label)
    return posts


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Self-healing supervisor wrapping nightly_pipeline.py"
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Preflight + detection only (no processing, no push)")
    parser.add_argument("--max", type=int, default=None,
                        help="Cap episodes processed this run (passed to nightly_pipeline)")
    parser.add_argument("--channel", default=None,
                        help="Restrict to one channel id (passed to nightly_pipeline)")
    parser.add_argument("--skip-arxiv", action="store_true",
                        help="Skip the arXiv phase (passed to nightly_pipeline)")
    parser.add_argument("--regen", action="store_true",
                        help="Reuse cached podcast output (skip re-download/translate)")
    args = parser.parse_args()

    sys.path.insert(0, str(SUPERVISOR_DIR))
    from health import run_all as preflight
    from notify import notify_success, notify_escalation

    log.info("=== Nightly pipeline supervisor starting ===")
    log.info("Preflight checks...")
    if not preflight():
        try:
            notify_escalation("Preflight failed — pipeline not started.", "See logs.")
        except Exception as exc:  # mail itself may be unconfigured
            log.error("(notify failed: %s)", exc)
        sys.exit(1)
    log.info("Preflight OK")

    pipeline_args: list[str] = []
    if args.max is not None:
        pipeline_args += ["--max", str(args.max)]
    if args.channel:
        pipeline_args += ["--channel", args.channel]
    if args.skip_arxiv:
        pipeline_args.append("--skip-arxiv")

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"nightly-{datetime.now().strftime('%Y%m%d-%H%M%S')}.log"
    start_time = time.time()

    try:
        _run_pipeline(pipeline_args, log_path, regen=args.regen, dry_run=args.dry_run)
    except Escalate as exc:
        log.error("⚠️  Escalating: %s", exc)
        tail = "\n".join(log_path.read_text(errors="replace").splitlines()[-80:])
        try:
            notify_escalation(str(exc).splitlines()[0][:200], tail)
        except Exception as mexc:
            log.error("(notify failed: %s)", mexc)
        sys.exit(1)

    if args.dry_run:
        log.info("[dry-run] Preflight + detection passed. Exiting before publish.")
        return

    elapsed = int(time.time() - start_time)
    try:
        published = _publish(_homupe_dir())
    except Escalate as exc:
        log.error("⚠️  Publish blocked: %s", exc)
        try:
            notify_escalation(str(exc).splitlines()[0][:200], str(exc)[-1500:])
        except Exception as mexc:
            log.error("(notify failed: %s)", mexc)
        sys.exit(1)

    if published:
        try:
            notify_success(published, elapsed)
        except Exception as mexc:
            log.error("(notify failed: %s)", mexc)
        log.info("=== Done: %d published in %ds ===", len(published), elapsed)
    else:
        log.info("=== Done: no new posts (%ds) ===", elapsed)


if __name__ == "__main__":
    main()
