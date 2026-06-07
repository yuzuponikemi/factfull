#!/usr/bin/env python3
"""Notification sender for the nightly pipeline supervisor.

Channels (all best-effort; a channel that fails never aborts the run):
  1. Telegram  — push to your phone. Reuses nanoclaw's bot. Primary channel.
  2. Log file  — always append to supervisor/logs/notify.log (durable trail).
  3. macOS     — osascript desktop notification, as a fallback when Telegram
                 is not configured.

Required env vars for Telegram:
    TELEGRAM_BOT_TOKEN   bot token (e.g. nanoclaw's @Nanoikm_bot token)
    TELEGRAM_CHAT_ID     destination chat id (e.g. 8552913958 = your DM)

Test:
    python supervisor/notify.py --test
"""

import json
import os
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path

LOG_FILE = Path(
    os.environ.get("SUPERVISOR_NOTIFY_LOG", Path(__file__).parent / "logs" / "notify.log")
)


def _log(subject: str, body: str) -> None:
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a") as f:
        f.write(f"\n[{ts}] {subject}\n{body}\n")


def _telegram(text: str) -> bool:
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        return False
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    data = urllib.parse.urlencode({"chat_id": chat_id, "text": text[:4000]}).encode()
    try:
        with urllib.request.urlopen(url, data=data, timeout=15) as resp:
            return bool(json.loads(resp.read()).get("ok"))
    except (urllib.error.URLError, OSError, ValueError) as e:
        _log("telegram send failed", str(e))
        return False


def _macos(subject: str) -> bool:
    if sys.platform != "darwin":
        return False
    try:
        subprocess.run(
            ["osascript", "-e",
             f'display notification {json.dumps(subject)} with title "nightly-pipeline"'],
            capture_output=True,
            timeout=10,
        )
        return True
    except (OSError, subprocess.SubprocessError):
        return False


def send(subject: str, body: str) -> None:
    """Best-effort multi-channel notify. Always logs; never raises."""
    _log(subject, body)
    text = f"{subject}\n\n{body}" if body else subject
    if not _telegram(text):
        # Only fall back to a desktop notification if Telegram is unconfigured/failed.
        _macos(subject)


def notify_success(published: list[str], elapsed_sec: int) -> None:
    minutes, seconds = divmod(elapsed_sec, 60)
    body = (
        "\n".join(f"  • {ep}" for ep in published)
        + f"\n\n完了時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )
    send(f"✅ nightly: {len(published)} 件 publish 完了 ({minutes}分{seconds}秒)", body)


def notify_escalation(reason: str, log_tail: str) -> None:
    body = f"要確認: {reason}\n\n--- ログ末尾 ---\n{log_tail}"
    send("⚠️ nightly: 自動修復不可 — 要確認", body)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Send a test notification")
    args = parser.parse_args()

    if args.test:
        send(
            "🔔 nightly supervisor — 通知テスト",
            f"テスト送信: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        )
        channel = "Telegram" if os.environ.get("TELEGRAM_BOT_TOKEN") else "macOS/log"
        print(f"✅ 通知送信完了（{channel}）。ログ: {LOG_FILE}")
