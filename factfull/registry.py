"""
factfull/registry.py
=====================
処理済みソースを SQLite で管理するレジストリ。

全ソース種別（podcast / paper / web / book）を一元管理する。

スキーマ:
  source_type : "podcast" | "paper" | "web" | "book"
  source_id   : video_id / arXiv ID / URL / ISBN など
  title       : タイトル（判明した時点で更新）
  status      : "pending" | "processing" | "done" | "failed"
  error       : 失敗時のエラーメッセージ
  graph_written: Neo4j に書き込み済みか
  added_at    : キューに追加した日時
  processed_at: 完了/失敗した日時

使い方:
    from factfull.registry import Registry

    reg = Registry()
    reg.add("podcast", "Q8Fkpi18QXU")
    reg.add("paper",   "1706.03762")

    for item in reg.pending():
        ...
        reg.mark_done(item["source_type"], item["source_id"], graph_written=True)
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_PATH = Path.home() / ".factfull" / "registry.db"


class Registry:
    def __init__(self, path: Path | str = DEFAULT_PATH) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path))
        self._conn.row_factory = sqlite3.Row
        self._setup()

    def _setup(self) -> None:
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS sources (
                source_type  TEXT NOT NULL,
                source_id    TEXT NOT NULL,
                title        TEXT DEFAULT '',
                status       TEXT NOT NULL DEFAULT 'pending',
                error        TEXT DEFAULT '',
                graph_written INTEGER NOT NULL DEFAULT 0,
                added_at     TEXT NOT NULL,
                processed_at TEXT DEFAULT '',
                PRIMARY KEY (source_type, source_id)
            )
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_status ON sources (status)
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_type ON sources (source_type)
        """)
        self._conn.commit()

    # ── 書き込み ─────────────────────────────────────────────────────────────

    def add(
        self,
        source_type: str,
        source_id: str,
        title: str = "",
        *,
        skip_if_exists: bool = True,
    ) -> bool:
        """ソースをキューに追加する。既存の場合は skip_if_exists=True でスキップ。

        Returns:
            True if added, False if skipped.
        """
        now = _now()
        if skip_if_exists:
            cur = self._conn.execute("""
                INSERT OR IGNORE INTO sources
                    (source_type, source_id, title, status, added_at)
                VALUES (?, ?, ?, 'pending', ?)
            """, (source_type, source_id, title, now))
        else:
            cur = self._conn.execute("""
                INSERT OR REPLACE INTO sources
                    (source_type, source_id, title, status, added_at)
                VALUES (?, ?, ?, 'pending', ?)
            """, (source_type, source_id, title, now))
        self._conn.commit()
        return cur.rowcount > 0

    def mark_processing(self, source_type: str, source_id: str) -> None:
        self._conn.execute("""
            UPDATE sources SET status = 'processing'
            WHERE source_type = ? AND source_id = ?
        """, (source_type, source_id))
        self._conn.commit()

    def mark_done(
        self,
        source_type: str,
        source_id: str,
        title: str = "",
        graph_written: bool = False,
    ) -> None:
        self._conn.execute("""
            UPDATE sources
            SET status = 'done', error = '', graph_written = ?,
                processed_at = ?, title = CASE WHEN ? != '' THEN ? ELSE title END
            WHERE source_type = ? AND source_id = ?
        """, (int(graph_written), _now(), title, title, source_type, source_id))
        self._conn.commit()

    def mark_failed(self, source_type: str, source_id: str, error: str = "") -> None:
        self._conn.execute("""
            UPDATE sources
            SET status = 'failed', error = ?, processed_at = ?
            WHERE source_type = ? AND source_id = ?
        """, (error[:500], _now(), source_type, source_id))
        self._conn.commit()

    def retry(self, source_type: str, source_id: str) -> None:
        """failed → pending に戻す。"""
        self._conn.execute("""
            UPDATE sources SET status = 'pending', error = '', processed_at = ''
            WHERE source_type = ? AND source_id = ?
        """, (source_type, source_id))
        self._conn.commit()

    # ── 読み取り ─────────────────────────────────────────────────────────────

    def exists(self, source_type: str, source_id: str) -> bool:
        row = self._conn.execute("""
            SELECT 1 FROM sources WHERE source_type = ? AND source_id = ?
        """, (source_type, source_id)).fetchone()
        return row is not None

    def is_done(self, source_type: str, source_id: str) -> bool:
        row = self._conn.execute("""
            SELECT status FROM sources WHERE source_type = ? AND source_id = ?
        """, (source_type, source_id)).fetchone()
        return row is not None and row["status"] == "done"

    def get(self, source_type: str, source_id: str) -> dict[str, Any] | None:
        row = self._conn.execute("""
            SELECT * FROM sources WHERE source_type = ? AND source_id = ?
        """, (source_type, source_id)).fetchone()
        return dict(row) if row else None

    def pending(self, source_type: str | None = None) -> list[dict[str, Any]]:
        """pending 状態のソース一覧（追加順）。"""
        if source_type:
            rows = self._conn.execute("""
                SELECT * FROM sources WHERE status = 'pending' AND source_type = ?
                ORDER BY added_at
            """, (source_type,)).fetchall()
        else:
            rows = self._conn.execute("""
                SELECT * FROM sources WHERE status = 'pending'
                ORDER BY added_at
            """).fetchall()
        return [dict(r) for r in rows]

    def failed(self, source_type: str | None = None) -> list[dict[str, Any]]:
        q = "SELECT * FROM sources WHERE status = 'failed'"
        params: list[str] = []
        if source_type:
            q += " AND source_type = ?"
            params.append(source_type)
        q += " ORDER BY processed_at DESC"
        return [dict(r) for r in self._conn.execute(q, params).fetchall()]

    def stats(self) -> dict[str, Any]:
        """ステータス別・種別別の集計を返す。"""
        by_status = {
            row["status"]: row["cnt"]
            for row in self._conn.execute("""
                SELECT status, count(*) AS cnt FROM sources GROUP BY status
            """).fetchall()
        }
        by_type = {
            row["source_type"]: row["cnt"]
            for row in self._conn.execute("""
                SELECT source_type, count(*) AS cnt FROM sources GROUP BY source_type
            """).fetchall()
        }
        return {"by_status": by_status, "by_type": by_type}

    def list_all(
        self,
        source_type: str | None = None,
        status: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        if source_type:
            clauses.append("source_type = ?")
            params.append(source_type)
        if status:
            clauses.append("status = ?")
            params.append(status)
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        rows = self._conn.execute(
            f"SELECT * FROM sources {where} ORDER BY added_at DESC LIMIT ?",
            params + [limit],
        ).fetchall()
        return [dict(r) for r in rows]

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> Registry:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
