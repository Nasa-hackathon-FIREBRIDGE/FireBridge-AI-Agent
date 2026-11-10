"""SQLite persistence for analyses, summaries, sessions, messages and long-term memories."""

import asyncio
import json
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import aiosqlite

SCHEMA = """
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;
CREATE TABLE IF NOT EXISTS analyses (
    id TEXT PRIMARY KEY, title TEXT NOT NULL, sources TEXT NOT NULL, pack TEXT NOT NULL,
    created_at TEXT NOT NULL, updated_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS summaries (
    analysis_id TEXT NOT NULL REFERENCES analyses(id) ON DELETE CASCADE, language TEXT NOT NULL,
    payload TEXT NOT NULL, created_at TEXT NOT NULL, PRIMARY KEY (analysis_id, language));
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY, analysis_id TEXT NOT NULL REFERENCES analyses(id) ON DELETE CASCADE,
    user_id TEXT, language TEXT NOT NULL, rolling_summary TEXT NOT NULL DEFAULT '',
    summarized_upto INTEGER NOT NULL DEFAULT 0, created_at TEXT NOT NULL, updated_at TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS sessions_user ON sessions(user_id, updated_at);
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    role TEXT NOT NULL, content TEXT NOT NULL, meta TEXT NOT NULL DEFAULT '{}', created_at TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS messages_session ON messages(session_id, id);
CREATE TABLE IF NOT EXISTS memories (
    id INTEGER PRIMARY KEY AUTOINCREMENT, scope TEXT NOT NULL, scope_id TEXT NOT NULL, kind TEXT NOT NULL,
    content TEXT NOT NULL, importance REAL NOT NULL, uses INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL, last_used_at TEXT, source_session TEXT);
CREATE INDEX IF NOT EXISTS memories_scope ON memories(scope, scope_id);
"""


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class Store:
    def __init__(self, path: Path | str):
        self._path = str(path)
        self._db: aiosqlite.Connection | None = None
        self._lock = asyncio.Lock()

    async def open(self) -> None:
        if self._path != ":memory:":
            Path(self._path).parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(self._path)
        self._db.row_factory = aiosqlite.Row
        await self._db.executescript(SCHEMA)
        await self._db.commit()

    async def close(self) -> None:
        if self._db:
            await self._db.close()
            self._db = None

    @property
    def db(self) -> aiosqlite.Connection:
        if not self._db:
            raise RuntimeError("Store is not open")
        return self._db

    async def ping(self) -> bool:
        async with self.db.execute("SELECT 1") as cursor:
            return (await cursor.fetchone()) is not None

    async def _write(self, sql: str, params: tuple = ()) -> int:
        """Execute a statement; returns the number of rows affected."""
        async with self._lock:
            cursor = await self.db.execute(sql, params)
            await self.db.commit()
            return cursor.rowcount

    async def _insert(self, sql: str, params: tuple = ()) -> int:
        """Insert one row; returns its id."""
        async with self._lock:
            cursor = await self.db.execute(sql, params)
            await self.db.commit()
            return int(cursor.lastrowid or 0)

    async def _rows(self, sql: str, params: tuple = ()) -> list[dict[str, Any]]:
        async with self.db.execute(sql, params) as cursor:
            return [dict(row) for row in await cursor.fetchall()]

    # ── analyses and summaries ───────────────────────────────────────────────
    async def save_analysis(self, analysis_id: str, title: str, sources: list[str], pack_json: str) -> None:
        stamp = now()
        await self._write(
            "INSERT INTO analyses (id, title, sources, pack, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(id) DO UPDATE SET title = excluded.title, sources = excluded.sources, pack = excluded.pack, updated_at = excluded.updated_at",
            (analysis_id, title, json.dumps(sources), pack_json, stamp, stamp),
        )
        # A rebuilt pack invalidates cached summaries.
        await self._write("DELETE FROM summaries WHERE analysis_id = ?", (analysis_id,))

    async def get_analysis(self, analysis_id: str) -> dict[str, Any] | None:
        rows = await self._rows("SELECT * FROM analyses WHERE id = ?", (analysis_id,))
        return rows[0] if rows else None

    async def list_analyses(self) -> list[dict[str, Any]]:
        return await self._rows(
            "SELECT id, title, sources, created_at, updated_at FROM analyses ORDER BY updated_at DESC"
        )

    async def save_summary(self, analysis_id: str, language: str, payload: dict[str, Any]) -> None:
        await self._write(
            "INSERT OR REPLACE INTO summaries (analysis_id, language, payload, created_at) VALUES (?, ?, ?, ?)",
            (analysis_id, language, json.dumps(payload), now()),
        )

    async def get_summary(self, analysis_id: str, language: str) -> dict[str, Any] | None:
        rows = await self._rows(
            "SELECT payload FROM summaries WHERE analysis_id = ? AND language = ?", (analysis_id, language)
        )
        return json.loads(rows[0]["payload"]) if rows else None

    # ── sessions and messages ────────────────────────────────────────────────
    async def create_session(self, analysis_id: str, user_id: str | None, language: str) -> dict[str, Any]:
        session_id = uuid.uuid4().hex
        stamp = now()
        await self._write(
            "INSERT INTO sessions (id, analysis_id, user_id, language, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
            (session_id, analysis_id, user_id, language, stamp, stamp),
        )
        return (await self.get_session(session_id)) or {}

    async def get_session(self, session_id: str) -> dict[str, Any] | None:
        rows = await self._rows("SELECT * FROM sessions WHERE id = ?", (session_id,))
        return rows[0] if rows else None

    async def list_sessions(
        self, user_id: str | None = None, analysis_id: str | None = None, limit: int = 50
    ) -> list[dict[str, Any]]:
        clauses, params = [], []
        if user_id:
            clauses.append("user_id = ?")
            params.append(user_id)
        if analysis_id:
            clauses.append("analysis_id = ?")
            params.append(analysis_id)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        return await self._rows(f"SELECT * FROM sessions {where} ORDER BY updated_at DESC LIMIT ?", (*params, limit))

    async def delete_session(self, session_id: str) -> bool:
        return bool(await self._write("DELETE FROM sessions WHERE id = ?", (session_id,)))

    async def update_session_summary(self, session_id: str, summary: str, summarized_upto: int) -> None:
        await self._write(
            "UPDATE sessions SET rolling_summary = ?, summarized_upto = ?, updated_at = ? WHERE id = ?",
            (summary, summarized_upto, now(), session_id),
        )

    async def add_message(self, session_id: str, role: str, content: str, meta: dict[str, Any] | None = None) -> int:
        message_id = await self._insert(
            "INSERT INTO messages (session_id, role, content, meta, created_at) VALUES (?, ?, ?, ?, ?)",
            (session_id, role, content, json.dumps(meta or {}), now()),
        )
        await self._write("UPDATE sessions SET updated_at = ? WHERE id = ?", (now(), session_id))
        return message_id

    async def messages(self, session_id: str, after_id: int = 0) -> list[dict[str, Any]]:
        rows = await self._rows(
            "SELECT * FROM messages WHERE session_id = ? AND id > ? ORDER BY id", (session_id, after_id)
        )
        for row in rows:
            row["meta"] = json.loads(row["meta"])
        return rows

    # ── long-term memories ───────────────────────────────────────────────────
    async def add_memory(
        self, scope: str, scope_id: str, kind: str, content: str, importance: float, source_session: str | None
    ) -> int:
        return await self._insert(
            "INSERT INTO memories (scope, scope_id, kind, content, importance, created_at, source_session) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (scope, scope_id, kind, content, importance, now(), source_session),
        )

    async def memories(self, scopes: list[tuple[str, str]]) -> list[dict[str, Any]]:
        if not scopes:
            return []
        where = " OR ".join("(scope = ? AND scope_id = ?)" for _ in scopes)
        return await self._rows(
            f"SELECT * FROM memories WHERE {where} ORDER BY id", tuple(v for pair in scopes for v in pair)
        )

    async def touch_memories(self, ids: list[int]) -> None:
        if ids:
            await self._write(
                f"UPDATE memories SET uses = uses + 1, last_used_at = ? WHERE id IN ({','.join('?' * len(ids))})",
                (now(), *ids),
            )

    async def reinforce_memory(self, memory_id: int, importance: float) -> None:
        await self._write(
            "UPDATE memories SET importance = MAX(importance, ?), uses = uses + 1, last_used_at = ? WHERE id = ?",
            (importance, now(), memory_id),
        )

    async def delete_memory(self, memory_id: int) -> bool:
        return bool(await self._write("DELETE FROM memories WHERE id = ?", (memory_id,)))

    async def delete_memories(self, scope: str, scope_id: str) -> int:
        return await self._write("DELETE FROM memories WHERE scope = ? AND scope_id = ?", (scope, scope_id))

    async def prune_memories(self, scope: str, scope_id: str, keep: int) -> None:
        """Keep the most valuable memories of a scope: importance, then use, then recency."""
        await self._write(
            "DELETE FROM memories WHERE scope = ? AND scope_id = ? AND id NOT IN ("
            "SELECT id FROM memories WHERE scope = ? AND scope_id = ? ORDER BY importance DESC, uses DESC, id DESC LIMIT ?)",
            (scope, scope_id, scope, scope_id, keep),
        )
