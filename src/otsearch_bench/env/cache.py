"""Append-only, versioned SQLite cache of Open Targets API responses.

Rows are never updated or deleted (enforced by triggers). Each fetch of a request appends a
row, so the history of ``response_sha256`` for a ``cache_key`` records environment drift.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

SCHEMA_VERSION = 1

_SCHEMA = """
CREATE TABLE IF NOT EXISTS cache_entries (
    row_id          INTEGER PRIMARY KEY AUTOINCREMENT,
    cache_key       TEXT NOT NULL,
    operation       TEXT,
    query           TEXT NOT NULL,
    variables_json  TEXT NOT NULL,
    request_json    TEXT NOT NULL,
    response_json   TEXT NOT NULL,
    response_sha256 TEXT NOT NULL,
    fetched_at      TEXT NOT NULL,
    data_version    TEXT,
    api_version     TEXT,
    endpoint        TEXT NOT NULL,
    latency_ms      REAL
);
CREATE INDEX IF NOT EXISTS ix_cache_key_row ON cache_entries (cache_key, row_id);
CREATE INDEX IF NOT EXISTS ix_operation ON cache_entries (operation);
CREATE TRIGGER IF NOT EXISTS cache_entries_no_update BEFORE UPDATE ON cache_entries
BEGIN SELECT RAISE(ABORT, 'cache_entries is append-only'); END;
CREATE TRIGGER IF NOT EXISTS cache_entries_no_delete BEFORE DELETE ON cache_entries
BEGIN SELECT RAISE(ABORT, 'cache_entries is append-only'); END;
"""


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def normalize_query(query: str) -> str:
    """Collapse whitespace so reformatting a GraphQL document does not change its key."""
    return " ".join(query.split())


def sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def cache_key(query: str, variables: dict[str, Any] | None) -> str:
    return sha256_hex(
        canonical_json({"query": normalize_query(query), "variables": variables or {}})
    )


def response_hash(body: Any) -> str:
    return sha256_hex(canonical_json(body))


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds")


class CacheRow(BaseModel):
    model_config = ConfigDict(frozen=True)

    row_id: int
    cache_key: str
    operation: str | None
    query: str
    variables: dict[str, Any]
    request: dict[str, Any]
    response: dict[str, Any]
    response_sha256: str
    fetched_at: str
    data_version: str | None
    api_version: str | None
    endpoint: str
    latency_ms: float | None


class CacheStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA)
        self._conn.execute(f"PRAGMA user_version={SCHEMA_VERSION}")

    def __enter__(self) -> CacheStore:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        self._conn.close()

    def append(
        self,
        *,
        query: str,
        variables: dict[str, Any],
        request: dict[str, Any],
        response: dict[str, Any],
        endpoint: str,
        data_version: str | None,
        api_version: str | None = None,
        operation: str | None = None,
        latency_ms: float | None = None,
        fetched_at: str | None = None,
    ) -> CacheRow:
        values = {
            "cache_key": cache_key(query, variables),
            "operation": operation,
            "query": query,
            "variables_json": canonical_json(variables),
            "request_json": canonical_json(request),
            "response_json": canonical_json(response),
            "response_sha256": response_hash(response),
            "fetched_at": fetched_at or utc_now_iso(),
            "data_version": data_version,
            "api_version": api_version,
            "endpoint": endpoint,
            "latency_ms": latency_ms,
        }
        columns = ", ".join(values)
        placeholders = ", ".join(f":{c}" for c in values)
        with self._lock:
            cur = self._conn.execute(
                f"INSERT INTO cache_entries ({columns}) VALUES ({placeholders})", values
            )
            row = self._conn.execute(
                "SELECT * FROM cache_entries WHERE row_id = ?", (cur.lastrowid,)
            ).fetchone()
        return _to_row(row)

    def latest(self, key: str) -> CacheRow | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM cache_entries WHERE cache_key = ? ORDER BY row_id DESC LIMIT 1",
                (key,),
            ).fetchone()
        return _to_row(row) if row else None

    def latest_for_operation(self, operation: str) -> CacheRow | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM cache_entries WHERE operation = ? ORDER BY row_id DESC LIMIT 1",
                (operation,),
            ).fetchone()
        return _to_row(row) if row else None

    def history(self, key: str) -> list[CacheRow]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM cache_entries WHERE cache_key = ? ORDER BY row_id", (key,)
            ).fetchall()
        return [_to_row(r) for r in rows]

    def iter_rows(self, operation: str | None = None) -> Iterator[CacheRow]:
        """All rows in insertion order, optionally filtered by operation name."""
        sql = "SELECT * FROM cache_entries"
        params: tuple[Any, ...] = ()
        if operation is not None:
            sql += " WHERE operation = ?"
            params = (operation,)
        with self._lock:
            rows = self._conn.execute(sql + " ORDER BY row_id", params).fetchall()
        for r in rows:
            yield _to_row(r)

    def drifted_keys(self) -> list[str]:
        """Keys that have been fetched with more than one distinct response hash."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT cache_key FROM cache_entries GROUP BY cache_key "
                "HAVING COUNT(DISTINCT response_sha256) > 1 ORDER BY MIN(row_id)"
            ).fetchall()
        return [r["cache_key"] for r in rows]

    def count(self) -> int:
        with self._lock:
            return self._conn.execute("SELECT COUNT(*) FROM cache_entries").fetchone()[0]


def _to_row(row: sqlite3.Row) -> CacheRow:
    return CacheRow(
        row_id=row["row_id"],
        cache_key=row["cache_key"],
        operation=row["operation"],
        query=row["query"],
        variables=json.loads(row["variables_json"]),
        request=json.loads(row["request_json"]),
        response=json.loads(row["response_json"]),
        response_sha256=row["response_sha256"],
        fetched_at=row["fetched_at"],
        data_version=row["data_version"],
        api_version=row["api_version"],
        endpoint=row["endpoint"],
        latency_ms=row["latency_ms"],
    )
