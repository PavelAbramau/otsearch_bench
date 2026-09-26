"""Environment drift detection over the append-only response cache."""

from __future__ import annotations

from itertools import pairwise
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from otsearch_bench.env.cache import CacheStore


class FetchRecord(BaseModel):
    row_id: int
    fetched_at: str
    data_version: str | None
    response_sha256: str


class DriftEntry(BaseModel):
    cache_key: str
    operation: str | None
    variables: dict[str, Any]
    n_fetches: int
    n_distinct_hashes: int
    n_changes: int  # consecutive fetches whose hashes differ
    first_fetched_at: str
    last_fetched_at: str
    data_versions: list[str | None]
    history: list[FetchRecord]


def drift_report(cache: CacheStore | str | Path) -> list[DriftEntry]:
    """List cache keys whose response hash changed between fetches, oldest key first."""
    store = cache if isinstance(cache, CacheStore) else CacheStore(cache)
    try:
        entries = []
        for key in store.drifted_keys():
            rows = store.history(key)
            hashes = [r.response_sha256 for r in rows]
            entries.append(
                DriftEntry(
                    cache_key=key,
                    operation=rows[0].operation,
                    variables=rows[0].variables,
                    n_fetches=len(rows),
                    n_distinct_hashes=len(set(hashes)),
                    n_changes=sum(a != b for a, b in pairwise(hashes)),
                    first_fetched_at=rows[0].fetched_at,
                    last_fetched_at=rows[-1].fetched_at,
                    data_versions=list(dict.fromkeys(r.data_version for r in rows)),
                    history=[
                        FetchRecord(
                            row_id=r.row_id,
                            fetched_at=r.fetched_at,
                            data_version=r.data_version,
                            response_sha256=r.response_sha256,
                        )
                        for r in rows
                    ],
                )
            )
        return entries
    finally:
        if store is not cache:
            store.close()


def format_drift_report(entries: list[DriftEntry]) -> str:
    if not entries:
        return "No drift: every cache key has a single response hash."
    lines = [f"{len(entries)} cache key(s) drifted:"]
    for e in entries:
        lines.append(
            f"- {e.operation} {e.variables} key={e.cache_key[:12]} fetches={e.n_fetches} "
            f"changes={e.n_changes} versions={e.data_versions}"
        )
        lines.extend(
            f"    #{h.row_id} {h.fetched_at} v{h.data_version} {h.response_sha256[:12]}"
            for h in e.history
        )
    return "\n".join(lines)
