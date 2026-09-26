"""``OTAdapter``: the only component allowed to touch the Open Targets Platform API."""

from __future__ import annotations

import email.utils
import json
import os
import random
import re
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict, Field

from otsearch_bench.env import queries as q
from otsearch_bench.env.cache import CacheRow, CacheStore, cache_key
from otsearch_bench.env.drift import DriftEntry, drift_report
from otsearch_bench.env.errors import (
    CacheMissError,
    EntityNotFoundError,
    NetworkDisabledError,
    OTGraphQLError,
    OTHTTPError,
    RetryExhaustedError,
)
from otsearch_bench.env.models import (
    AssociatedDiseases,
    AssociatedTargets,
    Disease,
    Drug,
    EntityResolution,
    EntityType,
    EvidencePage,
    KnownDrugs,
    Meta,
    SearchHit,
    Target,
)
from otsearch_bench.usage import Usage, estimate_tokens

DEFAULT_ENDPOINT = "https://api.platform.opentargets.org/api/v4/graphql"
DEFAULT_CACHE_PATH = Path("data/cache/opentargets.sqlite")
CACHE_PATH_ENV = "OTSEARCH_CACHE_DB"
RETRYABLE_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})
MAX_PAGE_SIZE = 3000  # API-enforced maximum for association pagination
DEFAULT_ENTITY_TYPES = (EntityType.TARGET, EntityType.DISEASE, EntityType.DRUG)

_OPERATION_RE = re.compile(r"\b(?:query|mutation)\s+([A-Za-z_][A-Za-z0-9_]*)")


class Mode(StrEnum):
    LIVE = "live"  # serve from cache when present, otherwise fetch and append
    REPLAY = "replay"  # cache only; a miss raises CacheMissError
    REFRESH = "refresh"  # always fetch and append a new row; old rows are kept


class CallRecord(BaseModel):
    """What one logical adapter request did: provenance, cache status and cost delta."""

    model_config = ConfigDict(frozen=True)

    cache_key: str
    operation: str | None
    variables: dict[str, Any]
    cache_hit: bool
    row_id: int
    response_sha256: str
    data_version: str | None
    fetched_at: str
    network_attempts: int
    latency_ms: float | None
    usage: Usage
    response: dict[str, Any] = Field(repr=False)


class RateLimiter:
    """Spaces requests at least ``1 / rate_per_s`` seconds apart (thread-safe)."""

    def __init__(
        self,
        rate_per_s: float | None,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self._interval = 1.0 / rate_per_s if rate_per_s else 0.0
        self._clock = clock
        self._sleep = sleep
        self._next_slot = float("-inf")
        self._lock = threading.Lock()

    def acquire(self) -> float:
        if self._interval <= 0:
            return 0.0
        with self._lock:
            now = self._clock()
            wait = max(0.0, self._next_slot - now)
            self._next_slot = max(now, self._next_slot) + self._interval
        if wait:
            self._sleep(wait)
        return wait


def operation_name(query: str) -> str | None:
    match = _OPERATION_RE.search(query)
    return match.group(1) if match else None


class OTAdapter:
    """Typed, cached, rate-limited access to the Open Targets Platform GraphQL API.

    Counters (``call_count``, ``network_call_count``, ``tokens``, ``cost_usd`` ...) cover the
    lifetime of this instance, i.e. one run; ``reset_counters()`` starts a new one.
    """

    def __init__(
        self,
        cache: CacheStore | str | Path | None = None,
        *,
        mode: Mode | str = Mode.LIVE,
        endpoint: str = DEFAULT_ENDPOINT,
        timeout_s: float = 60.0,
        max_retries: int = 5,
        backoff_base_s: float = 0.5,
        backoff_max_s: float = 30.0,
        rate_limit_per_s: float | None = 4.0,
        meta_ttl_s: float = 600.0,
        usd_per_network_call: float = 0.0,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
        seed: int | None = None,
    ):
        if cache is None:
            cache = Path(os.environ.get(CACHE_PATH_ENV, DEFAULT_CACHE_PATH))
        self._owns_cache = not isinstance(cache, CacheStore)
        self.cache = cache if isinstance(cache, CacheStore) else CacheStore(cache)
        self.mode = Mode(mode)
        self.endpoint = endpoint
        self.max_retries = max_retries
        self.backoff_base_s = backoff_base_s
        self.backoff_max_s = backoff_max_s
        self.meta_ttl_s = meta_ttl_s
        self.usd_per_network_call = usd_per_network_call
        self._timeout_s = timeout_s
        self._transport = transport
        self._sleep = sleep
        self._clock = clock
        self._rng = random.Random(seed)
        self._rate_limiter = RateLimiter(rate_limit_per_s, clock=clock, sleep=sleep)
        self._http: httpx.Client | None = None
        self._lock = threading.RLock()
        self._live_meta: Meta | None = None
        self._live_meta_at = float("-inf")
        self._data_version: str | None = None
        self._data_versions_seen: dict[str, None] = {}
        self._usage = Usage()
        self._call_log: list[CallRecord] = []

    # --- lifecycle ------------------------------------------------------------------------

    def __enter__(self) -> OTAdapter:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        if self._http is not None:
            self._http.close()
            self._http = None
        if self._owns_cache:
            self.cache.close()

    # --- counters -------------------------------------------------------------------------

    @property
    def usage(self) -> Usage:
        return self._usage.model_copy()

    @property
    def call_count(self) -> int:
        return self._usage.calls

    @property
    def network_call_count(self) -> int:
        return self._usage.network_calls

    @property
    def cache_hit_count(self) -> int:
        return self._usage.cache_hits

    @property
    def retry_count(self) -> int:
        return self._usage.retries

    @property
    def tokens(self) -> int:
        return self._usage.tokens

    @property
    def cost_usd(self) -> float:
        return self._usage.usd

    @property
    def call_log(self) -> tuple[CallRecord, ...]:
        return tuple(self._call_log)

    @property
    def last_call(self) -> CallRecord | None:
        return self._call_log[-1] if self._call_log else None

    @property
    def data_version(self) -> str | None:
        """Data release of the most recent response this adapter served (or fetched meta)."""
        return self._data_version

    @property
    def data_versions_seen(self) -> tuple[str, ...]:
        return tuple(self._data_versions_seen)

    def reset_counters(self) -> None:
        with self._lock:
            self._usage = Usage()
            self._call_log = []
            self._data_versions_seen = {}

    def drift_report(self) -> list[DriftEntry]:
        return drift_report(self.cache)

    # --- core request path ----------------------------------------------------------------

    def execute(
        self,
        query: str,
        variables: Mapping[str, Any] | None = None,
        *,
        force_network: bool = False,
    ) -> CallRecord:
        """Run one GraphQL request under the adapter's mode. All typed methods go through here."""
        variables = {k: v for k, v in (variables or {}).items() if v is not None}
        key = cache_key(query, variables)
        operation = operation_name(query)

        if self.mode is not Mode.REFRESH and not force_network:
            row = self.cache.latest(key)
            if row is not None:
                return self._record(row, cache_hit=True, attempts=0, request_bytes=0)
            if self.mode is Mode.REPLAY:
                raise CacheMissError(key, operation, variables)

        if operation == "Meta":
            data_version = api_version = None  # taken from the response itself below
        else:
            meta = self._ensure_live_meta()
            data_version, api_version = str(meta.data_version), str(meta.api_version)

        request = {"query": query, "variables": variables, "operationName": operation}
        body, attempts, latency_ms, request_bytes = self._post(request)
        if body.get("errors"):
            raise OTGraphQLError(operation, body["errors"])
        if operation == "Meta":
            meta = Meta.model_validate(body["data"]["meta"])
            data_version, api_version = str(meta.data_version), str(meta.api_version)

        row = self.cache.append(
            query=query,
            variables=variables,
            request=request,
            response=body,
            endpoint=self.endpoint,
            data_version=data_version,
            api_version=api_version,
            operation=operation,
            latency_ms=latency_ms,
        )
        return self._record(row, cache_hit=False, attempts=attempts, request_bytes=request_bytes)

    def _record(
        self, row: CacheRow, *, cache_hit: bool, attempts: int, request_bytes: int
    ) -> CallRecord:
        response_text = json.dumps(row.response, separators=(",", ":"))
        delta = Usage(
            calls=1,
            network_calls=attempts,
            cache_hits=int(cache_hit),
            retries=max(0, attempts - 1),
            request_bytes=request_bytes,
            response_bytes=0 if cache_hit else len(response_text.encode()),
            output_tokens=estimate_tokens(response_text),
            usd=attempts * self.usd_per_network_call,
        )
        record = CallRecord(
            cache_key=row.cache_key,
            operation=row.operation,
            variables=row.variables,
            cache_hit=cache_hit,
            row_id=row.row_id,
            response_sha256=row.response_sha256,
            data_version=row.data_version,
            fetched_at=row.fetched_at,
            network_attempts=attempts,
            latency_ms=row.latency_ms,
            usage=delta,
            response=row.response,
        )
        with self._lock:
            # network attempts/retries were already counted as they happened in _post
            self._usage = self._usage + delta.model_copy(update={"network_calls": 0, "retries": 0})
            self._call_log.append(record)
            if row.data_version:
                self._data_version = row.data_version
                self._data_versions_seen[row.data_version] = None
        return record

    def _ensure_live_meta(self) -> Meta:
        with self._lock:
            fresh = self._clock() - self._live_meta_at < self.meta_ttl_s
            if self._live_meta is not None and fresh:
                return self._live_meta
        record = self.execute(q.META, force_network=True)
        meta = Meta.model_validate(record.response["data"]["meta"])
        with self._lock:
            self._live_meta, self._live_meta_at = meta, self._clock()
        return meta

    def _client(self) -> httpx.Client:
        if self.mode is Mode.REPLAY:
            raise NetworkDisabledError("network access attempted in REPLAY mode")
        with self._lock:
            if self._http is None:
                self._http = httpx.Client(
                    timeout=self._timeout_s,
                    transport=self._transport,
                    headers={
                        "Content-Type": "application/json",
                        "User-Agent": "otsearch-bench/0.1",
                    },
                )
            return self._http

    def _post(self, request: dict[str, Any]) -> tuple[dict[str, Any], int, float, int]:
        client = self._client()
        payload = json.dumps(request).encode()
        attempt = 0
        while True:
            attempt += 1
            self._rate_limiter.acquire()
            with self._lock:
                self._usage = self._usage + Usage(
                    network_calls=1, retries=int(attempt > 1), request_bytes=0
                )
            started = time.perf_counter()
            retry_after: float | None = None
            try:
                response = client.post(self.endpoint, content=payload)
            except httpx.TransportError as exc:
                error: Exception = exc
            else:
                latency_ms = (time.perf_counter() - started) * 1000
                if response.status_code == 200:
                    try:
                        return response.json(), attempt, latency_ms, len(payload) * attempt
                    except ValueError as exc:
                        error = exc
                elif response.status_code in RETRYABLE_STATUS:
                    error = OTHTTPError(response.status_code, response.text)
                    retry_after = _parse_retry_after(response.headers.get("Retry-After"))
                else:
                    _raise_for_client_error(request.get("operationName"), response)
            if attempt > self.max_retries:
                raise RetryExhaustedError(
                    f"{request.get('operationName')} failed after {attempt} attempts: {error}"
                ) from error
            self._sleep(self._backoff(attempt, retry_after))

    def _backoff(self, attempt: int, retry_after: float | None) -> float:
        if retry_after is not None:
            return min(retry_after, self.backoff_max_s)
        ceiling = min(self.backoff_max_s, self.backoff_base_s * 2 ** (attempt - 1))
        return ceiling * (0.5 + 0.5 * self._rng.random())  # jittered exponential backoff

    # --- typed queries --------------------------------------------------------------------

    def get_meta(self) -> Meta:
        """API/data versions. Fetched live in LIVE/REFRESH; latest cached in REPLAY."""
        if self.mode is Mode.REPLAY:
            row = self.cache.latest_for_operation("Meta")
            if row is None:
                raise CacheMissError(cache_key(q.META, {}), "Meta", {})
            return Meta.model_validate(row.response["data"]["meta"])
        self._live_meta_at = float("-inf")
        return self._ensure_live_meta()

    def resolve_entity(
        self,
        name: str,
        entity_types: Sequence[EntityType | str] | None = DEFAULT_ENTITY_TYPES,
        *,
        search_size: int = 10,
    ) -> EntityResolution:
        """Map a free-text name to an ID: exact ``mapIds`` first, ranked ``search`` fallback."""
        names = sorted({EntityType(e).value for e in entity_types}) if entity_types else None
        record = self.execute(q.MAP_IDS, {"terms": [name], "entityNames": names})
        hits = [
            SearchHit.model_validate(hit)
            for mapping in record.response["data"]["mapIds"]["mappings"]
            for hit in mapping.get("hits") or []
        ]
        method = "mapIds"
        if not hits:
            page = {"index": 0, "size": search_size}
            record = self.execute(q.SEARCH, {"q": name, "entityNames": names, "page": page})
            hits = [SearchHit.model_validate(h) for h in record.response["data"]["search"]["hits"]]
            method = "search"
        return EntityResolution(
            query=name,
            entity_types=names,
            method=method,
            best=_pick_best(name, hits),
            candidates=hits,
        )

    def get_target(self, ensembl_id: str) -> Target:
        data = self._entity(q.TARGET, {"ensemblId": ensembl_id}, "target", ensembl_id)
        return Target.model_validate(data)

    def get_disease(self, efo_id: str) -> Disease:
        data = self._entity(q.DISEASE, {"efoId": efo_id}, "disease", efo_id)
        return Disease.model_validate(data)

    def get_drug(self, chembl_id: str) -> Drug:
        data = self._entity(q.DRUG, {"chemblId": chembl_id}, "drug", chembl_id)
        return Drug.model_validate(data)

    def get_known_drugs(self, ensembl_id: str) -> KnownDrugs:
        data = self._entity(q.KNOWN_DRUGS, {"ensemblId": ensembl_id}, "target", ensembl_id)
        block = data["drugAndClinicalCandidates"]
        return KnownDrugs.model_validate(
            {"target": data, "count": block["count"], "rows": block["rows"]}
        )

    def get_associated_diseases(
        self,
        ensembl_id: str,
        *,
        page_index: int = 0,
        page_size: int = 25,
        enable_indirect: bool = False,
        disease_ids: Sequence[str] | None = None,
    ) -> AssociatedDiseases:
        page = _page(page_index, page_size)
        variables = {
            "ensemblId": ensembl_id,
            "page": page,
            "enableIndirect": True if enable_indirect else None,
            "Bs": sorted(disease_ids) if disease_ids else None,
        }
        data = self._entity(q.ASSOCIATED_DISEASES, variables, "target", ensembl_id)
        block = data["associatedDiseases"]
        return AssociatedDiseases.model_validate(
            {"target": data, "count": block["count"], "rows": block["rows"], **_page_fields(page)}
        )

    def get_all_associated_diseases(
        self, ensembl_id: str, *, enable_indirect: bool = False
    ) -> AssociatedDiseases:
        """Every associated disease, paging at the API maximum page size."""
        first = self.get_associated_diseases(
            ensembl_id, page_size=MAX_PAGE_SIZE, enable_indirect=enable_indirect
        )
        rows, index = list(first.rows), 1
        while len(rows) < first.count:
            page = self.get_associated_diseases(
                ensembl_id,
                page_index=index,
                page_size=MAX_PAGE_SIZE,
                enable_indirect=enable_indirect,
            )
            if not page.rows:
                break
            rows.extend(page.rows)
            index += 1
        return first.model_copy(update={"rows": rows, "page_size": len(rows)})

    def get_associated_targets(
        self,
        efo_id: str,
        *,
        page_index: int = 0,
        page_size: int = 25,
        enable_indirect: bool = False,
        target_ids: Sequence[str] | None = None,
    ) -> AssociatedTargets:
        page = _page(page_index, page_size)
        variables = {
            "efoId": efo_id,
            "page": page,
            "enableIndirect": True if enable_indirect else None,
            "Bs": sorted(target_ids) if target_ids else None,
        }
        data = self._entity(q.ASSOCIATED_TARGETS, variables, "disease", efo_id)
        block = data["associatedTargets"]
        return AssociatedTargets.model_validate(
            {"disease": data, "count": block["count"], "rows": block["rows"], **_page_fields(page)}
        )

    def get_evidence(
        self,
        ensembl_id: str,
        efo_id: str,
        *,
        size: int = 50,
        cursor: str | None = None,
        datasource_ids: Sequence[str] | None = None,
    ) -> EvidencePage:
        variables = {
            "ensemblId": ensembl_id,
            "efoIds": [efo_id],
            "size": size,
            "cursor": cursor,
            "datasourceIds": sorted(datasource_ids) if datasource_ids else None,
        }
        data = self._entity(q.EVIDENCE, variables, "target", ensembl_id)
        block = data["evidences"]
        return EvidencePage(
            target_id=ensembl_id,
            disease_id=efo_id,
            count=block["count"],
            cursor=block["cursor"],
            rows=block["rows"],
        )

    def get_all_evidence(
        self,
        ensembl_id: str,
        efo_id: str,
        *,
        datasource_ids: Sequence[str] | None = None,
        page_size: int = 5000,
        max_pages: int | None = None,
    ) -> EvidencePage:
        """Follow evidence cursors until exhausted; ``cursor`` stays set if ``max_pages`` cut it."""
        page = self.get_evidence(ensembl_id, efo_id, size=page_size, datasource_ids=datasource_ids)
        rows, pages = list(page.rows), 1
        while page.cursor and len(rows) < page.count and (max_pages is None or pages < max_pages):
            page = self.get_evidence(
                ensembl_id,
                efo_id,
                size=page_size,
                cursor=page.cursor,
                datasource_ids=datasource_ids,
            )
            if not page.rows:
                break
            rows.extend(page.rows)
            pages += 1
        return EvidencePage(
            target_id=ensembl_id,
            disease_id=efo_id,
            count=page.count,
            cursor=page.cursor if len(rows) < page.count else None,
            rows=rows,
        )

    def _entity(
        self, query: str, variables: dict[str, Any], root: str, entity_id: str
    ) -> dict[str, Any]:
        record = self.execute(query, variables)
        data = record.response["data"][root]
        if data is None:
            raise EntityNotFoundError(root, entity_id)
        return data


def _page(index: int, size: int) -> dict[str, int]:
    if index < 0 or not 1 <= size <= MAX_PAGE_SIZE:
        raise ValueError(f"invalid page index={index} size={size}")
    return {"index": index, "size": size}


def _page_fields(page: dict[str, int]) -> dict[str, int]:
    return {"page_index": page["index"], "page_size": page["size"]}


def _pick_best(name: str, hits: list[SearchHit]) -> SearchHit | None:
    wanted = name.strip().casefold()
    for hit in hits:
        if hit.name.casefold() == wanted or hit.id.casefold() == wanted:
            return hit
    return hits[0] if hits else None


def _parse_retry_after(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        parsed = email.utils.parsedate_to_datetime(value)
        if parsed is None:
            return None
        return max(0.0, (parsed - datetime.now(UTC)).total_seconds())


def _raise_for_client_error(operation: str | None, response: httpx.Response) -> None:
    try:
        body = response.json()
    except ValueError:
        body = None
    if isinstance(body, dict) and body.get("errors"):
        raise OTGraphQLError(operation, body["errors"])
    raise OTHTTPError(response.status_code, response.text)
