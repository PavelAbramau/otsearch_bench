"""Environment access layer: the Open Targets Platform API behind a cached, typed adapter."""

from otsearch_bench.env.adapter import (
    DEFAULT_ENDPOINT,
    CallRecord,
    Mode,
    OTAdapter,
    RateLimiter,
)
from otsearch_bench.env.cache import CacheRow, CacheStore, cache_key, response_hash
from otsearch_bench.env.drift import DriftEntry, drift_report, format_drift_report
from otsearch_bench.env.errors import (
    CacheMissError,
    EntityNotFoundError,
    NetworkDisabledError,
    OTError,
    OTGraphQLError,
    OTHTTPError,
    RetryExhaustedError,
)
from otsearch_bench.env.models import *  # noqa: F403

__all__ = [
    "DEFAULT_ENDPOINT",
    "CacheMissError",
    "CacheRow",
    "CacheStore",
    "CallRecord",
    "DriftEntry",
    "EntityNotFoundError",
    "Mode",
    "NetworkDisabledError",
    "OTAdapter",
    "OTError",
    "OTGraphQLError",
    "OTHTTPError",
    "RateLimiter",
    "RetryExhaustedError",
    "cache_key",
    "drift_report",
    "format_drift_report",
    "response_hash",
]
