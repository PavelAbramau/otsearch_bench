"""Exceptions raised by the environment access layer."""

from __future__ import annotations

from typing import Any


class OTError(Exception):
    """Base class for all Open Targets adapter errors."""


class CacheMissError(OTError):
    """REPLAY mode was asked for a request that has no cached response."""

    def __init__(self, cache_key: str, operation: str | None, variables: dict[str, Any]):
        super().__init__(f"no cached response for {operation} {variables} (key={cache_key[:12]})")
        self.cache_key = cache_key
        self.operation = operation
        self.variables = variables


class NetworkDisabledError(OTError):
    """A network request was attempted while the adapter is in REPLAY mode."""


class OTHTTPError(OTError):
    """The API returned a non-retryable HTTP status."""

    def __init__(self, status_code: int, body: str):
        super().__init__(f"HTTP {status_code}: {body[:300]}")
        self.status_code = status_code
        self.body = body


class RetryExhaustedError(OTError):
    """A retryable failure persisted past ``max_retries``."""


class OTGraphQLError(OTError):
    """The API answered with a GraphQL ``errors`` array. Such responses are never cached."""

    def __init__(self, operation: str | None, errors: list[dict[str, Any]]):
        messages = "; ".join(str(e.get("message", e)) for e in errors)
        super().__init__(f"{operation}: {messages}")
        self.operation = operation
        self.errors = errors


class EntityNotFoundError(OTError, LookupError):
    """The API returned null for an entity lookup (the null response is still cached)."""

    def __init__(self, entity: str, entity_id: str):
        super().__init__(f"{entity} {entity_id!r} not found")
        self.entity = entity
        self.entity_id = entity_id
