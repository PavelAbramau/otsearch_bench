"""Model access for LLM-backed policies and judges: structured calls, cached for replay.

Every response is appended to a SQLite cache (same append-only store as the OT adapter), keyed by
sha256 of (prompt version, model, system, user, output schema). LIVE serves cached responses and
fills misses, REPLAY never calls the model, REFRESH always calls and appends.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol, TypeVar

from pydantic import BaseModel

from otsearch_bench.env.adapter import Mode
from otsearch_bench.env.cache import CacheStore, cache_key
from otsearch_bench.env.errors import CacheMissError, OTError
from otsearch_bench.usage import Usage, estimate_tokens

M = TypeVar("M", bound=BaseModel)

DEFAULT_MODEL = "claude-sonnet-5"
# USD per million (input, output) tokens.
PRICES = {
    "claude-sonnet-5": (2.0, 10.0),
    "claude-opus-5": (5.0, 25.0),
    "claude-fable-5-1": (10.0, 50.0),
    "claude-haiku-4-5": (1.0, 5.0),
}


class LLMRefusalError(OTError):
    """The model declined or produced no parseable structured output. Never cached."""


class LLMClient(Protocol):
    model: str

    @property
    def usage(self) -> Usage: ...

    def structured(self, *, prompt_version: str, system: str, user: str, output: type[M]) -> M: ...


def _request(model: str, system: str, user: str, output: type[BaseModel], max_tokens: int) -> dict:
    return {
        "model": model,
        "system": system,
        "user": user,
        "schema": output.model_json_schema(),
        "max_tokens": max_tokens,
    }


class CachedAnthropicClient:
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        cache: CacheStore | str | Path = "data/cache/llm.sqlite",
        mode: Mode | str = Mode.LIVE,
        max_tokens: int = 16000,
        client: Any = None,
    ):
        self.model = model
        self.mode = Mode(mode)
        self.max_tokens = max_tokens
        self.cache = cache if isinstance(cache, CacheStore) else CacheStore(cache)
        self._client = client
        self._usage = Usage()
        self._lock = threading.Lock()

    @property
    def usage(self) -> Usage:
        return self._usage.model_copy()

    def _anthropic(self) -> Any:
        if self._client is None:
            import anthropic

            self._client = anthropic.Anthropic(max_retries=5)
        return self._client

    def structured(self, *, prompt_version: str, system: str, user: str, output: type[M]) -> M:
        request = _request(self.model, system, user, output, self.max_tokens)
        query = f"llm:{prompt_version}"
        key = cache_key(query, request)
        if self.mode is not Mode.REFRESH:
            row = self.cache.latest(key)
            if row is not None:
                self._charge(row.response["usage"], cache_hit=True)
                return output.model_validate(row.response["parsed"])
            if self.mode is Mode.REPLAY:
                raise CacheMissError(key, prompt_version, {"model": self.model})

        response = self._anthropic().messages.parse(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
            output_format=output,
        )
        usage = {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        }
        self._charge(usage, cache_hit=False)
        parsed = (
            None if response.stop_reason in ("refusal", "max_tokens") else response.parsed_output
        )
        if parsed is None:
            raise LLMRefusalError(f"{self.model} stop_reason={response.stop_reason}")
        self.cache.append(
            query=query,
            variables=request,
            request=request,
            response={
                "parsed": parsed.model_dump(mode="json"),
                "stop_reason": response.stop_reason,
                "model": response.model,
                "usage": usage,
            },
            endpoint=f"anthropic:{self.model}",
            data_version=None,
            operation=prompt_version,
        )
        return parsed

    def _charge(self, usage: dict[str, int], *, cache_hit: bool) -> None:
        price_in, price_out = PRICES.get(self.model, (0.0, 0.0))
        tokens_in, tokens_out = usage["input_tokens"], usage["output_tokens"]
        delta = Usage(
            llm_calls=1,
            llm_cache_hits=int(cache_hit),
            input_tokens=tokens_in,
            output_tokens=tokens_out,
            usd=0.0 if cache_hit else (tokens_in * price_in + tokens_out * price_out) / 1e6,
        )
        with self._lock:
            self._usage = self._usage + delta


class FakeLLMClient:
    """Deterministic stand-in for tests: ``responder(prompt_version, user, output) -> dict``."""

    def __init__(
        self,
        responder: Callable[[str, str, type[BaseModel]], dict[str, Any]],
        model: str = "fake-llm",
    ):
        self.model = model
        self._responder = responder
        self._usage = Usage()
        self.requests: list[tuple[str, str]] = []

    @property
    def usage(self) -> Usage:
        return self._usage.model_copy()

    def structured(self, *, prompt_version: str, system: str, user: str, output: type[M]) -> M:
        self.requests.append((prompt_version, user))
        result = output.model_validate(self._responder(prompt_version, user, output))
        self._usage = self._usage + Usage(
            llm_calls=1,
            input_tokens=estimate_tokens(system + user),
            output_tokens=estimate_tokens(result.model_dump_json()),
        )
        return result
