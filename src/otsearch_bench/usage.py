"""Call/token/cost accounting shared by the environment layer and (later) model calls."""

from __future__ import annotations

from pydantic import BaseModel


class Usage(BaseModel):
    """Additive usage counters. ``+`` combines two snapshots; ``-`` gives a delta."""

    calls: int = 0  # logical requests (cache hits included)
    network_calls: int = 0  # HTTP attempts actually sent, retries included
    cache_hits: int = 0
    retries: int = 0
    request_bytes: int = 0
    response_bytes: int = 0
    input_tokens: int = 0
    output_tokens: int = 0  # for API calls: estimated tokens of the served observation
    usd: float = 0.0
    llm_calls: int = 0  # model requests (cache hits included); never counted in ``calls``
    llm_cache_hits: int = 0

    def _combine(self, other: Usage, sign: int) -> Usage:
        return Usage(
            **{
                name: getattr(self, name) + sign * getattr(other, name)
                for name in type(self).model_fields
            }
        )

    def __add__(self, other: Usage) -> Usage:
        return self._combine(other, 1)

    def __sub__(self, other: Usage) -> Usage:
        return self._combine(other, -1)

    @property
    def tokens(self) -> int:
        return self.input_tokens + self.output_tokens


def estimate_tokens(text: str) -> int:
    """Rough token estimate (~4 characters per token) for observations fed to a model."""
    return (len(text) + 3) // 4
