from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable
from typing import Any

import httpx
import pytest

META_BODY = {
    "data": {
        "meta": {
            "name": "fake",
            "apiVersion": {"x": "26", "y": "6", "z": "3", "suffix": None},
            "dataVersion": {"year": "26", "month": "06", "iteration": None},
        }
    }
}


class FakeOT:
    """httpx transport that answers GraphQL operations from a table of canned bodies."""

    def __init__(self, bodies: dict[str, Any] | None = None):
        self.bodies: dict[str, Any] = {"Meta": META_BODY, **(bodies or {})}
        self.requests: list[dict[str, Any]] = []
        self.by_operation: Counter[str] = Counter()
        # optional per-operation queue of httpx.Response objects served before the canned body
        self.scripted: dict[str, list[httpx.Response]] = {}

    def handler(self, request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        op = payload.get("operationName")
        self.requests.append(payload)
        self.by_operation[op] += 1
        if self.scripted.get(op):
            return self.scripted[op].pop(0)
        body = self.bodies[op]
        if callable(body):
            body = body(payload)
        return httpx.Response(200, json=body)

    @property
    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handler)


@pytest.fixture
def fake_ot() -> FakeOT:
    return FakeOT()


@pytest.fixture
def no_sleep() -> Callable[[float], None]:
    sleeps: list[float] = []

    def sleep(seconds: float) -> None:
        sleeps.append(seconds)

    sleep.calls = sleeps  # type: ignore[attr-defined]
    return sleep


def forbid_network(request: httpx.Request) -> httpx.Response:
    raise AssertionError(f"network touched: {request.content[:200]!r}")
