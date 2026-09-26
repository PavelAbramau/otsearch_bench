"""``Trajectory``: the complete, replayable record of one run. Serialized as JSONL."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel

from otsearch_bench.env.cache import sha256_hex
from otsearch_bench.search.actions import Action
from otsearch_bench.search.decisions import (
    Answer,
    Formulation,
    PruneDecision,
    Reformulation,
    RejectedAction,
    StoppingDecision,
)
from otsearch_bench.search.state import Budget
from otsearch_bench.usage import Usage

TRAJECTORY_SCHEMA_VERSION = 1


class QueryInput(BaseModel):
    query_id: str
    question: str

    @classmethod
    def coerce(cls, query: str | QueryInput) -> QueryInput:
        if isinstance(query, QueryInput):
            return query
        return cls(query_id=f"q-{sha256_hex(query)[:12]}", question=query)


class CallProvenance(BaseModel):
    cache_key: str
    operation: str | None
    cache_hit: bool
    row_id: int
    response_sha256: str
    data_version: str | None
    fetched_at: str
    network_attempts: int


class ObservationError(BaseModel):
    type: str
    message: str


class Observation(BaseModel):
    action_id: str
    kind: str
    ok: bool
    result: dict[str, Any] | None
    error: ObservationError | None
    calls: list[CallProvenance]
    raw_responses: list[dict[str, Any]] | None
    usage: Usage


class Step(BaseModel):
    index: int
    phase: Literal["formulate", "expand"]
    state_fingerprint: str  # before the step's actions ran
    state_fingerprint_after: str
    formulation: Formulation | None = None
    proposed_actions: list[Action]
    admitted_actions: list[Action]
    rejected_actions: list[RejectedAction]
    observations: list[Observation]
    cache_hit_flags: list[bool]  # one per adapter call made during the step
    cost_delta: Usage
    budget: Budget  # after the step
    prune: PruneDecision | None = None
    reformulation: Reformulation | None = None
    stopping: StoppingDecision | None = None
    wall_clock_s: float


_HEADER_FIELDS = (
    "schema_version",
    "trajectory_id",
    "agent_id",
    "versions",
    "query",
    "seed",
    "adapter_mode",
    "started_at",
)


class Trajectory(BaseModel):
    schema_version: int = TRAJECTORY_SCHEMA_VERSION
    trajectory_id: str
    agent_id: str
    versions: dict[str, Any]
    query: QueryInput
    seed: int
    adapter_mode: str
    started_at: str
    steps: list[Step]
    finished_at: str
    stopping: StoppingDecision
    answer: Answer
    total_usage: Usage
    data_versions_seen: list[str]
    final_state_fingerprint: str
    final_state: dict[str, Any]

    @staticmethod
    def make_id(agent_id: str, query_id: str, seed: int) -> str:
        return sha256_hex(f"{agent_id}|{query_id}|{seed}")[:16]

    @property
    def filename(self) -> str:
        safe_query = re.sub(r"[^A-Za-z0-9._-]+", "_", self.query.query_id)
        return f"{self.agent_id}__{safe_query}__seed{self.seed}.jsonl"

    def to_jsonl(self, path: str | Path) -> Path:
        """Write to ``path``; if it is a directory, use the (agent, query, seed) filename."""
        path = Path(path)
        if path.is_dir():
            path = path / self.filename
        path.parent.mkdir(parents=True, exist_ok=True)
        data = self.model_dump(mode="json")
        steps = data.pop("steps")
        header = {k: data.pop(k) for k in _HEADER_FIELDS}
        lines = [
            {"record": "header", **header},
            *({"record": "step", **step} for step in steps),
            {"record": "result", **data},
        ]
        with path.open("w", encoding="utf-8") as fh:
            for line in lines:
                fh.write(json.dumps(line, ensure_ascii=False, sort_keys=True) + "\n")
        return path

    @classmethod
    def from_jsonl(cls, path: str | Path) -> Trajectory:
        merged: dict[str, Any] = {"steps": []}
        with Path(path).open(encoding="utf-8") as fh:
            for raw in fh:
                record = json.loads(raw)
                kind = record.pop("record")
                if kind == "step":
                    merged["steps"].append(record)
                elif kind in ("header", "result"):
                    merged.update(record)
                else:
                    raise ValueError(f"unknown trajectory record type {kind!r}")
        return cls.model_validate(merged)
