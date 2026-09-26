"""Query-set records: questions, constructed reference answers, and difficulty metadata."""

from __future__ import annotations

import json
from collections import Counter
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from otsearch_bench.env.cache import sha256_hex, utc_now_iso

QUERYSET_SCHEMA_VERSION = 1


class QueryClass(StrEnum):
    INVERSE = "inverse_constructed"
    UNANSWERABLE = "unanswerable"
    CANARY = "canary"


CLASS_PREFIX = {QueryClass.INVERSE: "inv", QueryClass.UNANSWERABLE: "una", QueryClass.CANARY: "can"}


class AnswerShape(StrEnum):
    SINGLE_EDGE = "single_edge"
    AGGREGATION = "aggregation"


class AnswerType(StrEnum):
    ENTITY = "entity"  # naming any one accepted entity is correct
    ENTITY_SET = "entity_set"  # the full set is the answer; scored by F1
    CATEGORY = "category"  # one of the options listed in the question
    COUNT = "count"
    IDENTIFIER = "identifier"
    NO_EVIDENCE = "no_evidence"  # correct behaviour: stop and report no evidence


class EntityRef(BaseModel):
    id: str
    entity_type: str
    name: str
    aliases: list[str] = []


class PathEdge(BaseModel):
    source: str
    target: str
    edge_type: str
    evidence_ids: list[str] = []
    score: float | None = None
    attributes: dict[str, Any] = {}


class ReferencePath(BaseModel):
    nodes: list[EntityRef]
    edges: list[PathEdge]


class ReferenceAnswer(BaseModel):
    answer_type: AnswerType
    entities: list[EntityRef] = []
    category: str | None = None
    category_options: list[str] = []
    count: int | None = None
    identifier: str | None = None
    no_evidence: bool = False
    path: ReferencePath | None = None
    explanation: str


EVIDENCE_BUCKETS = ((0, "none"), (10, "low"), (100, "mid"))  # upper bounds; above -> "high"
HIGH_BRANCHING = 50


class Difficulty(BaseModel):
    hops: int
    branching: list[int]  # out-degree at each hop along the path, starting from the anchor
    shape: AnswerShape
    evidence_count: int  # OT evidence records on the answer-bearing edge(s)
    contradictory: bool
    contradiction_kinds: list[str] = []

    @property
    def evidence_bucket(self) -> str:
        for upper, label in EVIDENCE_BUCKETS:
            if self.evidence_count <= upper:
                return label
        return "high"

    @property
    def branching_bucket(self) -> str:
        return "high" if self.branching and max(self.branching) >= HIGH_BRANCHING else "low"

    def cell(self) -> tuple[int, str, str, str, bool]:
        """The full stratification cell used when sampling."""
        return (
            self.hops,
            self.shape.value,
            self.evidence_bucket,
            self.branching_bucket,
            self.contradictory,
        )


class CallRef(BaseModel):
    operation: str | None
    cache_key: str
    row_id: int
    response_sha256: str
    query: str | None = None  # stored for canary calls so they can be re-executed
    variables: dict[str, Any] | None = None


class Provenance(BaseModel):
    data_version: str
    api_version: str | None = None
    calls: list[CallRef] = []
    notes: dict[str, Any] = {}


class QueryRecord(BaseModel):
    query_id: str
    query_class: QueryClass
    template_id: str
    question: str
    anchors: list[EntityRef]
    triple: dict[str, EntityRef] = {}  # keys: drug, target, disease
    reference: ReferenceAnswer
    difficulty: Difficulty
    provenance: Provenance

    @property
    def stratum(self) -> str:
        d = self.difficulty
        return f"{self.query_class.value} | hops={d.hops} | {d.shape.value}"

    @staticmethod
    def make_id(
        query_class: QueryClass, template_id: str, anchor_ids: list[str], data_version: str
    ) -> str:
        digest = sha256_hex("|".join([query_class.value, template_id, *anchor_ids, data_version]))
        return f"{CLASS_PREFIX[query_class]}-{template_id}-{digest[:10]}"


class QuerySet(BaseModel):
    schema_version: int = QUERYSET_SCHEMA_VERSION
    name: str
    query_class: QueryClass
    data_version: str
    generator_version: str
    created_at: str = Field(default_factory=utc_now_iso)
    config: dict[str, Any] = {}
    records: list[QueryRecord]

    def counts_by_stratum(self) -> dict[str, int]:
        return dict(sorted(Counter(r.stratum for r in self.records).items()))

    def to_jsonl(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        header = self.model_dump(mode="json", exclude={"records"})
        with path.open("w", encoding="utf-8") as fh:
            fh.write(json.dumps({"record": "header", **header}, sort_keys=True) + "\n")
            for record in self.records:
                line = {"record": "query", **record.model_dump(mode="json")}
                fh.write(json.dumps(line, sort_keys=True, ensure_ascii=False) + "\n")
        return path

    @classmethod
    def from_jsonl(cls, path: str | Path) -> QuerySet:
        header: dict[str, Any] = {}
        records: list[dict[str, Any]] = []
        with Path(path).open(encoding="utf-8") as fh:
            for raw in fh:
                line = json.loads(raw)
                kind = line.pop("record")
                if kind == "header":
                    header = line
                elif kind == "query":
                    records.append(line)
                else:
                    raise ValueError(f"unknown query-set record type {kind!r}")
        return cls.model_validate({**header, "records": records})


def load_querysets(directory: str | Path) -> list[QuerySet]:
    return [QuerySet.from_jsonl(p) for p in sorted(Path(directory).glob("*.jsonl"))]
