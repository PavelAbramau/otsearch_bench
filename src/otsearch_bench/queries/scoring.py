"""Score free-form model answers against constructed reference answers."""

from __future__ import annotations

import re
from collections.abc import Callable

from pydantic import BaseModel

from otsearch_bench.env.adapter import OTAdapter
from otsearch_bench.env.errors import OTError
from otsearch_bench.queries.models import AnswerType, EntityRef, QueryRecord

SET_F1_THRESHOLD = 0.5

# (name as written by the model, entity type) -> candidate OT ids
EntityResolver = Callable[[str, str], set[str]]


class ProbeAnswer(BaseModel):
    """Structured answer requested from the model."""

    short_answer: str
    entities: list[str]
    category: str | None
    count: int | None
    identifier: str | None
    no_evidence: bool


class Score(BaseModel):
    query_id: str
    correct: bool
    f1: float | None = None
    matched_ids: list[str] = []
    note: str | None = None


_ROMAN = {"i": "1", "ii": "2", "iii": "3", "iv": "4"}


def normalize(text: str) -> str:
    text = re.sub(r"\([^)]*\)", " ", text.lower())
    tokens = re.sub(r"[^a-z0-9]+", " ", text).split()
    if "phase" in tokens:
        tokens = [_ROMAN.get(t, t) for t in tokens]
    return " ".join(tokens)


def _normalize_identifier(text: str) -> str:
    return re.sub(r"\.\d+$", "", text.strip()).upper()


def adapter_resolver(adapter: OTAdapter) -> EntityResolver:
    """Map a model-written name to OT ids via exact ``mapIds`` matches only (no fuzzy search)."""

    def resolve(name: str, entity_type: str) -> set[str]:
        cleaned = re.sub(r"\([^)]*\)", " ", name).strip()
        if not cleaned:
            return set()
        try:
            resolution = adapter.resolve_entity(cleaned, [entity_type])
        except OTError:
            return set()
        if resolution.method != "mapIds":
            return set()
        return {hit.id for hit in resolution.candidates}

    return resolve


def match_entities(
    names: list[str], accepted: list[EntityRef], resolver: EntityResolver | None
) -> tuple[set[str], int]:
    """Accepted ids matched by the model's names, and the number of distinct names it gave."""
    lookup = {e.id: {normalize(n) for n in (e.name, e.id, *e.aliases) if n} for e in accepted}
    entity_type = accepted[0].entity_type if accepted else None
    matched: set[str] = set()
    predicted: set[str] = set()
    for name in names:
        key = normalize(name)
        if not key:
            continue
        predicted.add(key)
        hits = {i for i, keys in lookup.items() if key in keys}
        if not hits and resolver is not None and entity_type is not None:
            hits = resolver(name, entity_type) & lookup.keys()
        matched |= hits
    return matched, len(predicted)


def score_answer(
    record: QueryRecord, answer: ProbeAnswer | None, resolver: EntityResolver | None = None
) -> Score:
    ref = record.reference
    qid = record.query_id
    if answer is None:
        return Score(query_id=qid, correct=False, note="no parsed answer")

    if ref.answer_type is AnswerType.NO_EVIDENCE:
        return Score(query_id=qid, correct=answer.no_evidence)
    if answer.no_evidence:
        return Score(query_id=qid, correct=False, note="claimed no evidence")

    match ref.answer_type:
        case AnswerType.ENTITY:
            matched, _ = match_entities(answer.entities, ref.entities, resolver)
            return Score(query_id=qid, correct=bool(matched), matched_ids=sorted(matched))
        case AnswerType.ENTITY_SET:
            matched, n_predicted = match_entities(answer.entities, ref.entities, resolver)
            precision = len(matched) / n_predicted if n_predicted else 0.0
            recall = len(matched) / len(ref.entities) if ref.entities else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            return Score(
                query_id=qid,
                correct=f1 >= SET_F1_THRESHOLD,
                f1=f1,
                matched_ids=sorted(matched),
            )
        case AnswerType.CATEGORY:
            given = normalize(answer.category or "")
            return Score(
                query_id=qid, correct=bool(given) and given == normalize(ref.category or "")
            )
        case AnswerType.COUNT:
            return Score(
                query_id=qid, correct=answer.count is not None and answer.count == ref.count
            )
        case AnswerType.IDENTIFIER:
            candidates = [answer.identifier or "", *answer.entities]
            target = _normalize_identifier(ref.identifier or "")
            return Score(
                query_id=qid,
                correct=any(_normalize_identifier(c) == target for c in candidates if c),
            )
    return Score(query_id=qid, correct=False, note=f"unscorable answer type {ref.answer_type}")
