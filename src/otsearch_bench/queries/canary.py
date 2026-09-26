"""CANARY queries: stable, verifiable answers re-run every batch to measure environment drift.

Each canary stores the exact adapter call (query + variables + response hash) its answer came
from, so ``check_canaries`` can re-fetch it and report response-level and answer-level drift
independently of any policy change.
"""

from __future__ import annotations

import random
from typing import Any

from pydantic import BaseModel

from otsearch_bench.env import queries as gql
from otsearch_bench.env.adapter import CallRecord, OTAdapter
from otsearch_bench.env.models import Drug
from otsearch_bench.queries.index import CacheIndex, moa_targets
from otsearch_bench.queries.inverse import disease_ref, drug_ref, target_ref
from otsearch_bench.queries.models import (
    AnswerType,
    CallRef,
    Difficulty,
    EntityRef,
    Provenance,
    QueryClass,
    QueryRecord,
    QuerySet,
    ReferenceAnswer,
)
from otsearch_bench.queries.templates import (
    STAGE_LABELS,
    TEMPLATES,
    render,
    stage_label,
    stage_rank,
)

GENERATOR_VERSION = "canary-v1"


def _call_ref(record: CallRecord, query: str) -> CallRef:
    return CallRef(
        operation=record.operation,
        cache_key=record.cache_key,
        row_id=record.row_id,
        response_sha256=record.response_sha256,
        query=query,
        variables=record.variables,
    )


def _canary(
    index: CacheIndex,
    template_id: str,
    anchors: list[EntityRef],
    triple: dict[str, EntityRef],
    reference: ReferenceAnswer,
    branching: list[int],
    evidence_count: int,
    call: CallRef,
    fields: dict[str, str],
) -> QueryRecord:
    return QueryRecord(
        query_id=QueryRecord.make_id(
            QueryClass.CANARY, template_id, [a.id for a in anchors], index.data_version
        ),
        query_class=QueryClass.CANARY,
        template_id=template_id,
        question=render(template_id, **fields),
        anchors=anchors,
        triple=triple,
        reference=reference,
        difficulty=Difficulty(
            hops=1,
            branching=branching,
            shape=TEMPLATES[template_id].shape,
            evidence_count=evidence_count,
            contradictory=False,
        ),
        provenance=Provenance(
            data_version=index.data_version, api_version=index.api_version, calls=[call]
        ),
    )


def build_canaries(
    index: CacheIndex, adapter: OTAdapter, rng: random.Random, *, per_kind: int = 10
) -> list[QueryRecord]:
    """Ensembl IDs, targets of approved single-target drugs, and approved indications."""
    approved = [
        d
        for d in sorted(index.drugs.values(), key=lambda d: d.id)
        if d.maximum_clinical_stage == "APPROVAL" and len(moa_targets(d)) == 1
    ]
    stage_codes = sorted(
        {
            i.max_clinical_stage
            for d in index.drugs.values()
            for i in (d.indications.rows if d.indications else [])
            if i.max_clinical_stage in STAGE_LABELS
        },
        key=stage_rank,
    )
    stage_options = [stage_label(c) for c in stage_codes]
    records: list[QueryRecord] = []

    target_ids = sorted({t for d in approved for t in moa_targets(d)})
    for t_id in rng.sample(target_ids, min(per_kind, len(target_ids))):
        target = adapter.get_target(t_id)
        call = _call_ref(adapter.last_call, gql.TARGET)
        T = target_ref(index, target.id, target.approved_symbol)
        records.append(
            _canary(
                index,
                "canary_ensembl_id",
                [T],
                {"target": T},
                ReferenceAnswer(
                    answer_type=AnswerType.IDENTIFIER,
                    identifier=target.id,
                    entities=[T],
                    explanation=f"{target.approved_symbol} is {target.id} in Ensembl.",
                ),
                [1],
                1,
                call,
                {"target": target.approved_symbol},
            )
        )

    drug_pool = rng.sample(approved, len(approved))
    used: set[str] = set()
    for candidate in drug_pool:
        if sum(r.template_id == "canary_drug_target" for r in records) >= per_kind:
            break
        drug = adapter.get_drug(candidate.id)
        call = _call_ref(adapter.last_call, gql.DRUG)
        moa = moa_targets(drug)
        if len(moa) != 1:
            continue
        ((t_id, symbol),) = moa.items()
        D, T = drug_ref(index, drug.id, drug.name), target_ref(index, t_id, symbol)
        used.add(drug.id)
        records.append(
            _canary(
                index,
                "canary_drug_target",
                [D],
                {"drug": D, "target": T},
                ReferenceAnswer(
                    answer_type=AnswerType.ENTITY,
                    entities=[T],
                    explanation=f"{D.name} acts on {T.name}.",
                ),
                [1],
                1,
                call,
                {"drug": D.name},
            )
        )

    for candidate in drug_pool:
        if sum(r.template_id == "canary_approved_indication" for r in records) >= per_kind:
            break
        if candidate.id in used:
            continue
        drug = adapter.get_drug(candidate.id)
        call = _call_ref(adapter.last_call, gql.DRUG)
        approvals = sorted(
            (
                i
                for i in (drug.indications.rows if drug.indications else [])
                if i.disease and i.max_clinical_stage == "APPROVAL"
            ),
            key=lambda i: i.disease.id,
        )
        if not approvals:
            continue
        ind = approvals[0]
        D, X = drug_ref(index, drug.id, drug.name), disease_ref(ind.disease.id, ind.disease.name)
        records.append(
            _canary(
                index,
                "canary_approved_indication",
                [D, X],
                {"drug": D, "disease": X},
                ReferenceAnswer(
                    answer_type=AnswerType.CATEGORY,
                    category=stage_label("APPROVAL"),
                    category_options=stage_options,
                    explanation=f"{D.name} is approved for {X.name}.",
                ),
                [drug.indications.count if drug.indications else 0],
                1,
                call,
                {"drug": D.name, "disease": X.name, "options": ", ".join(stage_options)},
            )
        )
    return records


# --- drift check ----------------------------------------------------------------------------


class CanaryResult(BaseModel):
    query_id: str
    operation: str | None
    cache_key: str
    previous_sha256: str
    current_sha256: str
    response_changed: bool
    previous_answer: str | None
    current_answer: str | None
    answer_changed: bool


def reference_answer(record: QueryRecord) -> str | None:
    ref = record.reference
    if ref.identifier is not None:
        return ref.identifier
    if ref.answer_type is AnswerType.ENTITY:
        return ",".join(sorted(e.id for e in ref.entities))
    return ref.category


def answer_from_response(record: QueryRecord, response: dict[str, Any]) -> str | None:
    data = response.get("data") or {}
    match record.template_id:
        case "canary_ensembl_id":
            return (data.get("target") or {}).get("id")
        case "canary_drug_target" if data.get("drug"):
            return ",".join(sorted(moa_targets(Drug.model_validate(data["drug"]))))
        case "canary_approved_indication" if data.get("drug"):
            drug = Drug.model_validate(data["drug"])
            disease_id = record.triple["disease"].id
            for ind in drug.indications.rows if drug.indications else []:
                if ind.disease and ind.disease.id == disease_id:
                    return stage_label(ind.max_clinical_stage)
            return None
    return None


def check_canaries(adapter: OTAdapter, queryset: QuerySet) -> list[CanaryResult]:
    """Re-fetch every canary's source call over the network and compare hash and answer."""
    results = []
    for record in queryset.records:
        for call in record.provenance.calls:
            if call.query is None:
                continue
            fresh = adapter.execute(call.query, call.variables, force_network=True)
            previous, current = (
                reference_answer(record),
                answer_from_response(record, fresh.response),
            )
            results.append(
                CanaryResult(
                    query_id=record.query_id,
                    operation=call.operation,
                    cache_key=call.cache_key,
                    previous_sha256=call.response_sha256,
                    current_sha256=fresh.response_sha256,
                    response_changed=fresh.response_sha256 != call.response_sha256,
                    previous_answer=previous,
                    current_answer=current,
                    answer_changed=previous != current,
                )
            )
    return results
