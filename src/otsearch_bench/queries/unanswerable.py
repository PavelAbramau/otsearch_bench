"""UNANSWERABLE queries: triples with no connecting evidence, verified by exhaustive traversal.

A candidate (drug D, target T, disease X) with T a mechanism-of-action target of D is accepted
only if, counting X together with all of its ontology descendants:

* hop 1 from D: D has no clinical indication for X;
* hop 2 from D (= hop 1 from each target t of D, T included): no t is associated with X
  (every indirect association page listed) and no t has direct evidence for X;
* hop 2 from each t: no drug acting on t has clinical evidence for X.

Correct agent behaviour on these questions is to stop early and report no evidence.
"""

from __future__ import annotations

import random
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field

from otsearch_bench.env.adapter import OTAdapter
from otsearch_bench.env.errors import OTError
from otsearch_bench.queries.index import CacheIndex, moa_targets
from otsearch_bench.queries.inverse import disease_ref, drug_ref, target_ref
from otsearch_bench.queries.models import (
    AnswerType,
    CallRef,
    Difficulty,
    Provenance,
    QueryClass,
    QueryRecord,
    ReferenceAnswer,
)
from otsearch_bench.queries.templates import (
    STAGE_LABELS,
    TEMPLATES,
    render,
    stage_label,
    stage_rank,
)

GENERATOR_VERSION = "unanswerable-v1"
MAX_DIRECT_ASSOCIATIONS = 400  # keeps exhaustive indirect listings tractable
MAX_DESCENDANTS = 200
TEMPLATE_CYCLE = ("drug_stage", "sibling_drug", "target_disease_drugs", "drug_disease_target")


@dataclass
class AbsenceCheck:
    verified: bool
    reason: str
    calls: list[CallRef]
    traversed: dict[str, int] = field(default_factory=dict)


def verify_absence(
    adapter: OTAdapter, drug_id: str, target_id: str, disease_id: str
) -> AbsenceCheck:
    """Exhaustively traverse 2 hops around the triple. Must run single-threaded on the adapter."""
    start = len(adapter.call_log)
    traversed: Counter[str] = Counter()

    def result(verified: bool, reason: str) -> AbsenceCheck:
        calls = [
            CallRef(
                operation=r.operation,
                cache_key=r.cache_key,
                row_id=r.row_id,
                response_sha256=r.response_sha256,
            )
            for r in adapter.call_log[start:]
        ]
        return AbsenceCheck(verified, reason, calls, dict(traversed))

    disease = adapter.get_disease(disease_id)
    if len(disease.descendants) > MAX_DESCENDANTS:
        return result(False, "reject: disease too broad")
    related = {disease_id, *disease.descendants}
    traversed["disease_descendants"] = len(disease.descendants)

    drug = adapter.get_drug(drug_id)
    targets = list(moa_targets(drug))
    traversed["drug_targets"] = len(targets)
    if target_id not in targets:
        return result(False, "reject: target is not a mechanism-of-action target")
    indications = drug.indications
    if indications is None or len(indications.rows) < indications.count:
        return result(False, "reject: incomplete indication listing")
    traversed["drug_indications"] = len(indications.rows)
    if any(i.disease and i.disease.id in related for i in indications.rows):
        return result(False, "reject: hop 1 drug indication")

    for t in targets:
        assoc = adapter.get_all_associated_diseases(t, enable_indirect=True)
        traversed["association_rows"] += len(assoc.rows)
        if len(assoc.rows) < assoc.count:
            return result(False, "reject: incomplete association listing")
        if any(r.disease.id in related for r in assoc.rows):
            return result(False, "reject: hop 2 target association")
        if adapter.get_evidence(t, disease_id, size=1).count:
            return result(False, "reject: hop 2 direct evidence")

    for t in targets:
        known = adapter.get_known_drugs(t)
        traversed["known_drug_rows"] += len(known.rows)
        if any(set(row.disease_ids) & related for row in known.rows):
            return result(False, "reject: hop 2 other drug on target")

    return result(True, "verified: no path within 2 hops")


def build_unanswerable(
    index: CacheIndex,
    adapter: OTAdapter,
    n: int,
    rng: random.Random,
    *,
    max_attempts: int = 1000,
    log: Callable[[str], None] = print,
) -> tuple[list[QueryRecord], dict[str, int]]:
    stats: Counter[str] = Counter()
    drugs = [
        d
        for d in sorted(index.drugs.values(), key=lambda d: d.id)
        if (moa := moa_targets(d))
        and len(moa) <= 3
        and all(
            t in index.target_associations
            and index.target_associations[t].count <= MAX_DIRECT_ASSOCIATIONS
            for t in moa
        )
    ]
    diseases = sorted(
        {
            (i.disease.id, i.disease.name)
            for d in index.drugs.values()
            for i in (d.indications.rows if d.indications else [])
            if i.disease
        }
    )
    if not drugs or not diseases:
        return [], {"error: empty candidate pool": 1}
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
    log(f"[unanswerable] {len(drugs)} low-degree drugs x {len(diseases)} diseases")

    records: list[QueryRecord] = []
    seen: set[tuple[str, str]] = set()
    attempts = 0
    while len(records) < n and attempts < max_attempts:
        attempts += 1
        drug = rng.choice(drugs)
        moa = moa_targets(drug)
        t_id = rng.choice(sorted(moa))
        x_id, x_name = rng.choice(diseases)
        if (drug.id, x_id) in seen:
            continue
        seen.add((drug.id, x_id))
        if any(i.disease and i.disease.id == x_id for i in drug.indications.rows):
            stats["prefilter: indicated"] += 1
            continue
        if any((t, x_id) in index.pair_associations for t in moa):
            stats["prefilter: associated"] += 1
            continue
        try:
            check = verify_absence(adapter, drug.id, t_id, x_id)
        except OTError as exc:
            stats[f"error: {type(exc).__name__}"] += 1
            continue
        stats[check.reason] += 1
        if not check.verified:
            continue

        template_id = TEMPLATE_CYCLE[len(records) % len(TEMPLATE_CYCLE)]
        if template_id == "drug_disease_target" and len(moa) < 2:
            template_id = "drug_stage"
        D = drug_ref(index, drug.id, drug.name)
        T = target_ref(index, t_id, moa[t_id])
        X = disease_ref(x_id, x_name)
        anchors = [T, X] if template_id == "target_disease_drugs" else [D, X]
        records.append(
            QueryRecord(
                query_id=QueryRecord.make_id(
                    QueryClass.UNANSWERABLE, template_id, [D.id, T.id, X.id], index.data_version
                ),
                query_class=QueryClass.UNANSWERABLE,
                template_id=template_id,
                question=render(
                    template_id,
                    drug=D.name,
                    target=T.name,
                    disease=X.name,
                    options=", ".join(stage_options),
                ),
                anchors=anchors,
                triple={"drug": D, "target": T, "disease": X},
                reference=ReferenceAnswer(
                    answer_type=AnswerType.NO_EVIDENCE,
                    no_evidence=True,
                    category_options=stage_options if template_id == "drug_stage" else [],
                    explanation=(
                        f"No path within 2 hops connects {D.name} or its targets to {X.name} "
                        f"or its descendants (traversed: {check.traversed})."
                    ),
                ),
                difficulty=Difficulty(
                    hops=2,
                    branching=[
                        check.traversed.get("drug_targets", 0),
                        check.traversed.get("association_rows", 0),
                    ],
                    shape=TEMPLATES[template_id].shape,
                    evidence_count=0,
                    contradictory=False,
                ),
                provenance=Provenance(
                    data_version=index.data_version,
                    api_version=index.api_version,
                    calls=check.calls,
                    notes={"verification": check.traversed},
                ),
            )
        )
        if len(records) % 10 == 0:
            log(f"[unanswerable] {len(records)}/{n} verified after {attempts} attempts")
    stats["attempts"] = attempts
    return records, dict(stats)
