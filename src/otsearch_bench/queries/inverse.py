"""INVERSE-CONSTRUCTED queries: sample a real path from the cache first, then template a question.

Every candidate starts from a (drug, target, disease) triple mined from the cache index where the
drug acts on the target and has a clinical indication for the disease. The path is fixed before
any text is produced; the path is stored as the constructed reference answer.
"""

from __future__ import annotations

import random
from collections import Counter, defaultdict

from otsearch_bench.queries.index import CacheIndex, PairEvidence, drug_aliases, moa_targets
from otsearch_bench.queries.models import (
    AnswerType,
    CallRef,
    Difficulty,
    EntityRef,
    PathEdge,
    Provenance,
    QueryClass,
    QueryRecord,
    ReferenceAnswer,
    ReferencePath,
)
from otsearch_bench.queries.templates import (
    DATATYPE_LABELS,
    PHASE_3_OR_LATER,
    STAGE_LABELS,
    TEMPLATES,
    datatype_label,
    display_drug,
    known_stage,
    render,
    stage_label,
    stage_rank,
)
from otsearch_bench.search.graph import EdgeType

GENERATOR_VERSION = "inverse-v1"
MAX_PATH_EVIDENCE_IDS = 25
TOP_SCORE_MARGIN = 0.01  # argmax questions need a clear winner
MAX_SET_SIZE = 25  # larger answer sets are not a fair "list all" question


def drug_ref(index: CacheIndex, drug_id: str, name: str | None) -> EntityRef:
    drug = index.drugs.get(drug_id)
    label = name or (drug.name if drug else drug_id)
    return EntityRef(
        id=drug_id, entity_type="drug", name=display_drug(label), aliases=drug_aliases(drug)
    )


def target_ref(index: CacheIndex, target_id: str, symbol: str | None) -> EntityRef:
    target = index.targets.get(target_id)
    aliases = sorted({s.label for s in target.symbol_synonyms}) if target else []
    name = symbol or (target.approved_symbol if target else target_id)
    return EntityRef(id=target_id, entity_type="target", name=name, aliases=aliases)


def disease_ref(disease_id: str, name: str | None) -> EntityRef:
    return EntityRef(id=disease_id, entity_type="disease", name=name or disease_id)


def edge(
    source: str,
    target: str,
    edge_type: EdgeType,
    evidence_ids: list[str] | tuple[str, ...] = (),
    score: float | None = None,
    **attributes: object,
) -> PathEdge:
    return PathEdge(
        source=source,
        target=target,
        edge_type=edge_type.value,
        evidence_ids=list(evidence_ids)[:MAX_PATH_EVIDENCE_IDS],
        score=score,
        attributes=attributes,
    )


def clinical_drugs(
    pe: PairEvidence, index: CacheIndex
) -> dict[str, tuple[EntityRef, str, list[str]]]:
    """Drugs with clinical evidence for a pair -> (ref, highest stage, evidence ids)."""
    out: dict[str, tuple[EntityRef, str, list[str]]] = {}
    for r in pe.clinical_rows():
        if r.drug is None:
            continue
        ref, stage, ids = out.get(r.drug.id, (drug_ref(index, r.drug.id, r.drug.name), "", []))
        if stage_rank(r.clinical_stage) > stage_rank(stage):
            stage = r.clinical_stage or stage
        out[r.drug.id] = (ref, stage, [*ids, r.id])
    return dict(sorted(out.items()))


def _drug_edges(
    drugs: dict[str, tuple[EntityRef, str, list[str]]], target: EntityRef, disease: EntityRef
) -> list[PathEdge]:
    return [
        e
        for ref, stage, ids in drugs.values()
        for e in (
            edge(ref.id, target.id, EdgeType.TARGETS, [f"moa:{ref.id}:{target.id}"]),
            edge(ref.id, disease.id, EdgeType.INDICATED_FOR, ids, max_clinical_stage=stage),
        )
    ]


class _Builder:
    def __init__(self, index: CacheIndex):
        self.index = index
        self.records: dict[str, QueryRecord] = {}

    def add(
        self,
        template_id: str,
        *,
        anchors: list[EntityRef],
        triple: dict[str, EntityRef],
        reference: ReferenceAnswer,
        branching: list[int],
        evidence_count: int,
        pair: PairEvidence,
        calls: list[CallRef | None],
        fields: dict[str, str],
    ) -> None:
        spec = TEMPLATES[template_id]
        query_id = QueryRecord.make_id(
            QueryClass.INVERSE, template_id, [a.id for a in anchors], self.index.data_version
        )
        if query_id in self.records:
            return
        kinds = pair.contradiction_kinds()
        unique_calls = {c.row_id: c for c in calls if c is not None}
        self.records[query_id] = QueryRecord(
            query_id=query_id,
            query_class=QueryClass.INVERSE,
            template_id=template_id,
            question=render(template_id, **fields),
            anchors=anchors,
            triple=triple,
            reference=reference,
            difficulty=Difficulty(
                hops=spec.hops,
                branching=branching,
                shape=spec.shape,
                evidence_count=evidence_count,
                contradictory=bool(kinds),
                contradiction_kinds=kinds,
            ),
            provenance=Provenance(
                data_version=self.index.data_version,
                api_version=self.index.api_version,
                calls=[unique_calls[k] for k in sorted(unique_calls)],
            ),
        )


def generate_candidates(index: CacheIndex) -> list[QueryRecord]:
    """Every templated query the cache supports, before stratified sampling."""
    b = _Builder(index)
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
    datatype_options = [datatype_label(i) for i in DATATYPE_LABELS]

    for drug in sorted(index.drugs.values(), key=lambda d: d.id):
        moa = moa_targets(drug)
        if not moa or not drug.indications:
            continue
        D = drug_ref(index, drug.id, drug.name)
        drug_call = index.drug_calls.get(drug.id)

        for ind in sorted(drug.indications.rows, key=lambda r: r.id):
            if ind.disease is None or not known_stage(ind.max_clinical_stage):
                continue
            X = disease_ref(ind.disease.id, ind.disease.name)
            crawled = [
                t
                for t in moa
                if (pe := index.evidence.get((t, X.id)))
                and pe.checked
                and pe.total_count is not None
            ]
            if not crawled:
                continue
            t_id = crawled[0]
            pe = index.evidence[(t_id, X.id)]
            T = target_ref(index, t_id, moa[t_id])
            triple = {"drug": D, "target": T, "disease": X}
            calls = [
                drug_call,
                index.known_drug_calls.get(t_id),
                index.pair_association_calls.get((t_id, X.id)),
                *pe.calls,
            ]
            moa_edge = edge(D.id, T.id, EdgeType.TARGETS, [f"moa:{D.id}:{T.id}"])
            drug_rows = pe.clinical_rows(drug_id=D.id)
            names = {"drug": D.name, "target": T.name, "disease": X.name}

            # hop 1, single edge: drug -> disease clinical stage
            b.add(
                "drug_stage",
                anchors=[D, X],
                triple=triple,
                reference=ReferenceAnswer(
                    answer_type=AnswerType.CATEGORY,
                    category=stage_label(ind.max_clinical_stage),
                    category_options=stage_options,
                    path=ReferencePath(
                        nodes=[D, T, X],
                        edges=[
                            moa_edge,
                            edge(
                                D.id,
                                X.id,
                                EdgeType.INDICATED_FOR,
                                [ind.id, *(r.id for r in drug_rows)],
                                max_clinical_stage=ind.max_clinical_stage,
                            ),
                        ],
                    ),
                    explanation=(
                        f"Open Targets records {D.name} reaching "
                        f"{stage_label(ind.max_clinical_stage)} for {X.name}."
                    ),
                ),
                branching=[drug.indications.count],
                evidence_count=len(drug_rows),
                pair=pe,
                calls=calls,
                fields={**names, "options": ", ".join(stage_options)},
            )

            # hop 1, single edge: drug -> target
            if len(moa) == 1:
                accepted = [target_ref(index, t, s) for t, s in moa.items()]
                b.add(
                    "drug_target",
                    anchors=[D, X],
                    triple=triple,
                    reference=ReferenceAnswer(
                        answer_type=AnswerType.ENTITY,
                        entities=accepted,
                        path=ReferencePath(
                            nodes=[D, *accepted],
                            edges=[
                                edge(D.id, a.id, EdgeType.TARGETS, [f"moa:{D.id}:{a.id}"])
                                for a in accepted
                            ],
                        ),
                        explanation=f"{D.name} acts on {', '.join(a.name for a in accepted)}.",
                    ),
                    branching=[len(moa)],
                    evidence_count=len(drug_rows),
                    pair=pe,
                    calls=calls,
                    fields=names,
                )

            # hop 2, single edge: which of the drug's targets is associated with the disease
            if 2 <= len(moa) <= 6 and all((t, X.id) in index.pair_association_queried for t in moa):
                linked = [
                    t
                    for t in moa
                    if (a := index.pair_associations.get((t, X.id))) is not None and a.score > 0
                ]
                if linked and len(linked) < len(moa):
                    accepted = [target_ref(index, t, moa[t]) for t in linked]
                    b.add(
                        "drug_disease_target",
                        anchors=[D, X],
                        triple=triple,
                        reference=ReferenceAnswer(
                            answer_type=AnswerType.ENTITY,
                            entities=accepted,
                            path=ReferencePath(
                                nodes=[D, *accepted, X],
                                edges=[
                                    e
                                    for a in accepted
                                    for e in (
                                        edge(D.id, a.id, EdgeType.TARGETS, [f"moa:{D.id}:{a.id}"]),
                                        edge(
                                            a.id,
                                            X.id,
                                            EdgeType.ASSOCIATED_WITH,
                                            score=index.pair_associations[(a.id, X.id)].score,
                                        ),
                                    )
                                ],
                            ),
                            explanation=(
                                f"Of {D.name}'s {len(moa)} targets, "
                                f"{', '.join(a.name for a in accepted)} are associated "
                                f"with {X.name}."
                            ),
                        ),
                        branching=[
                            len(moa),
                            max(
                                index.target_associations[t].count
                                if t in index.target_associations
                                else 0
                                for t in moa
                            ),
                        ],
                        evidence_count=sum(
                            index.evidence[(t, X.id)].total_count or 0
                            for t in linked
                            if (t, X.id) in index.evidence
                        ),
                        pair=pe,
                        calls=[
                            *calls,
                            *(index.pair_association_calls.get((t, X.id)) for t in moa),
                        ],
                        fields=names,
                    )

            # hop 1, single edge: dominant evidence datatype of the association
            pa = index.pair_associations.get((t_id, X.id))
            if pa is not None:
                ranked = sorted(
                    (c for c in pa.datatype_scores if c.id in DATATYPE_LABELS),
                    key=lambda c: -c.score,
                )
                if len(ranked) >= 2 and ranked[0].score - ranked[1].score >= TOP_SCORE_MARGIN:
                    b.add(
                        "target_disease_top_datatype",
                        anchors=[T, X],
                        triple=triple,
                        reference=ReferenceAnswer(
                            answer_type=AnswerType.CATEGORY,
                            category=datatype_label(ranked[0].id),
                            category_options=datatype_options,
                            path=ReferencePath(
                                nodes=[T, X],
                                edges=[
                                    edge(
                                        T.id,
                                        X.id,
                                        EdgeType.ASSOCIATED_WITH,
                                        score=pa.score,
                                        datatype_scores={c.id: c.score for c in pa.datatype_scores},
                                    )
                                ],
                            ),
                            explanation=(
                                f"{datatype_label(ranked[0].id)} scores {ranked[0].score:.3f}, "
                                f"ahead of {datatype_label(ranked[1].id)} ({ranked[1].score:.3f})."
                            ),
                        ),
                        branching=[len(pa.datatype_scores)],
                        evidence_count=pe.total_count or 0,
                        pair=pe,
                        calls=calls,
                        fields={**names, "options": ", ".join(datatype_options)},
                    )

            # hop 2, aggregation: drugs on the target with clinical evidence for the disease
            sets = clinical_drugs(pe, index)
            kd = index.known_drugs.get(t_id)
            kd_count = kd.count if kd else len(sets)
            clinical_all = pe.clinical_rows()
            if 2 <= len(sets) <= MAX_SET_SIZE:
                refs = [ref for ref, _, _ in sets.values()]
                b.add(
                    "target_disease_drugs",
                    anchors=[T, X],
                    triple=triple,
                    reference=ReferenceAnswer(
                        answer_type=AnswerType.ENTITY_SET,
                        entities=refs,
                        path=ReferencePath(nodes=[T, X, *refs], edges=_drug_edges(sets, T, X)),
                        explanation=f"{len(refs)} drugs acting on {T.name} have clinical "
                        f"evidence for {X.name}.",
                    ),
                    branching=[kd_count, len(sets)],
                    evidence_count=len(clinical_all),
                    pair=pe,
                    calls=calls,
                    fields=names,
                )
            late = {k: v for k, v in sets.items() if v[1] in PHASE_3_OR_LATER}
            if late and len(sets) <= 2 * MAX_SET_SIZE:
                b.add(
                    "target_disease_late_count",
                    anchors=[T, X],
                    triple=triple,
                    reference=ReferenceAnswer(
                        answer_type=AnswerType.COUNT,
                        count=len(late),
                        entities=[ref for ref, _, _ in late.values()],
                        path=ReferencePath(
                            nodes=[T, X, *(ref for ref, _, _ in late.values())],
                            edges=_drug_edges(late, T, X),
                        ),
                        explanation=f"{len(late)} of {len(sets)} drugs reached Phase 3 or later.",
                    ),
                    branching=[kd_count, len(sets)],
                    evidence_count=len(clinical_all),
                    pair=pe,
                    calls=calls,
                    fields=names,
                )

            # hop 3: drug -> target -> other drug -> disease
            if len(moa) == 1:
                siblings = {k: v for k, v in sets.items() if k != D.id}
                sibling_rows = pe.clinical_rows(exclude_drug=D.id)
                sibling_refs = [ref for ref, _, _ in siblings.values()]
                sibling_path = ReferencePath(
                    nodes=[D, T, X, *sibling_refs],
                    edges=[moa_edge, *_drug_edges(siblings, T, X)],
                )
                if siblings and len(siblings) <= 2 * MAX_SET_SIZE:
                    b.add(
                        "sibling_drug",
                        anchors=[D, X],
                        triple=triple,
                        reference=ReferenceAnswer(
                            answer_type=AnswerType.ENTITY,
                            entities=sibling_refs,
                            path=sibling_path,
                            explanation=f"{len(siblings)} other drugs on {T.name} have "
                            f"clinical evidence for {X.name}; any one is correct.",
                        ),
                        branching=[1, kd_count, len(siblings)],
                        evidence_count=len(sibling_rows),
                        pair=pe,
                        calls=calls,
                        fields=names,
                    )
                if 2 <= len(siblings) <= MAX_SET_SIZE:
                    b.add(
                        "sibling_drugs_all",
                        anchors=[D, X],
                        triple=triple,
                        reference=ReferenceAnswer(
                            answer_type=AnswerType.ENTITY_SET,
                            entities=sibling_refs,
                            path=sibling_path,
                            explanation=f"All {len(siblings)} other drugs on {T.name} with "
                            f"clinical evidence for {X.name}.",
                        ),
                        branching=[1, kd_count, len(siblings)],
                        evidence_count=len(sibling_rows),
                        pair=pe,
                        calls=calls,
                        fields=names,
                    )

        # hop 2, aggregation: argmax over the target's associated diseases
        if len(moa) == 1:
            t_id, symbol = next(iter(moa.items()))
            ta = index.target_associations.get(t_id)
            if ta is not None and len(ta.rows) >= 2:
                ranked_rows = sorted(ta.rows, key=lambda r: -r.score)
                top = ranked_rows[0]
                pe = index.evidence.get((t_id, top.disease.id))
                if (
                    top.score - ranked_rows[1].score >= TOP_SCORE_MARGIN
                    and pe is not None
                    and pe.checked
                ):
                    T = target_ref(index, t_id, symbol)
                    X_top = disease_ref(top.disease.id, top.disease.name)
                    b.add(
                        "drug_target_top_disease",
                        anchors=[D],
                        triple={"drug": D, "target": T, "disease": X_top},
                        reference=ReferenceAnswer(
                            answer_type=AnswerType.ENTITY,
                            entities=[X_top],
                            path=ReferencePath(
                                nodes=[D, T, X_top],
                                edges=[
                                    edge(D.id, T.id, EdgeType.TARGETS, [f"moa:{D.id}:{T.id}"]),
                                    edge(T.id, X_top.id, EdgeType.ASSOCIATED_WITH, score=top.score),
                                ],
                            ),
                            explanation=f"{X_top.name} has the top association score "
                            f"({top.score:.3f}) with {T.name}.",
                        ),
                        branching=[1, ta.count],
                        evidence_count=pe.total_count or 0,
                        pair=pe,
                        calls=[drug_call, index.target_association_calls.get(t_id), *pe.calls],
                        fields={"drug": D.name},
                    )

    return list(b.records.values())


def stratified_sample(
    records: list[QueryRecord], n: int, rng: random.Random, *, max_per_anchor: int = 3
) -> list[QueryRecord]:
    """Round-robin over difficulty cells so every populated cell is represented.

    Cells are (hops, shape, evidence bucket, branching bucket, contradictory). No non-disease
    anchor (drug or target) is used more than ``max_per_anchor`` times.
    """
    cells: dict[tuple, list[QueryRecord]] = defaultdict(list)
    for r in sorted(records, key=lambda r: r.query_id):
        cells[r.difficulty.cell()].append(r)
    for bucket in cells.values():
        rng.shuffle(bucket)
    order = sorted(cells, key=str)
    rng.shuffle(order)

    chosen: list[QueryRecord] = []
    anchor_use: Counter[str] = Counter()
    while len(chosen) < n:
        progressed = False
        for cell in order:
            bucket = cells[cell]
            while bucket:
                r = bucket.pop()
                keys = [a.id for a in r.anchors if a.entity_type != "disease"]
                if any(anchor_use[k] >= max_per_anchor for k in keys):
                    continue
                chosen.append(r)
                anchor_use.update(keys)
                progressed = True
                break
            if len(chosen) >= n:
                break
        if not progressed:
            break
    return chosen
