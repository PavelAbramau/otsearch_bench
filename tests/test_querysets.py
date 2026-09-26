import random
from types import SimpleNamespace

from otsearch_bench.env.models import (
    AssociatedDiseases,
    Disease,
    DiseaseAssociation,
    Drug,
    Evidence,
    KnownDrugs,
)
from otsearch_bench.queries.index import CacheIndex, PairEvidence
from otsearch_bench.queries.inverse import generate_candidates, stratified_sample
from otsearch_bench.queries.leakage import (
    ProbeResult,
    build_report,
    run_probe,
    score_probes,
    wilson_interval,
)
from otsearch_bench.queries.models import (
    AnswerShape,
    AnswerType,
    Difficulty,
    EntityRef,
    Provenance,
    QueryClass,
    QueryRecord,
    QuerySet,
    ReferenceAnswer,
)
from otsearch_bench.queries.scoring import ProbeAnswer, normalize, score_answer
from otsearch_bench.queries.unanswerable import verify_absence

T1, X1, X2 = "ENSG01", "EFO_1", "EFO_2"


def _evidence(
    eid,
    *,
    drug=None,
    stage=None,
    on_target=None,
    on_trait=None,
    stops=None,
    ds="clinical_precedence",
):
    return Evidence.model_validate(
        {
            "id": eid,
            "datasourceId": ds,
            "datatypeId": "clinical" if ds == "clinical_precedence" else "genetic_association",
            "score": 0.5,
            "target": {"id": T1, "approvedSymbol": "GENE1"},
            "disease": {"id": X1, "name": "disease one"},
            "drug": {"id": drug, "name": drug.upper()} if drug else None,
            "clinicalStage": stage,
            "directionOnTarget": on_target,
            "directionOnTrait": on_trait,
            "trialStopReasonCategories": stops,
        }
    )


def _drug(drug_id, stage="APPROVAL"):
    return Drug.model_validate(
        {
            "id": drug_id,
            "name": drug_id.upper(),
            "drugType": "Small molecule",
            "maximumClinicalStage": stage,
            "tradeNames": [{"label": f"Brand-{drug_id}", "source": "x"}],
            "mechanismsOfAction": {
                "rows": [
                    {
                        "mechanismOfAction": "inhibitor",
                        "targets": [{"id": T1, "approvedSymbol": "GENE1"}],
                    }
                ]
            },
            "indications": {
                "count": 1,
                "rows": [
                    {
                        "id": f"ind-{drug_id}",
                        "maxClinicalStage": stage,
                        "disease": {"id": X1, "name": "disease one"},
                    }
                ],
            },
        }
    )


def _index():
    pe = PairEvidence(
        T1,
        X1,
        total_count=40,
        directional_datasources=("clinical_precedence",),
        directional_count=3,
        rows={
            e.id: e
            for e in (
                _evidence("e1", drug="drugA", stage="APPROVAL"),
                _evidence("e2", drug="drugB", stage="PHASE_2"),
                _evidence("e3", drug="drugC", stage="PHASE_3"),
            )
        },
    )
    top = PairEvidence(T1, X2, total_count=5, directional_datasources=(), directional_count=0)
    assoc = {
        "score": 0.8,
        "disease": {"id": X1, "name": "disease one"},
        "datatypeScores": [{"id": "clinical", "score": 0.9}, {"id": "literature", "score": 0.4}],
    }
    return CacheIndex(
        data_version="26.06",
        drugs={"drugA": _drug("drugA")},
        known_drugs={
            T1: KnownDrugs.model_validate(
                {"target": {"id": T1, "approvedSymbol": "GENE1"}, "count": 3, "rows": []}
            )
        },
        target_associations={
            T1: AssociatedDiseases.model_validate(
                {
                    "target": {"id": T1, "approvedSymbol": "GENE1"},
                    "count": 60,
                    "page_index": 0,
                    "page_size": 25,
                    "rows": [
                        {"score": 0.9, "disease": {"id": X2, "name": "disease two"}},
                        {"score": 0.8, "disease": {"id": X1, "name": "disease one"}},
                    ],
                }
            )
        },
        pair_associations={(T1, X1): DiseaseAssociation.model_validate(assoc)},
        evidence={(T1, X1): pe, (T1, X2): top},
    )


def _record(
    qid,
    *,
    correct_category="Phase 3",
    cls=QueryClass.INVERSE,
    hops=1,
    shape=AnswerShape.SINGLE_EDGE,
    contradictory=False,
    evidence=5,
):
    return QueryRecord(
        query_id=qid,
        query_class=cls,
        template_id="drug_stage",
        question=f"question {qid}?",
        anchors=[EntityRef(id=f"D{qid}", entity_type="drug", name=f"drug {qid}")],
        reference=ReferenceAnswer(
            answer_type=AnswerType.CATEGORY, category=correct_category, explanation="because"
        ),
        difficulty=Difficulty(
            hops=hops,
            branching=[3],
            shape=shape,
            evidence_count=evidence,
            contradictory=contradictory,
        ),
        provenance=Provenance(data_version="26.06"),
    )


def test_queryset_round_trip(tmp_path):
    qs = QuerySet(
        name="x",
        query_class=QueryClass.INVERSE,
        data_version="26.06",
        generator_version="t",
        records=[_record("a"), _record("b")],
    )
    loaded = QuerySet.from_jsonl(qs.to_jsonl(tmp_path / "x.jsonl"))
    assert loaded == qs
    assert loaded.counts_by_stratum() == {"inverse_constructed | hops=1 | single_edge": 2}


def test_difficulty_buckets():
    d = Difficulty(
        hops=2,
        branching=[3, 80],
        shape=AnswerShape.AGGREGATION,
        evidence_count=11,
        contradictory=True,
    )
    assert (d.evidence_bucket, d.branching_bucket) == ("mid", "high")
    assert d.cell() == (2, "aggregation", "mid", "high", True)


def test_contradiction_kinds():
    pe = PairEvidence(T1, X1, directional_count=2)
    pe.rows = {
        "g1": _evidence("g1", on_target="GoF", on_trait="risk", ds="gwas"),
        "g2": _evidence("g2", on_target="GoF", on_trait="protect", ds="gwas"),
    }
    assert pe.contradiction_kinds() == ["direction_of_effect_conflict"]
    pe.rows = {
        "g1": _evidence("g1", on_target="GoF", on_trait="risk", ds="gwas"),
        "g2": _evidence("g2", on_target="LoF", on_trait="protect", ds="gwas"),
    }
    assert pe.contradiction_kinds() == []
    pe.rows = {
        "c1": _evidence("c1", drug="d", stage="APPROVAL"),
        "c2": _evidence("c2", drug="e", stage="PHASE_3", stops=["Negative"]),
    }
    assert pe.contradiction_kinds() == ["clinical_outcome_conflict"]


def test_generate_candidates_builds_every_supported_template():
    records = generate_candidates(_index())
    assert {r.template_id for r in records} == {
        "drug_stage", "drug_target", "target_disease_top_datatype", "target_disease_drugs",
        "target_disease_late_count", "sibling_drug", "sibling_drugs_all", "drug_target_top_disease",
    }  # fmt: skip
    by_template = {r.template_id: r for r in records}
    assert len({r.query_id for r in records}) == len(records)
    assert by_template["drug_stage"].reference.category == "Approval"
    assert (
        by_template["target_disease_late_count"].reference.count == 2
    )  # drugA approval, drugC phase 3
    siblings = {e.id for e in by_template["sibling_drugs_all"].reference.entities}
    assert siblings == {"drugB", "drugC"}
    assert by_template["sibling_drug"].difficulty.hops == 3
    assert by_template["drug_target_top_disease"].reference.entities[0].id == X2
    assert by_template["target_disease_top_datatype"].reference.category == "clinical"
    assert all(r.reference.path is not None for r in records)


def test_stratified_sample_covers_every_cell():
    records = [_record(f"a{i}", evidence=500) for i in range(10)] + [
        _record(f"b{i}", evidence=0) for i in range(2)
    ]
    sample = stratified_sample(records, 4, random.Random(0))
    assert len(sample) == 4
    assert {r.difficulty.evidence_bucket for r in sample} == {"high", "none"}


class FakeAdapter:
    def __init__(self, *, associated=(), indications=(), sibling_diseases=()):
        self.call_log = []
        self._associated, self._indications, self._siblings = (
            associated,
            indications,
            sibling_diseases,
        )

    def _log(self, op):
        self.call_log.append(
            SimpleNamespace(
                operation=op, cache_key=op, row_id=len(self.call_log), response_sha256="h"
            )
        )

    def get_disease(self, efo_id):
        self._log("Disease")
        return Disease.model_validate({"id": efo_id, "name": "x", "descendants": ["EFO_CHILD"]})

    def get_drug(self, chembl_id):
        self._log("Drug")
        drug = _drug(chembl_id).model_dump(by_alias=True)
        drug["indications"] = {
            "count": len(self._indications),
            "rows": [
                {"id": f"i{d}", "maxClinicalStage": "PHASE_2", "disease": {"id": d}}
                for d in self._indications
            ],
        }
        return Drug.model_validate(drug)

    def get_all_associated_diseases(self, ensembl_id, *, enable_indirect=False):
        self._log("AssociatedDiseases")
        rows = [{"score": 0.1, "disease": {"id": d}} for d in self._associated]
        return AssociatedDiseases.model_validate(
            {
                "target": {"id": ensembl_id},
                "count": len(rows),
                "page_index": 0,
                "page_size": len(rows),
                "rows": rows,
            }
        )

    def get_evidence(self, ensembl_id, efo_id, size=1):
        self._log("Evidence")
        return SimpleNamespace(count=0)

    def get_known_drugs(self, ensembl_id):
        self._log("KnownDrugs")
        rows = [
            {
                "id": "r",
                "maxClinicalStage": "PHASE_1",
                "drug": {"id": "other"},
                "diseases": [{"disease": {"id": d}}],
            }
            for d in self._siblings
        ]
        return KnownDrugs.model_validate(
            {"target": {"id": ensembl_id}, "count": len(rows), "rows": rows}
        )


def test_verify_absence_accepts_and_rejects():
    ok = verify_absence(FakeAdapter(associated=["EFO_OTHER"]), "drugA", T1, "EFO_9")
    assert ok.verified and len(ok.calls) == 5
    assert ok.traversed["association_rows"] == 1
    assert not verify_absence(FakeAdapter(associated=["EFO_CHILD"]), "drugA", T1, "EFO_9").verified
    assert not verify_absence(FakeAdapter(indications=["EFO_9"]), "drugA", T1, "EFO_9").verified
    rejected = verify_absence(FakeAdapter(sibling_diseases=["EFO_CHILD"]), "drugA", T1, "EFO_9")
    assert rejected.reason == "reject: hop 2 other drug on target"


def _answer(**kw):
    base = {
        "short_answer": "",
        "entities": [],
        "category": None,
        "count": None,
        "identifier": None,
        "no_evidence": False,
    }
    return ProbeAnswer(**{**base, **kw})


def test_scoring_rules():
    record = generate_candidates(_index())
    by_template = {r.template_id: r for r in record}
    assert normalize("Phase III (pivotal)") == normalize("Phase 3")
    assert score_answer(by_template["drug_stage"], _answer(category="approval")).correct
    assert score_answer(by_template["drug_target"], _answer(entities=["gene1"])).correct
    assert (
        score_answer(by_template["sibling_drug"], _answer(entities=["Brand-drugB"])).correct
        is False
    )  # alias only on drugA
    resolver = lambda name, etype: {"drugB"} if name == "brandb" else set()  # noqa: E731
    assert score_answer(by_template["sibling_drug"], _answer(entities=["brandb"]), resolver).correct
    s = score_answer(by_template["sibling_drugs_all"], _answer(entities=["DRUGB", "aspirin"]))
    assert (round(s.f1, 2), s.correct) == (0.5, True)
    assert score_answer(by_template["target_disease_late_count"], _answer(count=2)).correct
    assert not score_answer(by_template["drug_stage"], _answer(no_evidence=True)).correct
    unanswerable = by_template["drug_stage"].model_copy(
        update={
            "reference": ReferenceAnswer(
                answer_type=AnswerType.NO_EVIDENCE, no_evidence=True, explanation="none"
            )
        }
    )
    assert score_answer(unanswerable, _answer(no_evidence=True)).correct


class FakeClient:
    model = "fake-model"

    def __init__(self):
        self.calls = 0

    def ask(self, system, question):
        self.calls += 1
        category = "Phase 3" if question.endswith(("a0?", "a1?")) else "Phase 1"
        return ProbeResult(
            stop_reason="end_turn",
            answer=_answer(short_answer=category, category=category),
            input_tokens=10,
            output_tokens=5,
        )


def test_probe_caches_responses_and_report_flags_contamination(tmp_path):
    records = [_record(f"a{i}") for i in range(4)] + [_record(f"z{i}", hops=2) for i in range(5)]
    client = FakeClient()
    responses = tmp_path / "responses.jsonl"
    probes = run_probe(records, client, responses, workers=2, log=lambda m: None)
    assert client.calls == 9 and len(probes) == 9
    run_probe(records, client, responses, workers=2, log=lambda m: None)
    assert client.calls == 9  # nothing re-asked

    report = build_report(
        score_probes(records, probes, None),
        model="fake-model",
        prompt_version="v1",
        data_version="26.06",
    )
    assert "**GATE: FAIL**" in report
    assert "| inverse_constructed | hops=1 | single_edge | 4 | 0.50 |" in report
    assert "**CONTAMINATED** (n<5)" in report
    assert "| inverse_constructed | hops=2 | single_edge | 5 | 0.00 |" in report
    lo, hi = wilson_interval(2, 4)
    assert 0.1 < lo < 0.5 < hi < 0.95
