"""Done-criterion test against the live Open Targets Platform API."""

import httpx
import pytest

from otsearch_bench.env import CacheStore, Mode, OTAdapter, drift_report
from tests.conftest import forbid_network

pytestmark = pytest.mark.integration

EGFR = "ENSG00000146648"


def test_egfr_live_replay_refresh_and_drift(tmp_path):
    db = tmp_path / "ot.sqlite"

    # Run 1: LIVE — network + cache writes.
    with OTAdapter(db, mode=Mode.LIVE) as live:
        resolved = live.resolve_entity("EGFR", ["target"])
        assert resolved.id == EGFR
        diseases = live.get_associated_diseases(resolved.id, page_size=25)
        assert diseases.target.approved_symbol == "EGFR"
        assert diseases.count > 1000 and len(diseases.rows) == 25
        assert live.network_call_count == 3  # Meta, MapIds, AssociatedDiseases
        assert live.data_version
        live_version = live.data_version
        assoc_key = live.last_call.cache_key

    # Run 2: REPLAY — identical results, zero network calls (transport would raise).
    with OTAdapter(db, mode=Mode.REPLAY, transport=httpx.MockTransport(forbid_network)) as replay:
        resolved2 = replay.resolve_entity("EGFR", ["target"])
        diseases2 = replay.get_associated_diseases(resolved2.id, page_size=25)
        assert replay.network_call_count == 0
        assert replay.cache_hit_count == replay.call_count == 2
        assert (resolved2, diseases2) == (resolved, diseases)
        assert replay.data_version == live_version

    # Run 3: REFRESH — refetch appends a row and keeps the old one.
    with OTAdapter(db, mode=Mode.REFRESH) as refresh:
        refresh.get_associated_diseases(EGFR, page_size=25)
        assert refresh.network_call_count == 2  # Meta + AssociatedDiseases
    with CacheStore(db) as store:
        history = store.history(assoc_key)
        assert len(history) == 2
        unchanged = history[0].response_sha256 == history[1].response_sha256
        assert (assoc_key in {e.cache_key for e in drift_report(store)}) != unchanged

        # Simulate the environment changing under us: a later fetch returns different scores.
        drifted = history[-1].response
        drifted["data"]["target"]["associatedDiseases"]["rows"][0]["score"] = 0.0
        store.append(
            query=history[-1].query,
            variables=history[-1].variables,
            request=history[-1].request,
            response=drifted,
            endpoint=history[-1].endpoint,
            data_version="99.01",
            operation=history[-1].operation,
        )
        report = drift_report(store)
        entry = next(e for e in report if e.cache_key == assoc_key)
        assert entry.n_fetches == 3 and entry.n_distinct_hashes >= 2
        assert entry.data_versions[-1] == "99.01"
        assert store.history(assoc_key)[0] == history[0]  # old rows untouched


def test_all_typed_methods_parse_live_responses(tmp_path):
    with OTAdapter(tmp_path / "ot.sqlite", mode=Mode.LIVE) as ot:
        target = ot.get_target(EGFR)
        assert target.approved_symbol == "EGFR"

        nsclc = ot.resolve_entity("non-small cell lung carcinoma", ["disease"])
        assert nsclc.id == "MONDO_0005233"
        disease = ot.get_disease(nsclc.id)
        assert disease.name == "non-small cell lung carcinoma"

        osimertinib = ot.resolve_entity("osimertinib", ["drug"])
        drug = ot.get_drug(osimertinib.id)
        assert any(t.id == EGFR for m in drug.mechanisms_of_action.rows for t in m.targets)

        known = ot.get_known_drugs(EGFR)
        assert known.count == len(known.rows) > 0
        assert any(r.drug and r.drug.id == osimertinib.id for r in known.rows)

        targets = ot.get_associated_targets(nsclc.id, page_size=10)
        assert any(r.target.id == EGFR for r in targets.rows)

        evidence = ot.get_evidence(EGFR, nsclc.id, size=5)
        assert evidence.count > 0 and len(evidence.rows) == 5
        assert all(e.target.id == EGFR for e in evidence.rows)
