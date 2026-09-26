import dataclasses
import json

import httpx
import pytest

from otsearch_bench.env import Mode, OTAdapter
from otsearch_bench.search import (
    ACTION_ADAPTER,
    EdgeType,
    GetEvidence,
    PolicyViolationError,
    Runner,
    Trajectory,
)
from otsearch_bench.search.stubs import EGFR, NSCLC, stub_agent
from tests.conftest import forbid_network

QUESTION = "What evidence links EGFR to non-small cell lung carcinoma?"


def _canned(fake):
    fake.bodies["MapIds"] = {
        "data": {
            "mapIds": {
                "mappings": [
                    {
                        "term": "EGFR",
                        "hits": [{"id": EGFR, "entity": "target", "name": "EGFR", "score": 1}],
                    }
                ]
            }
        }
    }
    fake.bodies["AssociatedDiseases"] = {
        "data": {
            "target": {
                "id": EGFR,
                "approvedSymbol": "EGFR",
                "associatedDiseases": {
                    "count": 2,
                    "rows": [
                        {
                            "score": 0.85,
                            "disease": {"id": NSCLC, "name": "non-small cell lung carcinoma"},
                            "datatypeScores": [{"id": "clinical", "score": 0.99}],
                            "datasourceScores": [],
                        },
                        {
                            "score": 0.77,
                            "disease": {"id": "MONDO_0005061", "name": "lung adenocarcinoma"},
                            "datatypeScores": [],
                            "datasourceScores": [],
                        },
                    ],
                },
            }
        }
    }
    fake.bodies["Evidence"] = {
        "data": {
            "target": {
                "id": EGFR,
                "evidences": {
                    "count": 2,
                    "cursor": None,
                    "rows": [
                        {
                            "id": f"ev{i}",
                            "datasourceId": "europepmc",
                            "datatypeId": "literature",
                            "score": score,
                            "target": {"id": EGFR, "approvedSymbol": "EGFR"},
                            "disease": {"id": NSCLC, "name": "non-small cell lung carcinoma"},
                            "drug": None,
                        }
                        for i, score in enumerate([1.0, 0.4])
                    ],
                },
            }
        }
    }


@pytest.fixture
def live_adapter(tmp_path, fake_ot):
    _canned(fake_ot)
    with OTAdapter(
        tmp_path / "c.sqlite", mode=Mode.LIVE, transport=fake_ot.transport, rate_limit_per_s=None
    ) as ot:
        yield ot


def test_stub_agent_produces_valid_trajectory(live_adapter):
    agent = stub_agent()
    traj = Runner(live_adapter).run(agent, QUESTION)

    assert [s.phase for s in traj.steps] == ["formulate", "expand", "expand", "expand"]
    assert [s.index for s in traj.steps] == [0, 1, 2, 3]
    assert traj.steps[0].formulation.anchor_mentions[0].mention == "EGFR"
    for step, action in zip(traj.steps[1:], agent.expansion.sequence, strict=True):
        assert [a.action_id for a in step.proposed_actions] == [action.action_id]
        assert step.admitted_actions == step.proposed_actions
        assert [o.ok for o in step.observations] == [True]
        assert step.cache_hit_flags == [False]
        assert step.cost_delta.calls == 1
    assert [s.stopping.stop for s in traj.steps[1:]] == [False, False, True]
    assert traj.stopping.reason == "reached step 3" and traj.stopping.source == "policy"

    assert traj.total_usage.calls == 3
    assert traj.steps[-1].budget.calls_used == 3
    fingerprints = [s.state_fingerprint_after for s in traj.steps[1:]]
    assert len(set(fingerprints)) == 3
    assert traj.final_state_fingerprint == fingerprints[-1]
    assert traj.answer.cited_evidence_ids == ["ev0", "ev1"]

    edges = {(u, v, d["edge_type"]) for u, v, _, d in traj.final_state["edges"]}
    assert (EGFR, NSCLC, EdgeType.HAS_EVIDENCE.value) in edges
    assert (EGFR, NSCLC, EdgeType.ASSOCIATED_WITH.value) in edges
    assert [a["entity_id"] for a in traj.final_state["anchors"]] == [EGFR]

    v = traj.versions
    assert set(v["components"]) == {
        "query_formulation", "expansion", "pruning", "reformulation", "stopping",
        "synthesis", "budget",
    }  # fmt: skip
    assert v["adapter"]["data_version"] == "26.06"
    assert {"git_sha", "git_dirty", "package_version"} <= set(v["code"])
    assert v["model"] == {"model": "none", "temperature": 0.0, "top_p": 1.0, "seed": 0}
    assert traj.data_versions_seen == ["26.06"]


def test_trajectory_round_trips_through_jsonl(live_adapter, tmp_path):
    traj = Runner(live_adapter).run(stub_agent(), QUESTION)
    path = traj.to_jsonl(tmp_path)
    assert path.name == f"{traj.agent_id}__{traj.query.query_id}__seed0.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert [r["record"] for r in records] == ["header", "step", "step", "step", "step", "result"]
    loaded = Trajectory.from_jsonl(path)
    assert loaded == traj
    assert loaded.model_dump(mode="json") == traj.model_dump(mode="json")
    assert isinstance(loaded.steps[3].admitted_actions[0], GetEvidence)


def test_replay_reproduces_the_trajectory_without_network(live_adapter, tmp_path):
    first = Runner(live_adapter).run(stub_agent(), QUESTION)
    with OTAdapter(
        live_adapter.cache, mode=Mode.REPLAY, transport=httpx.MockTransport(forbid_network)
    ) as replay:
        second = Runner(replay).run(stub_agent(), QUESTION)
        assert replay.network_call_count == 0
    assert [s.state_fingerprint_after for s in second.steps] == [
        s.state_fingerprint_after for s in first.steps
    ]
    assert all(flag for s in second.steps for flag in s.cache_hit_flags)
    assert second.answer == first.answer


def test_action_ids_are_stable_and_ignore_rationale():
    a = GetEvidence(ensembl_id=EGFR, efo_id=NSCLC, size=5)
    b = GetEvidence(ensembl_id=EGFR, efo_id=NSCLC, size=5, rationale="because")
    assert a.action_id == b.action_id
    assert ACTION_ADAPTER.validate_python(a.model_dump()) == a


def test_policy_that_calls_adapter_is_rejected(live_adapter):
    agent = stub_agent()

    @dataclasses.dataclass(frozen=True)
    class Cheating:
        name: str = "cheat"
        version: str = "1"

        def propose(self, state):
            live_adapter.resolve_entity("EGFR", ["target"])
            return []

    with pytest.raises(PolicyViolationError, match="called the adapter"):
        Runner(live_adapter).run(dataclasses.replace(agent, expansion=Cheating()), QUESTION)


def test_policy_that_mutates_state_is_rejected(live_adapter):
    @dataclasses.dataclass(frozen=True)
    class Mutating:
        name: str = "mutate"
        version: str = "1"

        def prune(self, state):
            state.graph.add_node("X")
            return None

    agent = dataclasses.replace(stub_agent(), pruning=Mutating())
    with pytest.raises(PolicyViolationError, match="mutated SearchState"):
        Runner(live_adapter).run(agent, QUESTION)


def test_agent_rejects_components_missing_a_role_or_version():
    with pytest.raises(TypeError):
        dataclasses.replace(stub_agent(), stopping=object())

    @dataclasses.dataclass(frozen=True)
    class Unversioned:
        name: str = "x"
        version: str = ""

        def decide(self, state): ...

    with pytest.raises(ValueError):
        dataclasses.replace(stub_agent(), stopping=Unversioned())
