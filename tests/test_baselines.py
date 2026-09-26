"""Baseline policies, reference systems and the factorial grid on a synthetic OT world."""

import json

import httpx
import pytest

from otsearch_bench.env import Mode, OTAdapter
from otsearch_bench.env.errors import CacheMissError
from otsearch_bench.search import Runner, Trajectory
from otsearch_bench.search.baselines.agents import (
    AgentContext,
    factorial_grid,
    one_shot_dump,
    reference_systems,
)
from otsearch_bench.search.baselines.expansion import BreadthFirstExpansion
from otsearch_bench.search.baselines.pruning import RecencyWindow, SubgraphSummarize
from otsearch_bench.search.baselines.roles import extract_mentions
from otsearch_bench.search.baselines.stopping import NoNewEdgesStopping
from otsearch_bench.search.graph import EdgeType, FrontierItem
from otsearch_bench.search.llm import CachedAnthropicClient, FakeLLMClient
from otsearch_bench.search.state import Anchor, Budget, SearchState
from tests.conftest import META_BODY, forbid_network

NUMBERS = ["zero", "one", "two", "three", "four", "five"]
TARGETS = {f"ENSG{i:04d}": f"GENE{i}" for i in range(6)}
DISEASES = {f"EFO_{i:04d}": f"disease {NUMBERS[i]}" for i in range(6)}
DRUGS = {f"CHEMBL{i}": f"DRUG{NUMBERS[i].upper()}" for i in range(4)}


def _t(i):
    return f"ENSG{i:04d}"


def _d(i):
    return f"EFO_{i:04d}"


ASSOC = {_t(i): [(_d((i + j) % 6), round(0.9 - 0.2 * j, 2)) for j in range(3)] for i in range(6)}
DRUG_TARGET = {f"CHEMBL{i}": _t(i) for i in range(4)}
DRUG_DISEASE = {f"CHEMBL{i}": [_d(i), _d((i + 1) % 6)] for i in range(4)}


def _hit(entity_id):
    kind = "target" if entity_id in TARGETS else "disease" if entity_id in DISEASES else "drug"
    name = {**TARGETS, **DISEASES, **DRUGS}[entity_id]
    return {"id": entity_id, "entity": kind, "name": name, "score": 1.0, "description": None}


NAME_INDEX = {n.lower(): i for i, n in {**TARGETS, **DISEASES, **DRUGS}.items()}


def world(payload):
    op, v = payload.get("operationName"), payload.get("variables") or {}
    t_ref = lambda t: {"id": t, "approvedSymbol": TARGETS[t]}  # noqa: E731
    d_ref = lambda d: {"id": d, "name": DISEASES[d]}  # noqa: E731
    if op == "Meta":
        return META_BODY
    if op == "MapIds":
        hits = [_hit(NAME_INDEX[t.lower()]) for t in v["terms"] if t.lower() in NAME_INDEX]
        return {"data": {"mapIds": {"mappings": [{"term": v["terms"][0], "hits": hits}]}}}
    if op == "Search":
        return {"data": {"search": {"total": 0, "hits": []}}}
    if op == "AssociatedDiseases":
        t = v["ensemblId"]
        rows = [
            {"score": s, "disease": d_ref(d), "datatypeScores": [], "datasourceScores": []}
            for d, s in ASSOC[t]
            if not v.get("Bs") or d in v["Bs"]
        ]
        return {
            "data": {
                "target": {**t_ref(t), "associatedDiseases": {"count": len(rows), "rows": rows}}
            }
        }
    if op == "AssociatedTargets":
        d = v["efoId"]
        rows = [
            {"score": s, "target": t_ref(t), "datatypeScores": [], "datasourceScores": []}
            for t, pairs in ASSOC.items()
            for dd, s in pairs
            if dd == d
        ]
        return {
            "data": {
                "disease": {**d_ref(d), "associatedTargets": {"count": len(rows), "rows": rows}}
            }
        }
    if op == "KnownDrugs":
        t = v["ensemblId"]
        rows = [
            {
                "id": f"kd-{c}",
                "maxClinicalStage": "PHASE_3",
                "drug": {"id": c, "name": DRUGS[c], "drugType": "Small molecule"},
                "diseases": [
                    {"diseaseFromSource": None, "disease": d_ref(d)} for d in DRUG_DISEASE[c]
                ],
            }
            for c, tt in DRUG_TARGET.items()
            if tt == t
        ]
        return {
            "data": {
                "target": {
                    **t_ref(t),
                    "drugAndClinicalCandidates": {"count": len(rows), "rows": rows},
                }
            }
        }
    if op == "Drug":
        c = v["chemblId"]
        return {
            "data": {
                "drug": {
                    "id": c,
                    "name": DRUGS[c],
                    "drugType": "Small molecule",
                    "maximumClinicalStage": "APPROVAL",
                    "mechanismsOfAction": {
                        "rows": [
                            {"mechanismOfAction": "inhibitor", "targets": [t_ref(DRUG_TARGET[c])]}
                        ]
                    },
                    "indications": {
                        "count": 2,
                        "rows": [
                            {
                                "id": f"ind-{c}-{d}",
                                "maxClinicalStage": "PHASE_2",
                                "disease": d_ref(d),
                            }
                            for d in DRUG_DISEASE[c]
                        ],
                    },
                }
            }
        }
    if op == "Evidence":
        t, d = v["ensemblId"], v["efoIds"][0]
        linked = any(dd == d for dd, _ in ASSOC.get(t, []))
        rows = [
            {
                "id": f"ev-{t}-{d}-{k}",
                "datasourceId": "europepmc",
                "datatypeId": "literature",
                "score": 0.5,
                "target": t_ref(t),
                "disease": d_ref(d),
                "drug": None,
            }
            for k in range(2 if linked else 0)
        ]
        return {
            "data": {
                "target": {"id": t, "evidences": {"count": len(rows), "cursor": None, "rows": rows}}
            }
        }
    return {"data": {"meta": {"name": "raw"}}}


def transport():
    def handler(request):
        return httpx.Response(200, json=world(json.loads(request.content)))

    return httpx.MockTransport(handler)


QUESTIONS = [
    "Which drugs acting on GENE1 have clinical evidence for disease two in Open Targets? List all of them.",
    "Name one other drug that acts on the same molecular target as DRUGZERO and has clinical evidence for disease one in Open Targets.",
    "How many distinct drugs acting on GENE0 have reached Phase 3 or later in clinical development for disease zero, according to Open Targets?",
    "Which evidence data type contributes the highest score to the Open Targets association between GENE2 and disease three? Answer with one of: literature, clinical.",
    "DRUGTWO has been in clinical development for disease two. Which human gene encodes its molecular target?",
    "According to Open Targets, what is the highest clinical development stage that DRUGTHREE has reached for disease four? Answer with one of: Phase 1, Phase 2.",
    "Among all diseases directly associated with the molecular target of DRUGONE in Open Targets, which has the highest overall association score?",
    "List all other drugs that act on the same molecular target as DRUGONE and have clinical evidence for disease two in Open Targets.",
    "Which drugs acting on GENE4 have clinical evidence for disease five in Open Targets? List all of them.",
    "Which drugs acting on GENE5 have clinical evidence for disease zero in Open Targets? List all of them.",
]


def fake_llm_responder(prompt_version, user, output):
    if prompt_version == "expansion_rank_v1":
        n = user.count("\n[")
        return {"scores": [{"index": i, "score": 1.0 / (i + 1)} for i in range(n)]}
    if prompt_version == "stopping_self_assess_v1":
        return {"sufficient": "evidence_id=ev-" in user, "reason": "fake"}
    if prompt_version == "synthesis_v1":
        ids = sorted({part.split()[0] for part in user.split("evidence_id=")[1:]})
        return {
            "text": "fake answer",
            "no_evidence": not ids,
            "entities": [],
            "category": None,
            "count": None,
            "claims": [{"text": "claim", "evidence_ids": ids[:2]}] if ids else [],
        }
    if prompt_version == "oneshot_graphql_v1":
        return {"query": "query Raw { meta { name } }", "rationale": "fake"}
    raise AssertionError(prompt_version)


def all_agents(ctx):
    return [cell.agent for cell in factorial_grid("score-greedy", ctx)] + reference_systems(ctx)


def test_every_grid_cell_and_reference_system_replays_on_10_queries(tmp_path):
    ctx = AgentContext(llm=FakeLLMClient(fake_llm_responder), call_budget=12, page_size=5)
    agents = all_agents(ctx)
    assert len(agents) == 1 + 4 + 2 + 2 + 2  # baseline, expansion, stopping, pruning variants, refs
    db = tmp_path / "ot.sqlite"
    live_runs = {}
    with OTAdapter(db, mode=Mode.LIVE, transport=transport(), rate_limit_per_s=None) as live:
        runner = Runner(live)
        for agent in agents:
            for q in QUESTIONS:
                live_runs[(agent.agent_id, q)] = runner.run(agent, q)
    with OTAdapter(db, mode=Mode.REPLAY, transport=httpx.MockTransport(forbid_network)) as replay:
        runner = Runner(replay)
        for agent in agents:
            for q in QUESTIONS:
                traj = runner.run(agent, q)
                live_traj = live_runs[(agent.agent_id, q)]
                assert [s.state_fingerprint_after for s in traj.steps] == [
                    s.state_fingerprint_after for s in live_traj.steps
                ], agent.name
                assert traj.stopping.stop and traj.answer == live_traj.answer
                assert all(flag for s in traj.steps for flag in s.cache_hit_flags)
                assert Trajectory.from_jsonl(traj.to_jsonl(tmp_path)) == traj
        assert replay.network_call_count == 0


def test_grid_cells_differ_from_baseline_in_exactly_one_role():
    ctx = AgentContext(llm=FakeLLMClient(fake_llm_responder))
    grid = factorial_grid("score-greedy", ctx)
    base = grid[0].agent.component_versions()
    assert grid[0].role is None
    for cell in grid[1:]:
        diff = [r for r, v in cell.agent.component_versions().items() if v != base[r]]
        assert diff == [cell.role], cell.label
    assert len({c.agent.agent_id for c in grid}) == len(grid)


def test_one_shot_dump_sweeps_two_hops_then_stops(tmp_path):
    with OTAdapter(tmp_path / "ot.sqlite", transport=transport(), rate_limit_per_s=None) as ot:
        traj = Runner(ot).run(one_shot_dump(AgentContext(page_size=5)), QUESTIONS[0])
    expand = [s for s in traj.steps if s.phase == "expand"]
    assert len(expand[0].admitted_actions) > 1  # a whole level at once
    assert traj.stopping.reason == "2-hop neighbourhood fetched"


def test_extract_mentions_from_templates():
    assert extract_mentions(QUESTIONS[0]) == ["GENE1", "disease two"]
    assert extract_mentions(QUESTIONS[2]) == ["GENE0", "disease zero"]
    assert extract_mentions(QUESTIONS[3]) == ["GENE2", "disease three"]
    assert extract_mentions(
        "According to Open Targets, what is the highest clinical development stage that "
        "ruxolitinib has reached for type 2 diabetes mellitus? Answer with one of: Phase 1."
    ) == ["ruxolitinib", "type 2 diabetes mellitus"]


def _state_with_edges():
    state = SearchState(question="q", seed=0, budget=Budget(max_calls=10))
    state.add_node("ENSG0000", "GENE0")
    state.add_anchor(
        Anchor(
            mention="GENE0",
            entity_id="ENSG0000",
            entity_type="target",
            name="GENE0",
            method="mapIds",
        )
    )
    for i, score in enumerate([0.9, 0.5, 0.1]):
        state.add_node(_d(i), DISEASES[_d(i)])
        state.add_edge(
            "ENSG0000", _d(i), EdgeType.ASSOCIATED_WITH, evidence_id=f"a{i}", score=score
        )
    return state


def test_pruning_policies():
    state = _state_with_edges()
    summarize = SubgraphSummarize(top_m=2).prune(state)
    assert [r.target for r in summarize.remove_edges] == [_d(2)]
    assert summarize.remove_nodes == [_d(2)]
    state.step = 5
    stale = RecencyWindow(k=3).prune(state).drop_frontier
    assert stale and all(state.frontier_added_step[i] == 0 for i in stale)


def test_no_new_edges_and_breadth_first_sweep():
    state = _state_with_edges()
    state.step, state.edge_count_history = 3, [3, 3, 3]
    assert NoNewEdgesStopping(patience=2).decide(state).stop
    state.edge_count_history = [1, 2, 3]
    assert not NoNewEdgesStopping(patience=2).decide(state).stop
    sweep = BreadthFirstExpansion(sweep=True).propose(state)
    anchor_level = [
        i for i in state.frontier if i.node == "ENSG0000" or i.node.startswith("ENSG0000|")
    ]
    assert len(sweep) == len({a.action_id for a in sweep}) >= len(anchor_level) - 1
    assert FrontierItem(node=f"ENSG0000|{_d(0)}", edge_type=EdgeType.HAS_EVIDENCE) in state.frontier


class _FakeAnthropic:
    def __init__(self):
        self.calls = 0
        self.messages = self

    def parse(self, *, model, max_tokens, system, messages, output_format):
        self.calls += 1
        parsed = output_format(sufficient=True, reason="ok")

        class Usage:
            input_tokens, output_tokens = 100, 20

        class Response:
            stop_reason, usage, parsed_output = "end_turn", Usage, parsed

        Response.model = model
        return Response


def test_cached_llm_client_live_then_replay(tmp_path):
    from otsearch_bench.search.baselines.stopping import Sufficiency

    api = _FakeAnthropic()
    kwargs = {
        "prompt_version": "stopping_self_assess_v1",
        "system": "s",
        "user": "u",
        "output": Sufficiency,
    }
    live = CachedAnthropicClient("claude-sonnet-5", cache=tmp_path / "llm.sqlite", client=api)
    assert live.structured(**kwargs).sufficient
    assert live.structured(**kwargs).sufficient
    assert api.calls == 1 and live.usage.llm_calls == 2 and live.usage.llm_cache_hits == 1
    assert live.usage.usd == pytest.approx((100 * 2.0 + 20 * 10.0) / 1e6)
    replay = CachedAnthropicClient(
        "claude-sonnet-5", cache=live.cache, mode=Mode.REPLAY, client=api
    )
    assert replay.structured(**kwargs).sufficient and api.calls == 1
    with pytest.raises(CacheMissError):
        replay.structured(**{**kwargs, "user": "other"})
