"""Baseline ExpansionPolicy implementations.

All map frontier items to actions the same way and differ only in how they order the frontier,
so any one can replace another in an Agent without touching the other roles.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from pydantic import BaseModel

from otsearch_bench.prompts import load_prompt, prompt_fingerprint
from otsearch_bench.search.actions import (
    Action,
    GetAssociatedDiseases,
    GetAssociatedTargets,
    GetDrug,
    GetEvidence,
    GetKnownDrugs,
)
from otsearch_bench.search.graph import EdgeType, FrontierItem, NodeType, split_pair
from otsearch_bench.search.llm import LLMClient
from otsearch_bench.search.state import SearchState, node_type_for

UNREACHED_DEPTH = 10**6


def frontier_action(item: FrontierItem, *, page_size: int = 25) -> Action | None:
    """The single API call that expands a frontier item."""
    pair = split_pair(item.node)
    if item.edge_type is EdgeType.HAS_EVIDENCE:
        return GetEvidence(ensembl_id=pair[0], efo_id=pair[1], size=page_size) if pair else None
    match node_type_for(item.node), item.edge_type:
        case NodeType.TARGET, EdgeType.ASSOCIATED_WITH:
            return GetAssociatedDiseases(ensembl_id=item.node, page_size=page_size)
        case NodeType.DISEASE, EdgeType.ASSOCIATED_WITH:
            return GetAssociatedTargets(efo_id=item.node, page_size=page_size)
        case NodeType.TARGET, EdgeType.TARGETS:
            return GetKnownDrugs(ensembl_id=item.node)
        case NodeType.DRUG, EdgeType.TARGETS | EdgeType.INDICATED_FOR:
            return GetDrug(chembl_id=item.node)
    return None


def item_nodes(item: FrontierItem) -> tuple[str, ...]:
    return split_pair(item.node) or (item.node,)


def describe(item: FrontierItem, state: SearchState) -> str:
    def label(node: str) -> str:
        name = state.graph.nodes[node].get("name") if node in state.graph else None
        return f"{name} ({node})" if name else node

    nodes = item_nodes(item)
    if item.edge_type is EdgeType.HAS_EVIDENCE:
        return f"fetch evidence records linking {label(nodes[0])} to {label(nodes[1])}"
    kind = node_type_for(item.node).value
    verb = {
        EdgeType.ASSOCIATED_WITH: "list associated "
        + ("diseases" if kind == "target" else "targets"),
        EdgeType.TARGETS: "list drugs acting on it" if kind == "target" else "get its targets",
        EdgeType.INDICATED_FOR: "get its clinical indications",
    }[item.edge_type]
    return f"{kind} {label(nodes[0])}: {verb}"


@dataclass(frozen=True)
class FrontierExpansion:
    """Shared propose(): first ``batch`` distinct actions in the policy's frontier order."""

    batch: int = 1
    page_size: int = 25

    def order(self, state: SearchState) -> list[FrontierItem]:
        return list(state.frontier)

    def propose(self, state: SearchState) -> list[Action]:
        actions: dict[str, Action] = {}
        for item in self.order(state):
            action = frontier_action(item, page_size=self.page_size)
            if action is not None:
                actions.setdefault(action.action_id, action)
            if len(actions) >= self.batch:
                break
        return list(actions.values())


@dataclass(frozen=True)
class RandomWalkExpansion(FrontierExpansion):
    """Uniform over the frontier; the RNG is derived from (seed, step) so replays match."""

    name: str = "random-walk"
    version: str = "1"

    def order(self, state: SearchState) -> list[FrontierItem]:
        items = list(state.frontier)
        random.Random(f"{state.seed}:{state.step}").shuffle(items)
        return items


@dataclass(frozen=True)
class DegreeGreedyExpansion(FrontierExpansion):
    """Highest-degree unexpanded node first (pairs use the sum of both endpoint degrees)."""

    name: str = "degree-greedy"
    version: str = "1"

    def order(self, state: SearchState) -> list[FrontierItem]:
        def degree(item: FrontierItem) -> int:
            return sum(state.graph.degree(n) for n in item_nodes(item) if n in state.graph)

        return sorted(state.frontier, key=lambda item: -degree(item))


def association_score(item: FrontierItem, state: SearchState) -> float:
    anchors = set(state.anchor_ids())
    nodes = item_nodes(item)
    if item.edge_type is not EdgeType.HAS_EVIDENCE and nodes[0] in anchors:
        return 1.0
    edges = (
        state.graph.get_edge_data(nodes[0], nodes[1]) or {}
        if len(nodes) == 2
        else {
            k: d
            for _, _, k, d in [
                *state.graph.in_edges(nodes[0], keys=True, data=True),
                *state.graph.out_edges(nodes[0], keys=True, data=True),
            ]
        }
    )
    scores = [
        d["score"]
        for d in edges.values()
        if d.get("edge_type") == EdgeType.ASSOCIATED_WITH.value and d.get("score") is not None
    ]
    return max(scores, default=0.0)


@dataclass(frozen=True)
class ScoreGreedyExpansion(FrontierExpansion):
    """Highest OT association score first; anchors' own expansions rank first."""

    name: str = "score-greedy"
    version: str = "1"

    def order(self, state: SearchState) -> list[FrontierItem]:
        return sorted(state.frontier, key=lambda item: -association_score(item, state))


@dataclass(frozen=True)
class BreadthFirstExpansion(FrontierExpansion):
    """All frontier items at depth d before depth d+1. ``sweep`` proposes the whole level."""

    sweep: bool = False
    name: str = "breadth-first"
    version: str = "1"

    def order(self, state: SearchState) -> list[FrontierItem]:
        depths = state.depths()

        def depth(item: FrontierItem) -> int:
            return min(depths.get(n, UNREACHED_DEPTH) for n in item_nodes(item))

        return sorted(state.frontier, key=depth)

    def propose(self, state: SearchState) -> list[Action]:
        if not self.sweep:
            return super().propose(state)
        depths = state.depths()
        ordered = self.order(state)
        if not ordered:
            return []
        level = min(depths.get(n, UNREACHED_DEPTH) for n in item_nodes(ordered[0]))
        same_level = [
            i
            for i in ordered
            if min(depths.get(n, UNREACHED_DEPTH) for n in item_nodes(i)) == level
        ]
        return FrontierExpansion(batch=len(same_level), page_size=self.page_size).propose(
            _with_frontier(state, same_level)
        )


def _with_frontier(state: SearchState, items: list[FrontierItem]) -> SearchState:
    """A shallow read-only view of ``state`` with a different frontier (never mutates state)."""
    view = SearchState.__new__(SearchState)
    view.__dict__.update(state.__dict__)
    view.frontier = items
    return view


class CandidateScore(BaseModel):
    index: int
    score: float


class CandidateRanking(BaseModel):
    scores: list[CandidateScore]


@dataclass(frozen=True)
class LLMScoredExpansion(FrontierExpansion):
    """The model scores up to ``max_candidates`` frontier items; the top ``batch`` are proposed."""

    llm: LLMClient | None = field(default=None, compare=False)
    prompt_version: str = "expansion_rank_v1"
    max_candidates: int = 30
    name: str = "llm-scored-expansion"
    version: str = ""

    def __post_init__(self) -> None:
        if self.llm is None:
            raise ValueError("LLMScoredExpansion needs an llm client")
        if not self.version:
            object.__setattr__(self, "version", prompt_fingerprint(self.prompt_version))

    def order(self, state: SearchState) -> list[FrontierItem]:
        candidates = list(state.frontier)[: self.max_candidates]
        if not candidates:
            return []
        anchors = ", ".join(f"{a.name} ({a.entity_id})" for a in state.anchors) or "none yet"
        listing = "\n".join(f"[{i}] {describe(c, state)}" for i, c in enumerate(candidates))
        user = (
            f"Question: {state.working_question}\n"
            f"Resolved entities: {anchors}\n"
            f"Graph so far: {state.graph.number_of_nodes()} nodes, "
            f"{state.graph.number_of_edges()} edges\n\n"
            f"Candidates:\n{listing}"
        )
        ranking = self.llm.structured(
            prompt_version=self.prompt_version,
            system=load_prompt(self.prompt_version),
            user=user,
            output=CandidateRanking,
        )
        score = {s.index: s.score for s in ranking.scores if 0 <= s.index < len(candidates)}
        return sorted(candidates, key=lambda c: -score.get(candidates.index(c), 0.0))
