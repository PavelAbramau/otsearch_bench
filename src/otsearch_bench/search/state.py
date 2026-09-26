"""``SearchState``: everything a policy may look at when deciding what to do next."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import networkx as nx
from pydantic import BaseModel
from pydantic_core import to_jsonable_python

from otsearch_bench.env.cache import canonical_json, sha256_hex
from otsearch_bench.search.actions import Action
from otsearch_bench.search.decisions import PruneDecision, Reformulation
from otsearch_bench.search.graph import (
    EXPANSIONS,
    EdgeRef,
    EdgeType,
    FrontierItem,
    NodeType,
    edge_key,
    evidence_item,
)
from otsearch_bench.usage import Usage


class Budget(BaseModel):
    max_calls: int
    max_tokens: int | None = None
    max_wall_clock_s: float | None = None
    max_usd: float | None = None
    calls_used: int = 0
    tokens_used: int = 0
    usd_used: float = 0.0
    wall_clock_s: float = 0.0

    @property
    def calls_remaining(self) -> int:
        return max(0, self.max_calls - self.calls_used)

    @property
    def tokens_remaining(self) -> int | None:
        return None if self.max_tokens is None else max(0, self.max_tokens - self.tokens_used)

    def exhausted_reason(self) -> str | None:
        if self.calls_used >= self.max_calls:
            return f"call budget exhausted ({self.calls_used}/{self.max_calls})"
        if self.max_tokens is not None and self.tokens_used >= self.max_tokens:
            return f"token budget exhausted ({self.tokens_used}/{self.max_tokens})"
        if self.max_wall_clock_s is not None and self.wall_clock_s >= self.max_wall_clock_s:
            return f"wall-clock budget exhausted ({self.wall_clock_s:.1f}s)"
        if self.max_usd is not None and self.usd_used >= self.max_usd:
            return f"cost budget exhausted (${self.usd_used:.4f})"
        return None

    def charge(self, delta: Usage, *, wall_clock_s: float) -> None:
        self.calls_used += delta.calls
        self.tokens_used += delta.tokens
        self.usd_used += delta.usd
        self.wall_clock_s = wall_clock_s


class Anchor(BaseModel):
    mention: str
    entity_id: str
    entity_type: str
    name: str
    method: str


def node_type_for(entity_id: str) -> NodeType:
    if entity_id.startswith("ENSG"):
        return NodeType.TARGET
    if entity_id.startswith("CHEMBL"):
        return NodeType.DRUG
    return NodeType.DISEASE


@dataclass
class SearchState:
    question: str
    seed: int
    budget: Budget
    working_question: str = ""
    anchors: list[Anchor] = field(default_factory=list)
    graph: nx.MultiDiGraph = field(default_factory=nx.MultiDiGraph)
    frontier: list[FrontierItem] = field(default_factory=list)
    expanded: set[FrontierItem] = field(default_factory=set)
    pruned_nodes: set[str] = field(default_factory=set)
    pending_actions: list[Action] = field(default_factory=list)
    step: int = 0
    frontier_added_step: dict[FrontierItem, int] = field(default_factory=dict)
    edges_added_total: int = 0
    edge_count_history: list[int] = field(default_factory=list)  # edges_added_total per step

    def __post_init__(self) -> None:
        self.working_question = self.working_question or self.question

    # --- mutation (Runner only) -----------------------------------------------------------

    def add_anchor(self, anchor: Anchor) -> None:
        if all(a.entity_id != anchor.entity_id for a in self.anchors):
            self.anchors.append(anchor)

    def add_node(self, node_id: str, name: str | None = None) -> None:
        if node_id in self.pruned_nodes:
            return
        node_type = node_type_for(node_id)
        if node_id not in self.graph:
            self.graph.add_node(node_id, node_type=node_type.value, name=name)
            for edge_type in EXPANSIONS[node_type]:
                self._push_frontier(FrontierItem(node=node_id, edge_type=edge_type))
        elif name and not self.graph.nodes[node_id].get("name"):
            self.graph.nodes[node_id]["name"] = name

    def add_edge(
        self,
        source: str,
        target: str,
        edge_type: EdgeType,
        *,
        evidence_id: str,
        score: float | None,
        **attrs: Any,
    ) -> EdgeRef | None:
        if source not in self.graph or target not in self.graph:
            return None  # an endpoint was pruned
        key = edge_key(edge_type, evidence_id)
        if not self.graph.has_edge(source, target, key):
            self.edges_added_total += 1
            if edge_type is EdgeType.ASSOCIATED_WITH:
                self._push_frontier(evidence_item(source, target))
            self.graph.add_edge(
                source,
                target,
                key=key,
                edge_type=edge_type.value,
                evidence_id=evidence_id,
                score=score,
                **attrs,
            )
        return EdgeRef(source=source, target=target, key=key)

    def _push_frontier(self, item: FrontierItem) -> None:
        if item not in self.expanded and item not in self.frontier:
            self.frontier.append(item)
            self.frontier_added_step.setdefault(item, self.step)

    def mark_expanded(self, items: list[FrontierItem]) -> None:
        self.expanded.update(items)
        self.frontier = [f for f in self.frontier if f not in self.expanded]

    def apply_prune(self, decision: PruneDecision) -> None:
        for ref in decision.remove_edges:
            if self.graph.has_edge(ref.source, ref.target, ref.key):
                self.graph.remove_edge(ref.source, ref.target, ref.key)
        for node in decision.remove_nodes:
            if node in self.graph:
                self.graph.remove_node(node)
            self.pruned_nodes.add(node)
        dropped = set(decision.drop_frontier)
        self.frontier = [
            f for f in self.frontier if f not in dropped and f.node not in self.pruned_nodes
        ]

    def apply_reformulation(self, reformulation: Reformulation) -> None:
        if reformulation.question:
            self.working_question = reformulation.question
        self.pending_actions.extend(reformulation.actions)

    # --- read-only views ------------------------------------------------------------------

    def anchor_ids(self) -> list[str]:
        return [a.entity_id for a in self.anchors if a.entity_id in self.graph]

    def depths(self) -> dict[str, int]:
        """Undirected hop distance of every node from the nearest anchor."""
        undirected = self.graph.to_undirected(as_view=True)
        depth = dict.fromkeys(self.anchor_ids(), 0)
        queue = list(depth)
        while queue:
            node = queue.pop(0)
            for neighbour in undirected.neighbors(node):
                if neighbour not in depth:
                    depth[neighbour] = depth[node] + 1
                    queue.append(neighbour)
        return depth

    def edges(self, edge_type: EdgeType | None = None) -> list[tuple[str, str, str, dict]]:
        return [
            (u, v, k, d)
            for u, v, k, d in self.graph.edges(keys=True, data=True)
            if edge_type is None or d["edge_type"] == edge_type.value
        ]

    def snapshot(self) -> dict[str, Any]:
        """Deterministic JSON-able view of the state (wall-clock excluded)."""
        nodes = sorted((n, d) for n, d in self.graph.nodes(data=True))
        edges = sorted(
            ([u, v, k, d] for u, v, k, d in self.graph.edges(keys=True, data=True)),
            key=lambda e: (e[0], e[1], e[2]),
        )
        return to_jsonable_python(
            {
                "question": self.question,
                "working_question": self.working_question,
                "seed": self.seed,
                "step": self.step,
                "anchors": [a.model_dump() for a in self.anchors],
                "nodes": nodes,
                "edges": edges,
                "frontier": [[f.node, f.edge_type.value] for f in self.frontier],
                "expanded": sorted([f.node, f.edge_type.value] for f in self.expanded),
                "pruned_nodes": sorted(self.pruned_nodes),
                "pending_actions": [a.action_id for a in self.pending_actions],
                "calls_used": self.budget.calls_used,
                "edge_count_history": self.edge_count_history,
            }
        )

    def fingerprint(self) -> str:
        return sha256_hex(canonical_json(self.snapshot()))
