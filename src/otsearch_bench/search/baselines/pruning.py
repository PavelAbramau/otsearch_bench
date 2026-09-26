"""Baseline PruningPolicy implementations."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from otsearch_bench.search.decisions import PruneDecision
from otsearch_bench.search.graph import EdgeRef, EdgeType, evidence_item
from otsearch_bench.search.state import SearchState


@dataclass(frozen=True)
class KeepAll:
    name: str = "keep-all"
    version: str = "1"

    def prune(self, state: SearchState) -> PruneDecision:
        return PruneDecision()


@dataclass(frozen=True)
class RecencyWindow:
    """Drop frontier items that have waited more than ``k`` steps. Retrieved edges are kept."""

    k: int = 3
    name: str = "recency-window"
    version: str = "1"

    def prune(self, state: SearchState) -> PruneDecision:
        stale = [
            item
            for item in state.frontier
            if state.step - state.frontier_added_step.get(item, state.step) > self.k
        ]
        return PruneDecision(
            drop_frontier=stale,
            reason=f"dropped {len(stale)} frontier items older than {self.k} steps"
            if stale
            else None,
        )


@dataclass(frozen=True)
class SubgraphSummarize:
    """Keep associations in the top ``top_m`` at both endpoints; drop the rest and orphans.

    Evidence, drug-target and indication edges are never removed, and anchors are never removed.
    """

    top_m: int = 10
    name: str = "subgraph-summarize"
    version: str = "1"

    def prune(self, state: SearchState) -> PruneDecision:
        associations = state.edges(EdgeType.ASSOCIATED_WITH)
        by_node: dict[str, list[tuple[float, str, str, str]]] = defaultdict(list)
        for u, v, k, d in associations:
            entry = (d.get("score") or 0.0, u, v, k)
            by_node[u].append(entry)
            by_node[v].append(entry)
        top = {
            node: {(u, v, k) for _, u, v, k in sorted(entries, reverse=True)[: self.top_m]}
            for node, entries in by_node.items()
        }
        removed = [
            (u, v, k)
            for u, v, k, _ in associations
            if (u, v, k) not in top[u] or (u, v, k) not in top[v]
        ]
        if not removed:
            return PruneDecision()

        removed_set = set(removed)
        anchors = set(state.anchor_ids())
        orphans = [
            n
            for n in state.graph.nodes
            if n not in anchors
            and all(
                (u, v, k) in removed_set
                for u, v, k in [
                    *state.graph.in_edges(n, keys=True),
                    *state.graph.out_edges(n, keys=True),
                ]
            )
            and state.graph.degree(n) > 0
        ]
        return PruneDecision(
            remove_edges=[EdgeRef(source=u, target=v, key=k) for u, v, k in removed],
            remove_nodes=orphans,
            drop_frontier=[evidence_item(u, v) for u, v, _ in removed],
            reason=(
                f"kept {len(associations) - len(removed)} of {len(associations)} associations "
                f"(top {self.top_m} per node); removed {len(orphans)} orphaned nodes"
            ),
        )
