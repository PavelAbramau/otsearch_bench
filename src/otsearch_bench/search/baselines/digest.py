"""Plain-text digests of the evidence subgraph for LLM-backed policies."""

from __future__ import annotations

from otsearch_bench.search.graph import EdgeType
from otsearch_bench.search.state import SearchState

_EDGE_ORDER = {
    EdgeType.HAS_EVIDENCE.value: 0,
    EdgeType.TARGETS.value: 1,
    EdgeType.INDICATED_FOR.value: 2,
    EdgeType.ASSOCIATED_WITH.value: 3,
}


def node_label(state: SearchState, node: str) -> str:
    name = state.graph.nodes[node].get("name") if node in state.graph else None
    return f"{name} ({node})" if name else node


def graph_digest(state: SearchState, *, max_edges: int = 80) -> str:
    """Anchors plus the highest-priority edges, each line carrying its OT evidence id."""
    anchors = ", ".join(f"{a.name} ({a.entity_id})" for a in state.anchors) or "none"
    edges = sorted(
        state.edges(),
        key=lambda e: (_EDGE_ORDER.get(e[3]["edge_type"], 9), -(e[3].get("score") or 0.0)),
    )
    lines = [
        f"Anchors: {anchors}",
        f"Graph: {state.graph.number_of_nodes()} nodes, {state.graph.number_of_edges()} edges "
        f"(showing {min(max_edges, len(edges))})",
    ]
    for u, v, _, d in edges[:max_edges]:
        score = f" score={d['score']:.3f}" if d.get("score") is not None else ""
        extra = "".join(
            f" {k}={d[k]}"
            for k in ("datasource_id", "max_clinical_stage", "action_type")
            if d.get(k)
        )
        lines.append(
            f"- {node_label(state, u)} -[{d['edge_type']}{score}{extra}]-> "
            f"{node_label(state, v)} evidence_id={d['evidence_id']}"
        )
    return "\n".join(lines)
