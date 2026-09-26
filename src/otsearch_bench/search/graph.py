"""Vocabulary of the evidence subgraph: node types, typed edges, and frontier items."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict


class NodeType(StrEnum):
    TARGET = "target"
    DISEASE = "disease"
    DRUG = "drug"


class EdgeType(StrEnum):
    ASSOCIATED_WITH = "associated_with"  # target -> disease, overall OT association score
    HAS_EVIDENCE = "has_evidence"  # target -> disease, one OT evidence record
    TARGETS = "targets"  # drug -> target (clinical candidates / mechanism of action)
    INDICATED_FOR = "indicated_for"  # drug -> disease (clinical indication)


# Which (node, edge_type) expansions exist for each node type.
EXPANSIONS: dict[NodeType, tuple[EdgeType, ...]] = {
    NodeType.TARGET: (EdgeType.ASSOCIATED_WITH, EdgeType.TARGETS),
    NodeType.DISEASE: (EdgeType.ASSOCIATED_WITH,),
    NodeType.DRUG: (EdgeType.TARGETS, EdgeType.INDICATED_FOR),
}


class FrontierItem(BaseModel):
    """An unexpanded (node, edge_type) pair."""

    model_config = ConfigDict(frozen=True)

    node: str
    edge_type: EdgeType


class EdgeRef(BaseModel):
    model_config = ConfigDict(frozen=True)

    source: str
    target: str
    key: str


def evidence_item(target: str, disease: str) -> FrontierItem:
    """Frontier item for fetching the evidence behind a target-disease association."""
    return FrontierItem(node=f"{target}|{disease}", edge_type=EdgeType.HAS_EVIDENCE)


def split_pair(node: str) -> tuple[str, str] | None:
    return tuple(node.split("|", 1)) if "|" in node else None  # type: ignore[return-value]


def edge_key(edge_type: EdgeType, evidence_id: str) -> str:
    return f"{edge_type.value}:{evidence_id}"
