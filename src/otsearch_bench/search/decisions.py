"""Values returned by policies. Policies decide; only the Runner applies decisions."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from otsearch_bench.env.models import EntityType
from otsearch_bench.search.actions import Action
from otsearch_bench.search.graph import EdgeRef, FrontierItem


class AnchorMention(BaseModel):
    mention: str
    entity_types: list[EntityType] | None = None


class Formulation(BaseModel):
    anchor_mentions: list[AnchorMention] = []
    actions: list[Action] = []
    notes: str | None = None


class RejectedAction(BaseModel):
    action: Action
    reason: str


class BudgetDecision(BaseModel):
    admitted: list[Action]
    rejected: list[RejectedAction] = []


class PruneDecision(BaseModel):
    drop_frontier: list[FrontierItem] = []
    remove_edges: list[EdgeRef] = []
    remove_nodes: list[str] = []
    reason: str | None = None

    @property
    def is_noop(self) -> bool:
        return not (self.drop_frontier or self.remove_edges or self.remove_nodes)


class Reformulation(BaseModel):
    reason: str
    question: str | None = None  # replaces the working question when set
    actions: list[Action] = []  # proposed first on the next step


class StoppingDecision(BaseModel):
    stop: bool
    reason: str
    source: Literal["policy", "runner"] = "policy"


class Claim(BaseModel):
    text: str
    evidence_ids: list[str] = []


class Answer(BaseModel):
    text: str
    no_evidence: bool
    entities: list[str] = []
    category: str | None = None
    count: int | None = None
    claims: list[Claim] = []
    cited_evidence_ids: list[str] = []
    confidence: float | None = None
