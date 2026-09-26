"""A hardcoded stub agent: a fixed three-call sequence that always stops at step 3."""

from __future__ import annotations

from dataclasses import dataclass, field

from otsearch_bench.env.models import EntityType
from otsearch_bench.search.actions import (
    Action,
    GetAssociatedDiseases,
    GetEvidence,
    ResolveEntity,
)
from otsearch_bench.search.agent import Agent, ModelSpec
from otsearch_bench.search.decisions import (
    AnchorMention,
    Answer,
    BudgetDecision,
    Formulation,
    PruneDecision,
    Reformulation,
    RejectedAction,
    StoppingDecision,
)
from otsearch_bench.search.graph import EdgeType
from otsearch_bench.search.state import Budget, SearchState

EGFR = "ENSG00000146648"
NSCLC = "MONDO_0005233"


@dataclass(frozen=True)
class FixedAnchorFormulation:
    anchor: str = "EGFR"
    name: str = "fixed-anchor-formulation"
    version: str = "1"

    def formulate(self, state: SearchState) -> Formulation:
        return Formulation(
            anchor_mentions=[AnchorMention(mention=self.anchor, entity_types=[EntityType.TARGET])]
        )


@dataclass(frozen=True)
class FixedSequenceExpansion:
    sequence: tuple[Action, ...] = field(
        default_factory=lambda: (
            ResolveEntity(name="EGFR", entity_types=[EntityType.TARGET]),
            GetAssociatedDiseases(ensembl_id=EGFR, page_size=5),
            GetEvidence(ensembl_id=EGFR, efo_id=NSCLC, size=5),
        )
    )
    name: str = "fixed-sequence-expansion"
    version: str = "1"

    def propose(self, state: SearchState) -> list[Action]:
        i = state.step - 1
        return [self.sequence[i]] if 0 <= i < len(self.sequence) else []


@dataclass(frozen=True)
class NoPruning:
    name: str = "no-pruning"
    version: str = "1"

    def prune(self, state: SearchState) -> PruneDecision:
        return PruneDecision()


@dataclass(frozen=True)
class NoReformulation:
    name: str = "no-reformulation"
    version: str = "1"

    def reformulate(self, state: SearchState) -> Reformulation | None:
        return None


@dataclass(frozen=True)
class StopAtStep:
    max_step: int = 3
    name: str = "stop-at-step"
    version: str = "1"

    def decide(self, state: SearchState) -> StoppingDecision:
        if state.step >= self.max_step:
            return StoppingDecision(stop=True, reason=f"reached step {self.max_step}")
        return StoppingDecision(stop=False, reason=f"step {state.step} < {self.max_step}")


@dataclass(frozen=True)
class EvidenceListingSynthesis:
    name: str = "evidence-listing-synthesis"
    version: str = "1"

    def synthesize(self, state: SearchState) -> Answer:
        evidence = sorted(state.edges(EdgeType.HAS_EVIDENCE), key=lambda e: -e[3]["score"])
        if not evidence:
            return Answer(text="No evidence found.", no_evidence=True)
        u, v = evidence[0][0], evidence[0][1]
        return Answer(
            text=f"{len(evidence)} evidence record(s) link {u} to {v}.",
            no_evidence=False,
            cited_evidence_ids=[e[3]["evidence_id"] for e in evidence],
        )


@dataclass(frozen=True)
class FixedCallBudget:
    max_calls: int = 10
    name: str = "fixed-call-budget"
    version: str = "1"

    def initial_budget(self) -> Budget:
        return Budget(max_calls=self.max_calls)

    def admit(self, state: SearchState, proposed: list[Action]) -> BudgetDecision:
        room = state.budget.calls_remaining
        return BudgetDecision(
            admitted=proposed[:room],
            rejected=[RejectedAction(action=a, reason="call budget") for a in proposed[room:]],
        )


def stub_agent() -> Agent:
    return Agent(
        name="stub",
        query_formulation=FixedAnchorFormulation(),
        expansion=FixedSequenceExpansion(),
        pruning=NoPruning(),
        reformulation=NoReformulation(),
        stopping=StopAtStep(3),
        synthesis=EvidenceListingSynthesis(),
        budget=FixedCallBudget(10),
        model=ModelSpec(model="none", temperature=0.0, top_p=1.0, seed=0),
        prompt_set_version="none",
    )
