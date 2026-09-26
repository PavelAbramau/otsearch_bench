"""Baseline StoppingPolicy implementations."""

from __future__ import annotations

from dataclasses import dataclass, field

from pydantic import BaseModel

from otsearch_bench.prompts import load_prompt, prompt_fingerprint
from otsearch_bench.search.baselines.digest import graph_digest
from otsearch_bench.search.decisions import StoppingDecision
from otsearch_bench.search.llm import LLMClient
from otsearch_bench.search.state import SearchState


@dataclass(frozen=True)
class FixedBudgetStopping:
    """Stop once ``n`` calls have been used."""

    n: int = 10
    name: str = "fixed-budget-stopping"
    version: str = "1"

    def decide(self, state: SearchState) -> StoppingDecision:
        used = state.budget.calls_used
        if used >= self.n:
            return StoppingDecision(stop=True, reason=f"used {used} >= {self.n} calls")
        return StoppingDecision(stop=False, reason=f"used {used} < {self.n} calls")


@dataclass(frozen=True)
class NoNewEdgesStopping:
    """Stop after ``patience`` consecutive steps that added no new edge, or an empty frontier."""

    patience: int = 2
    name: str = "no-new-edges-stopping"
    version: str = "1"

    def decide(self, state: SearchState) -> StoppingDecision:
        history = state.edge_count_history
        if len(history) > self.patience and history[-1] == history[-1 - self.patience]:
            return StoppingDecision(
                stop=True, reason=f"no new edges in the last {self.patience} steps"
            )
        if not state.frontier and not state.pending_actions and state.step > 0:
            return StoppingDecision(stop=True, reason="frontier exhausted")
        return StoppingDecision(stop=False, reason="still adding edges")


class Sufficiency(BaseModel):
    sufficient: bool
    reason: str


@dataclass(frozen=True)
class LLMSelfAssessStopping:
    """Asks the model whether the evidence gathered so far is sufficient to answer."""

    llm: LLMClient | None = field(default=None, compare=False)
    prompt_version: str = "stopping_self_assess_v1"
    min_step: int = 1
    max_edges: int = 80
    name: str = "llm-self-assess-stopping"
    version: str = ""

    def __post_init__(self) -> None:
        if self.llm is None:
            raise ValueError("LLMSelfAssessStopping needs an llm client")
        if not self.version:
            object.__setattr__(self, "version", prompt_fingerprint(self.prompt_version))

    def decide(self, state: SearchState) -> StoppingDecision:
        if state.step < self.min_step:
            return StoppingDecision(stop=False, reason=f"step {state.step} < {self.min_step}")
        verdict = self.llm.structured(
            prompt_version=self.prompt_version,
            system=load_prompt(self.prompt_version),
            user=f"Question: {state.working_question}\n\n"
            + graph_digest(state, max_edges=self.max_edges),
            output=Sufficiency,
        )
        return StoppingDecision(stop=verdict.sufficient, reason=verdict.reason)
