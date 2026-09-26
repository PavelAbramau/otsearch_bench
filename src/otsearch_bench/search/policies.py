"""The seven swappable policy roles.

Contract for every role: policies are pure deciders. They read ``SearchState`` and return
decision values; they never mutate state and never call the adapter. The Runner enforces both.
Each implementation carries a ``name`` and ``version`` that feed ``Agent.version_tuple()``.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from otsearch_bench.search.actions import Action
from otsearch_bench.search.decisions import (
    Answer,
    BudgetDecision,
    Formulation,
    PruneDecision,
    Reformulation,
    StoppingDecision,
)
from otsearch_bench.search.state import Budget, SearchState


@runtime_checkable
class QueryFormulationPolicy(Protocol):
    name: str
    version: str

    def formulate(self, state: SearchState) -> Formulation: ...


@runtime_checkable
class ExpansionPolicy(Protocol):
    name: str
    version: str

    def propose(self, state: SearchState) -> list[Action]: ...


@runtime_checkable
class PruningPolicy(Protocol):
    name: str
    version: str

    def prune(self, state: SearchState) -> PruneDecision: ...


@runtime_checkable
class ReformulationPolicy(Protocol):
    """Consulted when a step admitted no actions; returning None (or no actions) ends the run."""

    name: str
    version: str

    def reformulate(self, state: SearchState) -> Reformulation | None: ...


@runtime_checkable
class StoppingPolicy(Protocol):
    name: str
    version: str

    def decide(self, state: SearchState) -> StoppingDecision: ...


@runtime_checkable
class SynthesisPolicy(Protocol):
    name: str
    version: str

    def synthesize(self, state: SearchState) -> Answer: ...


@runtime_checkable
class BudgetPolicy(Protocol):
    name: str
    version: str

    def initial_budget(self) -> Budget: ...

    def admit(self, state: SearchState, proposed: list[Action]) -> BudgetDecision: ...
