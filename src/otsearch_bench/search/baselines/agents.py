"""Named baseline agents, the two non-agentic reference systems, and ``factorial_grid``."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel

from otsearch_bench.prompts import load_prompt, prompt_fingerprint
from otsearch_bench.search.actions import Action, RawGraphQL
from otsearch_bench.search.agent import ROLE_PROTOCOLS, Agent, ModelSpec
from otsearch_bench.search.baselines.expansion import (
    UNREACHED_DEPTH,
    BreadthFirstExpansion,
    DegreeGreedyExpansion,
    LLMScoredExpansion,
    RandomWalkExpansion,
    ScoreGreedyExpansion,
    item_nodes,
)
from otsearch_bench.search.baselines.pruning import KeepAll, RecencyWindow, SubgraphSummarize
from otsearch_bench.search.baselines.roles import (
    CallBudget,
    EvidenceSummarySynthesis,
    LLMSynthesis,
    NoReformulation,
    SpanFormulation,
)
from otsearch_bench.search.baselines.stopping import (
    FixedBudgetStopping,
    LLMSelfAssessStopping,
    NoNewEdgesStopping,
)
from otsearch_bench.search.decisions import Formulation, StoppingDecision
from otsearch_bench.search.llm import DEFAULT_MODEL, LLMClient
from otsearch_bench.search.state import SearchState


@dataclass(frozen=True)
class AgentContext:
    """Shared construction settings. Without an ``llm`` the LLM-backed roles are left out."""

    llm: LLMClient | None = field(default=None, compare=False)
    model: ModelSpec = field(
        default_factory=lambda: ModelSpec(model=DEFAULT_MODEL, temperature=1.0, top_p=1.0, seed=0)
    )
    call_budget: int = 20
    page_size: int = 25
    prompt_set_version: str = "prompts-v1"

    def synthesis(self) -> Any:
        return LLMSynthesis(llm=self.llm) if self.llm else EvidenceSummarySynthesis()


BASELINES = ("score-greedy", "breadth-first")


def baseline_agent(name: str, ctx: AgentContext) -> Agent:
    expansion = {
        "score-greedy": ScoreGreedyExpansion(page_size=ctx.page_size),
        "breadth-first": BreadthFirstExpansion(page_size=ctx.page_size),
    }
    if name not in expansion:
        raise KeyError(f"unknown baseline {name!r}; choose from {BASELINES}")
    return Agent(
        name=name,
        query_formulation=SpanFormulation(),
        expansion=expansion[name],
        pruning=KeepAll(),
        reformulation=NoReformulation(),
        stopping=NoNewEdgesStopping(patience=2),
        synthesis=ctx.synthesis(),
        budget=CallBudget(ctx.call_budget),
        model=ctx.model,
        prompt_set_version=ctx.prompt_set_version,
    )


def role_variants(ctx: AgentContext) -> dict[str, list[Any]]:
    """Alternatives per role. LLM-backed variants are included only when ``ctx.llm`` is set."""
    p = ctx.page_size
    variants: dict[str, list[Any]] = {
        "expansion": [
            RandomWalkExpansion(page_size=p),
            DegreeGreedyExpansion(page_size=p),
            ScoreGreedyExpansion(page_size=p),
            BreadthFirstExpansion(page_size=p),
        ],
        "stopping": [
            FixedBudgetStopping(n=max(1, ctx.call_budget // 2)),
            NoNewEdgesStopping(patience=2),
        ],
        "pruning": [KeepAll(), RecencyWindow(k=3), SubgraphSummarize(top_m=10)],
    }
    if ctx.llm is not None:
        variants["expansion"].append(LLMScoredExpansion(llm=ctx.llm, page_size=p))
        variants["stopping"].append(LLMSelfAssessStopping(llm=ctx.llm))
    return variants


class GridCell(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    label: str
    role: str | None  # None for the baseline itself
    component: str | None
    agent: Agent


def factorial_grid(
    baseline: str | Agent,
    ctx: AgentContext,
    variants: dict[str, list[Any]] | None = None,
) -> list[GridCell]:
    """The baseline plus one agent per alternative component, each differing in exactly one role."""
    base = baseline if isinstance(baseline, Agent) else baseline_agent(baseline, ctx)
    variants = role_variants(ctx) if variants is None else variants
    cells = [GridCell(label=base.name, role=None, component=None, agent=base)]
    for role, components in variants.items():
        if role not in ROLE_PROTOCOLS:
            raise KeyError(f"unknown role {role!r}")
        current = getattr(base, role)
        for component in components:
            if (component.name, component.version, type(component)) == (
                current.name,
                current.version,
                type(current),
            ) and component == current:
                continue
            label = f"{base.name} | {role}={component.name}"
            agent = dataclasses.replace(base, name=label, **{role: component})
            cells.append(GridCell(label=label, role=role, component=component.name, agent=agent))
    return cells


# --- reference systems --------------------------------------------------------------------


@dataclass(frozen=True)
class SweepDepthStopping:
    """Stop once every frontier item within ``depth`` hops of the anchors has been expanded."""

    depth: int = 2
    name: str = "sweep-depth-stopping"
    version: str = "1"

    def decide(self, state: SearchState) -> StoppingDecision:
        depths = state.depths()
        remaining = [
            item
            for item in state.frontier
            if min(depths.get(n, UNREACHED_DEPTH) for n in item_nodes(item)) < self.depth
        ]
        if remaining:
            return StoppingDecision(
                stop=False, reason=f"{len(remaining)} items within {self.depth} hops"
            )
        return StoppingDecision(stop=True, reason=f"{self.depth}-hop neighbourhood fetched")


def one_shot_dump(ctx: AgentContext, *, depth: int = 2, max_calls: int = 400) -> Agent:
    """Fetch the full ``depth``-hop neighbourhood of all anchors level by level, answer once."""
    return Agent(
        name="reference:one-shot-dump",
        query_formulation=SpanFormulation(),
        expansion=BreadthFirstExpansion(sweep=True, page_size=ctx.page_size),
        pruning=KeepAll(),
        reformulation=NoReformulation(),
        stopping=SweepDepthStopping(depth=depth),
        synthesis=LLMSynthesis(llm=ctx.llm, max_edges=2000)
        if ctx.llm
        else EvidenceSummarySynthesis(),
        budget=CallBudget(max_calls),
        model=ctx.model,
        prompt_set_version=ctx.prompt_set_version,
    )


class GraphQLDraft(BaseModel):
    query: str
    rationale: str


@dataclass(frozen=True)
class LLMGraphQLFormulation:
    """The model writes one GraphQL document; it is the only action of the run."""

    llm: LLMClient | None = field(default=None, compare=False)
    prompt_version: str = "oneshot_graphql_v1"
    name: str = "llm-graphql-formulation"
    version: str = ""

    def __post_init__(self) -> None:
        if self.llm is None:
            raise ValueError("LLMGraphQLFormulation needs an llm client")
        if not self.version:
            object.__setattr__(self, "version", prompt_fingerprint(self.prompt_version))

    def formulate(self, state: SearchState) -> Formulation:
        draft = self.llm.structured(
            prompt_version=self.prompt_version,
            system=load_prompt(self.prompt_version),
            user=f"Question: {state.working_question}",
            output=GraphQLDraft,
        )
        return Formulation(actions=[RawGraphQL(query=draft.query)], notes=draft.rationale)


@dataclass(frozen=True)
class NoExpansion:
    name: str = "no-expansion"
    version: str = "1"

    def propose(self, state: SearchState) -> list[Action]:
        return []


@dataclass(frozen=True)
class StopImmediately:
    name: str = "stop-immediately"
    version: str = "1"

    def decide(self, state: SearchState) -> StoppingDecision:
        return StoppingDecision(stop=True, reason="one-shot reference system")


def one_shot_graphql(ctx: AgentContext) -> Agent:
    """Single model-written GraphQL query, executed once, then one answer."""
    if ctx.llm is None:
        raise ValueError("one_shot_graphql needs an llm client")
    return Agent(
        name="reference:one-shot-graphql",
        query_formulation=LLMGraphQLFormulation(llm=ctx.llm),
        expansion=NoExpansion(),
        pruning=KeepAll(),
        reformulation=NoReformulation(),
        stopping=StopImmediately(),
        synthesis=LLMSynthesis(llm=ctx.llm, max_edges=2000),
        budget=CallBudget(1),
        model=ctx.model,
        prompt_set_version=ctx.prompt_set_version,
    )


def reference_systems(ctx: AgentContext) -> list[Agent]:
    systems = [one_shot_dump(ctx)]
    if ctx.llm is not None:
        systems.append(one_shot_graphql(ctx))
    return systems
