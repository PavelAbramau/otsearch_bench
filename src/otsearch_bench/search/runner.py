"""``Runner``: the only component that executes actions against the adapter."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any, TypeVar

from pydantic import BaseModel

from otsearch_bench.env.adapter import OTAdapter
from otsearch_bench.env.cache import utc_now_iso
from otsearch_bench.env.errors import (
    EntityNotFoundError,
    OTGraphQLError,
    OTHTTPError,
    RetryExhaustedError,
)
from otsearch_bench.env.models import (
    AssociatedDiseases,
    AssociatedTargets,
    Disease,
    Drug,
    EntityResolution,
    EvidencePage,
    KnownDrugs,
    Target,
)
from otsearch_bench.search.actions import Action
from otsearch_bench.search.agent import ROLE_PROTOCOLS, Agent
from otsearch_bench.search.decisions import StoppingDecision
from otsearch_bench.search.graph import EdgeType
from otsearch_bench.search.state import Anchor, SearchState
from otsearch_bench.search.trajectory import (
    CallProvenance,
    Observation,
    ObservationError,
    QueryInput,
    Step,
    Trajectory,
)
from otsearch_bench.usage import Usage

T = TypeVar("T")

# Failures that become an error observation. CacheMissError is deliberately absent: a REPLAY
# run that asks for something uncached is not a faithful replay and must fail loudly.
RECOVERABLE_ERRORS = (EntityNotFoundError, OTGraphQLError, OTHTTPError, RetryExhaustedError)

_DISPATCH: dict[str, Callable[[OTAdapter, Any], BaseModel]] = {
    "resolve_entity": lambda ot, a: ot.resolve_entity(a.name, a.entity_types),
    "get_target": lambda ot, a: ot.get_target(a.ensembl_id),
    "get_disease": lambda ot, a: ot.get_disease(a.efo_id),
    "get_drug": lambda ot, a: ot.get_drug(a.chembl_id),
    "get_known_drugs": lambda ot, a: ot.get_known_drugs(a.ensembl_id),
    "get_associated_diseases": lambda ot, a: ot.get_associated_diseases(
        a.ensembl_id, page_index=a.page_index, page_size=a.page_size
    ),
    "get_associated_targets": lambda ot, a: ot.get_associated_targets(
        a.efo_id, page_index=a.page_index, page_size=a.page_size
    ),
    "raw_graphql": lambda ot, a: RawGraphQLResult(data=ot.execute(a.query, a.variables).response),
    "get_evidence": lambda ot, a: ot.get_evidence(
        a.ensembl_id, a.efo_id, size=a.size, cursor=a.cursor, datasource_ids=a.datasource_ids
    ),
}


class RawGraphQLResult(BaseModel):
    data: dict[str, Any]


def llm_usage(agent: Agent) -> Usage:
    """Summed usage of every distinct LLM client held by the agent's components."""
    clients = {}
    for role in ROLE_PROTOCOLS:
        client = getattr(getattr(agent, role), "llm", None)
        if client is not None:
            clients[id(client)] = client
    total = Usage()
    for client in clients.values():
        total = total + client.usage
    return total


class PolicyViolationError(RuntimeError):
    """A policy mutated SearchState, called the adapter, or admitted an unproposed action."""


class Runner:
    def __init__(
        self,
        adapter: OTAdapter,
        *,
        max_steps: int = 50,
        record_raw_responses: bool = True,
        check_policy_purity: bool = True,
    ):
        self.adapter = adapter
        self.max_steps = max_steps
        self.record_raw_responses = record_raw_responses
        self.check_policy_purity = check_policy_purity

    def run(self, agent: Agent, query: str | QueryInput, *, seed: int | None = None) -> Trajectory:
        if not isinstance(agent, Agent):
            raise TypeError("Runner.run requires an Agent (nothing runs without a version tuple)")
        query = QueryInput.coerce(query)
        seed = agent.model.seed if seed is None else seed
        versions = agent.version_tuple(self.adapter)
        started_at, t0 = utc_now_iso(), time.perf_counter()
        usage_start = self._usage(agent)
        log_start = len(self.adapter.call_log)
        state = SearchState(
            question=query.question, seed=seed, budget=agent.budget.initial_budget()
        )

        formulation = self._policy(agent.query_formulation.formulate, state)
        first = self._step(agent, state, "formulate", list(formulation.actions), t0)
        steps = [first.model_copy(update={"formulation": formulation})]

        while True:
            state.step += 1
            pending, state.pending_actions = state.pending_actions, []
            proposed = pending + list(self._policy(agent.expansion.propose, state))
            step = self._step(agent, state, "expand", proposed, t0)
            steps.append(step)
            if step.stopping is not None and step.stopping.stop:
                break

        answer = self._policy(agent.synthesis.synthesize, state)
        versions_seen = dict.fromkeys(
            r.data_version for r in self.adapter.call_log[log_start:] if r.data_version
        )
        return Trajectory(
            trajectory_id=Trajectory.make_id(agent.agent_id, query.query_id, seed),
            agent_id=agent.agent_id,
            versions=versions,
            query=query,
            seed=seed,
            adapter_mode=self.adapter.mode.value,
            started_at=started_at,
            steps=steps,
            finished_at=utc_now_iso(),
            stopping=step.stopping,
            answer=answer,
            total_usage=self._usage(agent) - usage_start,
            data_versions_seen=list(versions_seen),
            final_state_fingerprint=state.fingerprint(),
            final_state=state.snapshot(),
        )

    # --- one step -------------------------------------------------------------------------

    def _step(
        self, agent: Agent, state: SearchState, phase: str, proposed: list[Action], t0: float
    ) -> Step:
        started = time.perf_counter()
        fingerprint_before = state.fingerprint()
        usage_before = self._usage(agent)

        decision = self._policy(agent.budget.admit, state, proposed)
        proposed_ids = {a.action_id for a in proposed}
        if any(a.action_id not in proposed_ids for a in decision.admitted):
            raise PolicyViolationError(f"{agent.budget.name} admitted an action nobody proposed")

        observations = [self._execute(action, state) for action in decision.admitted]
        state.budget.charge(
            self._usage(agent) - usage_before, wall_clock_s=time.perf_counter() - t0
        )

        prune = stopping = reformulation = None
        if phase == "expand":
            state.edge_count_history.append(state.edges_added_total)
            prune = self._policy(agent.pruning.prune, state)
            state.apply_prune(prune)
            stopping = self._policy(agent.stopping.decide, state)
            if not stopping.stop:
                stopping = self._runner_stop(state) or stopping
            if not stopping.stop and not decision.admitted:
                reformulation = self._policy(agent.reformulation.reformulate, state)
                if reformulation is not None:
                    state.apply_reformulation(reformulation)
                if reformulation is None or not reformulation.actions:
                    stopping = StoppingDecision(
                        stop=True,
                        reason="stalled: no admitted actions and no reformulation",
                        source="runner",
                    )

        return Step(
            index=state.step,
            phase=phase,
            state_fingerprint=fingerprint_before,
            state_fingerprint_after=state.fingerprint(),
            proposed_actions=proposed,
            admitted_actions=decision.admitted,
            rejected_actions=decision.rejected,
            observations=observations,
            cache_hit_flags=[c.cache_hit for o in observations for c in o.calls],
            cost_delta=self._usage(agent) - usage_before,
            budget=state.budget.model_copy(),
            prune=prune,
            reformulation=reformulation,
            stopping=stopping,
            wall_clock_s=time.perf_counter() - started,
        )

    def _usage(self, agent: Agent) -> Usage:
        return self.adapter.usage + llm_usage(agent)

    def _runner_stop(self, state: SearchState) -> StoppingDecision | None:
        if reason := state.budget.exhausted_reason():
            return StoppingDecision(stop=True, reason=reason, source="runner")
        if state.step >= self.max_steps:
            return StoppingDecision(
                stop=True, reason=f"runner max_steps={self.max_steps}", source="runner"
            )
        return None

    def _policy(self, fn: Callable[..., T], state: SearchState, *args: Any) -> T:
        if not self.check_policy_purity:
            return fn(state, *args)
        fingerprint, calls = state.fingerprint(), len(self.adapter.call_log)
        result = fn(state, *args)
        name = getattr(fn, "__qualname__", repr(fn))
        if len(self.adapter.call_log) != calls:
            raise PolicyViolationError(f"{name} called the adapter directly")
        if state.fingerprint() != fingerprint:
            raise PolicyViolationError(f"{name} mutated SearchState")
        return result

    # --- execution and state update -------------------------------------------------------

    def _execute(self, action: Action, state: SearchState) -> Observation:
        log_start, usage_before = len(self.adapter.call_log), self.adapter.usage
        result: BaseModel | None = None
        error: ObservationError | None = None
        try:
            result = _DISPATCH[action.kind](self.adapter, action)
        except RECOVERABLE_ERRORS as exc:
            error = ObservationError(type=type(exc).__name__, message=str(exc))
        records = self.adapter.call_log[log_start:]
        if result is not None:
            apply_result(state, action, result)
        state.mark_expanded(action.expands())
        return Observation(
            action_id=action.action_id,
            kind=action.kind,
            ok=error is None,
            result=result.model_dump(mode="json") if result is not None else None,
            error=error,
            calls=[
                CallProvenance(
                    cache_key=r.cache_key,
                    operation=r.operation,
                    cache_hit=r.cache_hit,
                    row_id=r.row_id,
                    response_sha256=r.response_sha256,
                    data_version=r.data_version,
                    fetched_at=r.fetched_at,
                    network_attempts=r.network_attempts,
                )
                for r in records
            ],
            raw_responses=[r.response for r in records] if self.record_raw_responses else None,
            usage=self.adapter.usage - usage_before,
        )


def apply_result(state: SearchState, action: Action, result: BaseModel) -> None:
    """Fold a typed observation into the evidence subgraph."""
    src = action.action_id
    match result:
        case EntityResolution(best=best) if best is not None:
            state.add_node(best.id, best.name)
            state.add_anchor(
                Anchor(
                    mention=result.query,
                    entity_id=best.id,
                    entity_type=best.entity,
                    name=best.name,
                    method=result.method,
                )
            )
        case Target():
            state.add_node(result.id, result.approved_symbol)
        case Disease():
            state.add_node(result.id, result.name)
        case Drug():
            state.add_node(result.id, result.name)
            for moa in result.mechanisms_of_action.rows if result.mechanisms_of_action else []:
                for target in moa.targets:
                    state.add_node(target.id, target.approved_symbol)
                    state.add_edge(
                        result.id,
                        target.id,
                        EdgeType.TARGETS,
                        evidence_id=f"moa:{result.id}:{target.id}",
                        score=None,
                        action_type=moa.action_type,
                        mechanism=moa.mechanism_of_action,
                        source_action=src,
                    )
            for ind in result.indications.rows if result.indications else []:
                if ind.disease is None:
                    continue
                state.add_node(ind.disease.id, ind.disease.name)
                state.add_edge(
                    result.id,
                    ind.disease.id,
                    EdgeType.INDICATED_FOR,
                    evidence_id=ind.id,
                    score=None,
                    max_clinical_stage=ind.max_clinical_stage,
                    source_action=src,
                )
        case KnownDrugs():
            state.add_node(result.target.id, result.target.approved_symbol)
            for row in result.rows:
                if row.drug is None:
                    continue
                state.add_node(row.drug.id, row.drug.name)
                state.add_edge(
                    row.drug.id,
                    result.target.id,
                    EdgeType.TARGETS,
                    evidence_id=row.id,
                    score=None,
                    max_clinical_stage=row.max_clinical_stage,
                    source_action=src,
                )
                for d in row.diseases:
                    if d.disease is None:
                        continue
                    state.add_node(d.disease.id, d.disease.name)
                    state.add_edge(
                        row.drug.id,
                        d.disease.id,
                        EdgeType.INDICATED_FOR,
                        evidence_id=f"{row.id}:{d.disease.id}",
                        score=None,
                        max_clinical_stage=row.max_clinical_stage,
                        source_action=src,
                    )
        case AssociatedDiseases():
            state.add_node(result.target.id, result.target.approved_symbol)
            for row in result.rows:
                state.add_node(row.disease.id, row.disease.name)
                _association_edge(state, result.target.id, row.disease.id, row, src)
        case AssociatedTargets():
            state.add_node(result.disease.id, result.disease.name)
            for row in result.rows:
                state.add_node(row.target.id, row.target.approved_symbol)
                _association_edge(state, row.target.id, result.disease.id, row, src)
        case RawGraphQLResult():
            _fold_raw(state, result.data, src)
        case EvidencePage():
            for ev in result.rows:
                state.add_node(ev.target.id, ev.target.approved_symbol)
                state.add_node(ev.disease.id, ev.disease.name)
                state.add_edge(
                    ev.target.id,
                    ev.disease.id,
                    EdgeType.HAS_EVIDENCE,
                    evidence_id=ev.id,
                    score=ev.score,
                    datasource_id=ev.datasource_id,
                    datatype_id=ev.datatype_id,
                    drug_id=ev.drug.id if ev.drug else None,
                    source_action=src,
                )


def _association_edge(state: SearchState, target: str, disease: str, row: Any, src: str) -> None:
    state.add_edge(
        target,
        disease,
        EdgeType.ASSOCIATED_WITH,
        evidence_id=f"association:{target}:{disease}",  # associations have no OT evidence ID
        score=row.score,
        datatype_scores={c.id: c.score for c in row.datatype_scores},
        source_action=src,
    )


_ENTITY_PREFIXES = ("ENSG", "CHEMBL", "EFO_", "MONDO_", "HP_", "Orphanet_", "OTAR_", "GO_")


def _fold_raw(state: SearchState, node: Any, src: str) -> None:
    """Fold an arbitrary GraphQL response: entities become nodes, evidence rows become edges."""
    if isinstance(node, list):
        for item in node:
            _fold_raw(state, item, src)
        return
    if not isinstance(node, dict):
        return
    entity_id = node.get("id")
    if isinstance(entity_id, str) and entity_id.startswith(_ENTITY_PREFIXES):
        state.add_node(entity_id, node.get("approvedSymbol") or node.get("name"))
    target, disease = node.get("target"), node.get("disease")
    if "datasourceId" in node and isinstance(target, dict) and isinstance(disease, dict):
        state.add_node(target["id"], target.get("approvedSymbol"))
        state.add_node(disease["id"], disease.get("name"))
        state.add_edge(
            target["id"],
            disease["id"],
            EdgeType.HAS_EVIDENCE,
            evidence_id=node["id"],
            score=node.get("score"),
            datasource_id=node["datasourceId"],
            source_action=src,
        )
    for value in node.values():
        _fold_raw(state, value, src)
