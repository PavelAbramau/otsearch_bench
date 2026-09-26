"""Baseline QueryFormulation, Synthesis, Budget and Reformulation policies."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from pydantic import BaseModel

from otsearch_bench.prompts import load_prompt, prompt_fingerprint
from otsearch_bench.queries.templates import TEMPLATES
from otsearch_bench.search.actions import Action, ResolveEntity
from otsearch_bench.search.baselines.digest import graph_digest, node_label
from otsearch_bench.search.decisions import (
    AnchorMention,
    Answer,
    BudgetDecision,
    Claim,
    Formulation,
    Reformulation,
    RejectedAction,
)
from otsearch_bench.search.graph import EdgeType
from otsearch_bench.search.llm import LLMClient
from otsearch_bench.search.state import Budget, SearchState

# Template scaffolding removed before reading spans; what remains between stopwords is a mention.
_PHRASES = (
    "according to open targets",
    "in open targets",
    "open targets",
    "evidence data type",
    "clinical development stage",
    "clinical development",
    "clinical evidence",
    "molecular targets",
    "molecular target",
    "phase 3 or later",
    "overall association score",
    "association evidence",
    "human gene",
    "name one other",
)
_STOPWORDS = frozenset(
    [
        "a",
        "all",
        "among",
        "association",
        "an",
        "and",
        "are",
        "as",
        "associated",
        "between",
        "contributes",
        "development",
        "direct",
        "directly",
        "distinct",
        "diseases",
        "do",
        "does",
        "drug",
        "drugs",
        "encodes",
        "evidence",
        "for",
        "from",
        "gene",
        "has",
        "have",
        "highest",
        "how",
        "in",
        "is",
        "it",
        "its",
        "list",
        "many",
        "more",
        "name",
        "of",
        "on",
        "or",
        "other",
        "reached",
        "same",
        "score",
        "than",
        "that",
        "the",
        "them",
        "this",
        "to",
        "what",
        "which",
        "with",
        "act",
        "acting",
        "acts",
        "has",
        "been",
        "were",
    ]
)
_TOKEN = re.compile(r"[A-Za-z0-9][\w\-/'.+]*")


def _template_patterns() -> list[re.Pattern[str]]:
    patterns = []
    for spec in TEMPLATES.values():
        regex = re.escape(spec.text)
        for slot in ("drug", "target", "disease"):
            regex = regex.replace(re.escape("{" + slot + "}"), f"(?P<{slot}>.+?)")
        regex = regex.replace(re.escape("{options}"), ".+")
        patterns.append(re.compile(regex, flags=re.S))
    return patterns


_TEMPLATE_PATTERNS = _template_patterns()


def template_mentions(question: str) -> list[str] | None:
    """Slot values if the question matches a benchmark template exactly, else None."""
    for pattern in _TEMPLATE_PATTERNS:
        match = pattern.fullmatch(question.strip())
        if match:
            ordered = sorted(match.re.groupindex.items(), key=lambda kv: kv[1])
            return [match.group(name).strip() for name, _ in ordered]
    return None


def extract_mentions(question: str) -> list[str]:
    """Entity mentions in a templated question: spans between template words."""
    text = question.split("Answer with one of")[0]
    lowered = text.lower()
    for phrase in _PHRASES:
        # same-length mask keeps offsets aligned with the original-case text
        lowered = lowered.replace(phrase, " |" + " " * (len(phrase) - 2))
    spans, current = [], []
    original = {m.start(): m.group() for m in _TOKEN.finditer(text)}
    for match in _TOKEN.finditer(lowered):
        word = match.group().rstrip(".?")
        if match.group() == "|" or word in _STOPWORDS or not word:
            if current:
                spans.append(" ".join(current))
            current = []
            continue
        current.append(original.get(match.start(), match.group()).rstrip(".?,"))
    if current:
        spans.append(" ".join(current))
    return list(dict.fromkeys(s for s in spans if s))


@dataclass(frozen=True)
class SpanFormulation:
    """Rule-based: one ResolveEntity per mention (no model call).

    Mentions come from the benchmark's own question templates when the question matches one
    (this baseline knows the question grammar; an LLM formulation would not need to), and from
    stopword-delimited spans otherwise.
    """

    max_mentions: int = 4
    name: str = "span-formulation"
    version: str = "1"

    def formulate(self, state: SearchState) -> Formulation:
        question = state.working_question
        mentions = (template_mentions(question) or extract_mentions(question))[: self.max_mentions]
        return Formulation(
            anchor_mentions=[AnchorMention(mention=m) for m in mentions],
            actions=[ResolveEntity(name=m) for m in mentions],
        )


@dataclass(frozen=True)
class NoReformulation:
    name: str = "no-reformulation"
    version: str = "1"

    def reformulate(self, state: SearchState) -> Reformulation | None:
        return None


@dataclass(frozen=True)
class CallBudget:
    """Admit proposals in order until the API-call budget is spent."""

    max_calls: int = 20
    name: str = "call-budget"
    version: str = "1"

    def initial_budget(self) -> Budget:
        return Budget(max_calls=self.max_calls)

    def admit(self, state: SearchState, proposed: list[Action]) -> BudgetDecision:
        room = state.budget.calls_remaining
        return BudgetDecision(
            admitted=proposed[:room],
            rejected=[RejectedAction(action=a, reason="call budget") for a in proposed[room:]],
        )


@dataclass(frozen=True)
class EvidenceSummarySynthesis:
    """Rule-based answer: neighbours of the anchors ranked by score, citing evidence ids."""

    max_entities: int = 5
    name: str = "evidence-summary-synthesis"
    version: str = "1"

    def synthesize(self, state: SearchState) -> Answer:
        anchors = set(state.anchor_ids())
        touching = [e for e in state.edges() if e[0] in anchors or e[1] in anchors]
        if not touching:
            return Answer(
                text="No evidence connecting the question's entities was found.",
                no_evidence=True,
            )
        touching.sort(key=lambda e: -(e[3].get("score") or 0.0))
        neighbours = list(
            dict.fromkeys(
                v if u in anchors else u
                for u, v, _, _ in touching
                if not (u in anchors and v in anchors)
            )
        )[: self.max_entities]
        cited = [
            e for e in touching if e[3]["edge_type"] == EdgeType.HAS_EVIDENCE.value
        ] or touching
        claims = [
            Claim(
                text=f"{node_label(state, u)} {d['edge_type']} {node_label(state, v)}",
                evidence_ids=[d["evidence_id"]],
            )
            for u, v, _, d in cited[:10]
        ]
        labels = ", ".join(node_label(state, n) for n in neighbours) or "none"
        return Answer(
            text=f"Top connected entities: {labels}.",
            no_evidence=False,
            entities=[state.graph.nodes[n].get("name") or n for n in neighbours],
            claims=claims,
            cited_evidence_ids=[c.evidence_ids[0] for c in claims],
        )


class SynthesisOutput(BaseModel):
    text: str
    no_evidence: bool
    entities: list[str]
    category: str | None
    count: int | None
    claims: list[Claim]


@dataclass(frozen=True)
class LLMSynthesis:
    """The model answers from the graph digest only, attaching evidence ids to each claim."""

    llm: LLMClient | None = field(default=None, compare=False)
    prompt_version: str = "synthesis_v1"
    max_edges: int = 150
    name: str = "llm-synthesis"
    version: str = ""

    def __post_init__(self) -> None:
        if self.llm is None:
            raise ValueError("LLMSynthesis needs an llm client")
        if not self.version:
            object.__setattr__(self, "version", prompt_fingerprint(self.prompt_version))

    def synthesize(self, state: SearchState) -> Answer:
        out = self.llm.structured(
            prompt_version=self.prompt_version,
            system=load_prompt(self.prompt_version),
            user=f"Question: {state.working_question}\n\n"
            + graph_digest(state, max_edges=self.max_edges),
            output=SynthesisOutput,
        )
        return Answer(
            **out.model_dump(),
            cited_evidence_ids=list(dict.fromkeys(i for c in out.claims for i in c.evidence_ids)),
        )
