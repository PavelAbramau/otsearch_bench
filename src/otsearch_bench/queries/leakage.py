"""Leakage probe: can the synthesis model answer the queries with tools disabled?

For every query the model sees only the question (plus a versioned instruction prompt). Its
answer is scored against the constructed reference; per-stratum parametric accuracy above the
threshold marks the stratum as contaminated. Run with
``uv run python -m otsearch_bench.queries.leakage``.
"""

from __future__ import annotations

import argparse
import math
import threading
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, ValidationError

from otsearch_bench.env import Mode, OTAdapter
from otsearch_bench.env.cache import sha256_hex, utc_now_iso
from otsearch_bench.queries.models import AnswerType, QueryClass, QueryRecord, load_querysets
from otsearch_bench.queries.scoring import (
    EntityResolver,
    ProbeAnswer,
    Score,
    adapter_resolver,
    score_answer,
)

DEFAULT_MODEL = "claude-sonnet-5"
PROMPT_VERSION = "leakage_probe_v1"
CONTAMINATION_THRESHOLD = 0.3
MIN_STRATUM_N = 5
PROMPTS_DIR = Path(__file__).resolve().parents[1] / "prompts"


def load_prompt(version: str = PROMPT_VERSION) -> str:
    return (PROMPTS_DIR / f"{version}.md").read_text(encoding="utf-8")


class ProbeResult(BaseModel):
    stop_reason: str | None
    answer: ProbeAnswer | None
    raw_text: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0


class ProbeRecord(BaseModel):
    query_id: str
    question_sha256: str
    model: str
    prompt_version: str
    created_at: str
    result: ProbeResult | None = None
    error: str | None = None

    @property
    def key(self) -> tuple[str, str, str, str]:
        return (self.query_id, self.question_sha256, self.model, self.prompt_version)


class ProbeClient(Protocol):
    model: str

    def ask(self, system: str, question: str) -> ProbeResult: ...


class AnthropicProbeClient:
    """Tools-free structured answer from a Claude model.

    Server-side refusal fallbacks are deliberately not enabled: an answer produced by a
    different model would contaminate the measurement. Refusals are recorded as such.
    """

    def __init__(self, model: str = DEFAULT_MODEL, *, max_tokens: int = 16000, client=None):
        import anthropic

        self.model = model
        self.max_tokens = max_tokens
        self._client = client or anthropic.Anthropic(max_retries=5)

    def ask(self, system: str, question: str) -> ProbeResult:
        response = self._client.messages.parse(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=[{"role": "user", "content": question}],
            output_format=ProbeAnswer,
        )
        text = next((b.text for b in response.content if b.type == "text"), None)
        answer = None
        if response.stop_reason not in ("refusal", "max_tokens"):
            try:
                answer = response.parsed_output
            except (ValueError, ValidationError):
                answer = None
        return ProbeResult(
            stop_reason=response.stop_reason,
            answer=answer,
            raw_text=text,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )


def _probe_key(record: QueryRecord, model: str, prompt_version: str) -> tuple[str, str, str, str]:
    return (record.query_id, sha256_hex(record.question), model, prompt_version)


def run_probe(
    records: list[QueryRecord],
    client: ProbeClient,
    responses_path: str | Path,
    *,
    prompt_version: str = PROMPT_VERSION,
    workers: int = 8,
    log: Callable[[str], None] = print,
) -> dict[str, ProbeRecord]:
    """Ask every question once. Successful responses are persisted and never re-asked."""
    responses_path = Path(responses_path)
    system = load_prompt(prompt_version)
    done: dict[tuple[str, str, str, str], ProbeRecord] = {}
    if responses_path.exists():
        for line in responses_path.read_text(encoding="utf-8").splitlines():
            rec = ProbeRecord.model_validate_json(line)
            if rec.error is None:
                done[rec.key] = rec
    todo = [r for r in records if _probe_key(r, client.model, prompt_version) not in done]
    log(f"[probe] {len(records) - len(todo)} cached, {len(todo)} to ask {client.model}")

    responses_path.parent.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()
    failures: dict[str, ProbeRecord] = {}

    with responses_path.open("a", encoding="utf-8") as fh:

        def persist(rec: ProbeRecord) -> None:
            with lock:
                fh.write(rec.model_dump_json() + "\n")
                fh.flush()
                if rec.error is None:
                    done[rec.key] = rec
                else:
                    failures[rec.query_id] = rec

        def ask(record: QueryRecord, *, raise_errors: bool = False) -> None:
            base = {
                "query_id": record.query_id,
                "question_sha256": sha256_hex(record.question),
                "model": client.model,
                "prompt_version": prompt_version,
            }
            try:
                result = client.ask(system, record.question)
            except Exception as exc:
                if raise_errors:
                    raise
                persist(
                    ProbeRecord(
                        **base, created_at=utc_now_iso(), error=f"{type(exc).__name__}: {exc}"
                    )
                )
                return
            persist(ProbeRecord(**base, created_at=utc_now_iso(), result=result))

        if todo:
            ask(todo[0], raise_errors=True)  # fail fast on credentials / model errors
            with ThreadPoolExecutor(max_workers=workers) as pool:
                list(pool.map(ask, todo[1:]))

    out: dict[str, ProbeRecord] = {}
    for r in records:
        key = _probe_key(r, client.model, prompt_version)
        if key in done:
            out[r.query_id] = done[key]
        elif r.query_id in failures:
            out[r.query_id] = failures[r.query_id]
    log(f"[probe] {len(out) - len(failures)} answered, {len(failures)} failed")
    return out


# --- scoring and reporting ------------------------------------------------------------------


@dataclass
class Scored:
    record: QueryRecord
    probe: ProbeRecord | None
    score: Score

    @property
    def answered(self) -> bool:
        return (
            self.probe is not None
            and self.probe.result is not None
            and self.probe.result.answer is not None
        )


def score_probes(
    records: list[QueryRecord], probes: dict[str, ProbeRecord], resolver: EntityResolver | None
) -> list[Scored]:
    scored = []
    for r in records:
        probe = probes.get(r.query_id)
        answer = probe.result.answer if probe and probe.result else None
        scored.append(Scored(r, probe, score_answer(r, answer, resolver)))
    return scored


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, (centre - margin) / denom), min(1.0, (centre + margin) / denom))


def category_majority_baseline(records: Iterable[QueryRecord]) -> float:
    """Accuracy of always giving the group's most common category (other questions: wrong)."""
    records = list(records)
    categories = Counter(
        r.reference.category for r in records if r.reference.answer_type is AnswerType.CATEGORY
    )
    if not records or not categories:
        return 0.0
    return categories.most_common(1)[0][1] / len(records)


_CLASS_ORDER = {QueryClass.INVERSE: 0, QueryClass.UNANSWERABLE: 1, QueryClass.CANARY: 2}


def _table(
    groups: dict[str, list[Scored]], threshold: float, min_n: int
) -> tuple[list[str], list[str]]:
    lines = [
        "| Stratum | n | Parametric accuracy | 95% CI | Category-majority baseline "
        "| Mean set F1 | Unparsed / refused | Flag |",
        "|---|---:|---:|---|---:|---:|---:|---|",
    ]
    flagged = []
    for name, items in groups.items():
        n = len(items)
        k = sum(s.score.correct for s in items)
        acc = k / n if n else 0.0
        lo, hi = wilson_interval(k, n)
        f1s = [s.score.f1 for s in items if s.score.f1 is not None]
        mean_f1 = f"{sum(f1s) / len(f1s):.2f}" if f1s else "—"
        unparsed = sum(not s.answered for s in items)
        flag = "**CONTAMINATED**" if acc > threshold else "ok"
        if acc > threshold:
            flagged.append(name)
        if n < min_n:
            flag += f" (n<{min_n})"
        lines.append(
            f"| {name} | {n} | {acc:.2f} | {lo:.2f}-{hi:.2f} | "
            f"{category_majority_baseline(s.record for s in items):.2f} | {mean_f1} | "
            f"{unparsed} | {flag} |"
        )
    return lines, flagged


def _group(scored: list[Scored], key: Callable[[Scored], str]) -> dict[str, list[Scored]]:
    groups: dict[str, list[Scored]] = defaultdict(list)
    for s in scored:
        groups[key(s)].append(s)
    return dict(sorted(groups.items()))


def build_report(
    scored: list[Scored],
    *,
    model: str,
    prompt_version: str,
    data_version: str,
    threshold: float = CONTAMINATION_THRESHOLD,
    min_n: int = MIN_STRATUM_N,
) -> str:
    by_stratum = dict(
        sorted(
            _group(scored, lambda s: s.record.stratum).items(),
            key=lambda kv: (_CLASS_ORDER[kv[1][0].record.query_class], kv[0]),
        )
    )
    main_table, flagged = _table(by_stratum, threshold, min_n)
    inverse = [s for s in scored if s.record.query_class is QueryClass.INVERSE]
    answerable_strata = [
        k for k, v in by_stratum.items() if v[0].record.query_class is QueryClass.INVERSE
    ]
    flagged_answerable = [k for k in flagged if k in answerable_strata]
    counts = Counter(s.record.query_class for s in scored)
    probes = [s.probe for s in scored if s.probe and s.probe.result]
    tokens_in = sum(p.result.input_tokens for p in probes)
    tokens_out = sum(p.result.output_tokens for p in probes)
    refusals = sum(1 for p in probes if p.result.stop_reason == "refusal")
    errors = sum(1 for s in scored if s.probe is None or s.probe.error)

    verdict = (
        f"**GATE: FAIL** — {len(flagged_answerable)} of {len(answerable_strata)} "
        "answerable strata exceed the contamination threshold."
        if flagged_answerable
        else f"**GATE: PASS** — none of the {len(answerable_strata)} answerable strata "
        "exceed the contamination threshold."
    )

    out = [
        "# Leakage probe",
        "",
        f"- Generated: {utc_now_iso()}",
        f"- Open Targets data release: {data_version}",
        f"- Model: `{model}` (tools disabled; question only), prompt `{prompt_version}`",
        f"- Contamination threshold: parametric accuracy > {threshold}",
        f"- Queries: {counts[QueryClass.INVERSE]} inverse-constructed, "
        f"{counts[QueryClass.UNANSWERABLE]} unanswerable, {counts[QueryClass.CANARY]} canary",
        f"- Probe usage: {tokens_in:,} input / {tokens_out:,} output tokens; "
        f"{refusals} refusals; {errors} failed requests",
        "",
        "## Verdict",
        "",
        verdict,
        "",
        "Flagged answerable strata: " + (", ".join(f"`{k}`" for k in flagged_answerable) or "none"),
        "",
        "## Parametric accuracy by stratum",
        "",
        "A stratum is query class x hops required x answer shape. Finer difficulty axes are "
        "broken out below for the inverse-constructed set.",
        "",
        *main_table,
        "",
    ]
    sections = [
        (
            "By answer template (all classes)",
            _group(scored, lambda s: f"{s.record.query_class.value} | {s.record.template_id}"),
        ),
        (
            "Inverse-constructed: by evidence-count bucket",
            _group(inverse, lambda s: f"evidence={s.record.difficulty.evidence_bucket}"),
        ),
        (
            "Inverse-constructed: by branching bucket",
            _group(inverse, lambda s: f"branching={s.record.difficulty.branching_bucket}"),
        ),
        (
            "Inverse-constructed: by contradictory evidence",
            _group(inverse, lambda s: f"contradictory={s.record.difficulty.contradictory}"),
        ),
        (
            "Inverse-constructed: full sampling cell",
            _group(
                inverse,
                lambda s: "hops={} | {} | ev={} | br={} | contra={}".format(
                    *s.record.difficulty.cell()
                ),
            ),
        ),
    ]
    for title, groups in sections:
        table, _ = _table(groups, threshold, min_n)
        out += [f"## {title}", "", *table, ""]

    out += ["## Examples from flagged strata", ""]
    if not flagged:
        out.append("No stratum was flagged.")
    for name in flagged:
        hits = [s for s in by_stratum[name] if s.score.correct][:3]
        out += [f"### `{name}`", ""]
        for s in hits:
            answer = s.probe.result.answer.short_answer if s.answered else ""
            out += [
                f"- **Q:** {s.record.question}",
                f"  - Reference: {s.record.reference.explanation}",
                f"  - Tools-free answer: {answer}",
            ]
        out.append("")

    out += [
        "## Method",
        "",
        "- **Inverse-constructed:** a (drug, target, disease) path is sampled from the adapter "
        "cache first; the question is templated from it and the path is the reference answer. "
        "Sampling is round-robin over (hops, shape, evidence bucket, branching bucket, "
        "contradictory) cells.",
        "- **Unanswerable:** no path within 2 hops (disease plus ontology descendants; all "
        'indirect association pages listed). Parametric "accuracy" here is the rate at which '
        "the model says no evidence exists, so a high value means the model abstains without "
        "tools rather than that the answer leaked.",
        "- **Canary:** stable identifiers and approved-drug facts, chosen to be memorisable. A "
        "high canary score is expected and is not part of the gate; canaries exist to measure "
        "environment drift.",
        "- **Scoring:** entity answers match on name, OT id, synonyms, or an exact `mapIds` "
        "resolution; sets are correct at F1 ≥ 0.5; categories and counts must match exactly; "
        "unparsed or refused answers count as wrong.",
        "- **Category-majority baseline:** accuracy from always answering the stratum's most "
        "common reference category (non-category questions count as wrong). Parametric accuracy "
        "near this baseline indicates guessing, not knowledge.",
        "- **Gate:** only inverse-constructed strata count towards the gate. Strata with n < "
        f"{min_n} are marked; their flags are unreliable.",
        "",
    ]
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--querysets", default="data/querysets")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--prompt-version", default=PROMPT_VERSION)
    parser.add_argument("--responses", default="data/leakage/probe_responses.jsonl")
    parser.add_argument("--scores", default="data/leakage/scores.jsonl")
    parser.add_argument("--report", default="reports/leakage.md")
    parser.add_argument("--cache", default="data/cache/opentargets.sqlite")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=CONTAMINATION_THRESHOLD)
    args = parser.parse_args(argv)

    querysets = load_querysets(args.querysets)
    if not querysets:
        print(f"no query sets found in {args.querysets}")
        return 1
    records = [r for qs in querysets for r in qs.records]
    data_version = querysets[0].data_version

    try:
        client = AnthropicProbeClient(args.model)
        probes = run_probe(
            records,
            client,
            args.responses,
            prompt_version=args.prompt_version,
            workers=args.workers,
        )
    except Exception as exc:
        print(f"leakage probe could not call {args.model}: {type(exc).__name__}: {exc}")
        return 2

    with OTAdapter(args.cache, mode=Mode.LIVE) as adapter:
        scored = score_probes(records, probes, adapter_resolver(adapter))

    scores_path = Path(args.scores)
    scores_path.parent.mkdir(parents=True, exist_ok=True)
    scores_path.write_text(
        "".join(s.score.model_dump_json() + "\n" for s in scored), encoding="utf-8"
    )

    report = build_report(
        scored,
        model=args.model,
        prompt_version=args.prompt_version,
        data_version=data_version,
        threshold=args.threshold,
    )
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    print(f"wrote {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
