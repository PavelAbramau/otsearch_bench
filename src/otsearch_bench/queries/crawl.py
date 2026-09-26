"""Populate the adapter cache with the neighbourhoods the query builders mine.

Seed diseases span therapeutic areas. From each: associated targets -> known drugs -> drug records
-> (target, disease) pairs annotated with their association row, total evidence count and every
non-literature evidence row. All network access goes through the adapter (LIVE mode).
"""

from __future__ import annotations

import random
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from otsearch_bench.env.adapter import OTAdapter
from otsearch_bench.env.errors import OTError
from otsearch_bench.env.models import EntityType
from otsearch_bench.queries.index import LITERATURE_DATASOURCES, moa_targets

SEED_DISEASES = (
    "breast carcinoma",
    "non-small cell lung carcinoma",
    "type 2 diabetes mellitus",
    "rheumatoid arthritis",
    "asthma",
    "Alzheimer disease",
    "schizophrenia",
    "hypertension",
    "psoriasis",
    "multiple sclerosis",
    "Crohn disease",
    "chronic myelogenous leukemia",
    "hypercholesterolemia",
    "migraine disorder",
    "osteoporosis",
    "glaucoma",
    "major depressive disorder",
    "atrial fibrillation",
    "ulcerative colitis",
    "melanoma",
    "prostate carcinoma",
    "epilepsy",
    "Parkinson disease",
    "cystic fibrosis",
    "pulmonary arterial hypertension",
    "chronic obstructive pulmonary disease",
    "atopic eczema",
    "gout",
    "obesity",
    "heart failure",
)


@dataclass(frozen=True)
class CrawlConfig:
    seed_diseases: tuple[str, ...] = SEED_DISEASES
    targets_per_disease_top: int = 12
    targets_per_disease_random: int = 4
    disease_target_page_size: int = 200
    target_disease_page_size: int = 25
    drugs_per_target: int = 2
    indications_per_drug: int = 2
    max_full_pairs: int = 700
    max_light_pairs: int = 300
    evidence_page_size: int = 5000
    max_evidence_pages: int = 3
    workers: int = 4
    seed: int = 20260914


@dataclass
class CrawlSummary:
    diseases: list[str] = field(default_factory=list)
    targets: list[str] = field(default_factory=list)
    drugs: list[str] = field(default_factory=list)
    full_pairs: list[tuple[str, str]] = field(default_factory=list)
    light_pairs: list[tuple[str, str]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def _pmap[T, R](
    fn: Callable[[T], R], items: Iterable[T], workers: int, errors: list[str]
) -> list[tuple[T, R | None]]:
    items = list(items)

    def safe(item: T) -> R | None:
        try:
            return fn(item)
        except OTError as exc:
            errors.append(f"{item!r}: {type(exc).__name__}: {exc}")
            return None

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(zip(items, pool.map(safe, items), strict=True))


def _dedupe[T](items: Iterable[T]) -> list[T]:
    return list(dict.fromkeys(items))


def crawl(
    adapter: OTAdapter,
    config: CrawlConfig | None = None,
    log: Callable[[str], None] = print,
) -> CrawlSummary:
    config = config or CrawlConfig()
    rng = random.Random(config.seed)
    s = CrawlSummary()
    w = config.workers

    resolved = _pmap(
        lambda name: adapter.resolve_entity(name, [EntityType.DISEASE]),
        config.seed_diseases,
        w,
        s.errors,
    )
    for name, res in resolved:
        if res is not None and res.id:
            s.diseases.append(res.id)
        else:
            log(f"  unresolved seed disease: {name}")
    s.diseases = _dedupe(s.diseases)
    log(f"[crawl] {len(s.diseases)} seed diseases resolved")

    disease_targets = _pmap(
        lambda d: adapter.get_associated_targets(d, page_size=config.disease_target_page_size),
        s.diseases,
        w,
        s.errors,
    )
    for _, at in disease_targets:
        if at is None:
            continue
        ids = [r.target.id for r in at.rows]
        top, rest = ids[: config.targets_per_disease_top], ids[config.targets_per_disease_top :]
        s.targets.extend(top + rng.sample(rest, min(config.targets_per_disease_random, len(rest))))
    s.targets = _dedupe(s.targets)
    log(f"[crawl] {len(s.targets)} targets selected")

    known = {t: kd for t, kd in _pmap(adapter.get_known_drugs, s.targets, w, s.errors) if kd}
    top_disease = {}
    for t, ad in _pmap(
        lambda t: adapter.get_associated_diseases(t, page_size=config.target_disease_page_size),
        s.targets,
        w,
        s.errors,
    ):
        if ad is not None and ad.rows:
            top_disease[t] = max(ad.rows, key=lambda r: r.score).disease.id
    log(f"[crawl] known drugs for {sum(bool(k.rows) for k in known.values())} targets")

    chosen: dict[str, list[str]] = {}
    for t in s.targets:
        kd = known.get(t)
        if kd is None:
            continue
        drug_ids = sorted({r.drug.id for r in kd.rows if r.drug and r.disease_ids})
        if drug_ids:
            chosen[t] = rng.sample(drug_ids, min(config.drugs_per_target, len(drug_ids)))
    drug_ids = _dedupe(d for ds in chosen.values() for d in ds)
    drugs = {d: drug for d, drug in _pmap(adapter.get_drug, drug_ids, w, s.errors) if drug}
    s.drugs = list(drugs)
    log(f"[crawl] {len(drugs)} drug records")

    full: list[tuple[str, str]] = []
    light: list[tuple[str, str]] = []
    for t, ds in chosen.items():
        if t in top_disease:
            full.append((t, top_disease[t]))
        for d in ds:
            drug = drugs.get(d)
            moa = moa_targets(drug) if drug else {}
            if drug is None or t not in moa or not drug.indications:
                continue
            diseases = sorted({i.disease.id for i in drug.indications.rows if i.disease})
            for x in rng.sample(diseases, min(config.indications_per_drug, len(diseases))):
                full.append((t, x))
                if 2 <= len(moa) <= 6:
                    light.extend((t2, x) for t2 in moa if t2 != t)
    full = _dedupe(full)
    rng.shuffle(full)
    s.full_pairs = full[: config.max_full_pairs]
    full_set = set(s.full_pairs)
    light = [p for p in _dedupe(light) if p not in full_set]
    rng.shuffle(light)
    s.light_pairs = light[: config.max_light_pairs]
    log(f"[crawl] annotating {len(s.full_pairs)} full + {len(s.light_pairs)} light pairs")

    def annotate(pair: tuple[str, str], *, directional: bool) -> bool:
        t, x = pair
        assoc = adapter.get_associated_diseases(t, disease_ids=[x], page_size=1)
        adapter.get_evidence(t, x, size=1)
        if directional:
            datasources = sorted(
                {c.id for r in assoc.rows for c in r.datasource_scores} - LITERATURE_DATASOURCES
            )
            if datasources:
                adapter.get_all_evidence(
                    t,
                    x,
                    datasource_ids=datasources,
                    page_size=config.evidence_page_size,
                    max_pages=config.max_evidence_pages,
                )
        return True

    _pmap(lambda p: annotate(p, directional=True), s.full_pairs, w, s.errors)
    _pmap(lambda p: annotate(p, directional=False), s.light_pairs, w, s.errors)
    log(f"[crawl] complete; {len(s.errors)} recoverable errors")
    return s
