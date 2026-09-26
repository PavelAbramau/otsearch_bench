"""Build the query sets: crawl -> cache index -> inverse / unanswerable / canary.

Run with ``uv run python -m otsearch_bench.queries.build``. Query sets are written as JSONL,
one file per class.
"""

from __future__ import annotations

import argparse
import dataclasses
import random
import time
from collections import Counter
from pathlib import Path

from otsearch_bench.env import Mode, OTAdapter
from otsearch_bench.queries import canary, inverse, unanswerable
from otsearch_bench.queries.crawl import CrawlConfig, crawl
from otsearch_bench.queries.index import CacheIndex
from otsearch_bench.queries.models import QueryClass, QuerySet


def log(message: str) -> None:
    print(message, flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", default="data/cache/opentargets.sqlite")
    parser.add_argument("--out", default="data/querysets")
    parser.add_argument("--inverse", type=int, default=240)
    parser.add_argument("--unanswerable", type=int, default=60)
    parser.add_argument("--canary", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260914)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--rate", type=float, default=4.0, help="max API requests per second")
    parser.add_argument("--skip-crawl", action="store_true")
    args = parser.parse_args(argv)
    out = Path(args.out)
    config = CrawlConfig(workers=args.workers, seed=args.seed)

    with OTAdapter(args.cache, mode=Mode.LIVE, rate_limit_per_s=args.rate) as adapter:
        meta = adapter.get_meta()
        data_version = str(meta.data_version)
        log(f"Open Targets data {data_version}, API {meta.api_version}")

        if not args.skip_crawl:
            started = time.monotonic()
            summary = crawl(adapter, config, log=log)
            log(
                f"[crawl] {time.monotonic() - started:.0f}s, {adapter.network_call_count} network "
                f"calls, {len(summary.errors)} errors"
            )
            for error in summary.errors[:10]:
                log(f"  {error}")
            adapter.reset_counters()

        index = CacheIndex.from_cache(adapter.cache, data_version=data_version)
        log(
            f"[index] drugs={len(index.drugs)} targets_with_known_drugs={len(index.known_drugs)} "
            f"evidence_pairs={len(index.evidence)} "
            f"fully_checked={sum(pe.checked for pe in index.evidence.values())}"
        )

        candidates = inverse.generate_candidates(index)
        cells = Counter(c.difficulty.cell() for c in candidates)
        inverse_records = inverse.stratified_sample(
            candidates, args.inverse, random.Random(args.seed)
        )
        QuerySet(
            name="inverse_constructed",
            query_class=QueryClass.INVERSE,
            data_version=data_version,
            generator_version=inverse.GENERATOR_VERSION,
            config={
                "seed": args.seed,
                "crawl": dataclasses.asdict(config),
                "candidates": len(candidates),
                "candidate_templates": dict(Counter(c.template_id for c in candidates)),
                "cells_available": len(cells),
                "cells_sampled": len({r.difficulty.cell() for r in inverse_records}),
            },
            records=inverse_records,
        ).to_jsonl(out / "inverse_constructed.jsonl")
        log(
            f"[inverse] {len(inverse_records)} sampled from {len(candidates)} candidates "
            f"in {len(cells)} cells"
        )

        unanswerable_records, stats = unanswerable.build_unanswerable(
            index, adapter, args.unanswerable, random.Random(args.seed + 1), log=log
        )
        QuerySet(
            name="unanswerable",
            query_class=QueryClass.UNANSWERABLE,
            data_version=data_version,
            generator_version=unanswerable.GENERATOR_VERSION,
            config={"seed": args.seed + 1, "verification_stats": stats},
            records=unanswerable_records,
        ).to_jsonl(out / "unanswerable.jsonl")
        log(f"[unanswerable] {len(unanswerable_records)} verified; {stats}")

        canary_records = canary.build_canaries(
            index, adapter, random.Random(args.seed + 2), per_kind=args.canary // 3
        )
        QuerySet(
            name="canary",
            query_class=QueryClass.CANARY,
            data_version=data_version,
            generator_version=canary.GENERATOR_VERSION,
            config={"seed": args.seed + 2},
            records=canary_records,
        ).to_jsonl(out / "canary.jsonl")
        log(f"[canary] {len(canary_records)} canaries")

    for name, records in (
        ("inverse", inverse_records),
        ("unanswerable", unanswerable_records),
        ("canary", canary_records),
    ):
        log(f"{name}: {dict(sorted(Counter(r.stratum for r in records).items()))}")
    met = (
        len(inverse_records) >= 200
        and len(unanswerable_records) >= 50
        and len(canary_records) >= args.canary
    )
    log("targets met" if met else "TARGETS NOT MET")
    return 0 if met else 1


if __name__ == "__main__":
    raise SystemExit(main())
