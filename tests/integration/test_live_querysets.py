"""Small live crawl -> cache index -> inverse candidates, end to end."""

import random

import pytest

from otsearch_bench.env import Mode, OTAdapter
from otsearch_bench.queries.crawl import CrawlConfig, crawl
from otsearch_bench.queries.index import CacheIndex
from otsearch_bench.queries.inverse import generate_candidates, stratified_sample
from otsearch_bench.queries.models import QueryClass, QuerySet

pytestmark = pytest.mark.integration


def test_small_crawl_yields_sampled_inverse_queries(tmp_path):
    config = CrawlConfig(
        seed_diseases=("rheumatoid arthritis",),
        targets_per_disease_top=4,
        targets_per_disease_random=0,
        drugs_per_target=1,
        indications_per_drug=1,
        max_full_pairs=6,
        max_light_pairs=0,
    )
    with OTAdapter(tmp_path / "ot.sqlite", mode=Mode.LIVE) as ot:
        data_version = str(ot.get_meta().data_version)
        summary = crawl(ot, config, log=lambda m: None)
        assert summary.targets and summary.drugs and not summary.errors
        index = CacheIndex.from_cache(ot.cache, data_version=data_version)

    candidates = generate_candidates(index)
    assert candidates
    sample = stratified_sample(candidates, 5, random.Random(0))
    assert sample and all(r.reference.path is not None for r in sample)
    assert all("Unknown" not in r.question for r in sample)
    qs = QuerySet(
        name="smoke",
        query_class=QueryClass.INVERSE,
        data_version=data_version,
        generator_version="test",
        records=sample,
    )
    assert QuerySet.from_jsonl(qs.to_jsonl(tmp_path / "smoke.jsonl")) == qs
