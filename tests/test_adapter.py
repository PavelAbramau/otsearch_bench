import httpx
import pytest

from otsearch_bench.env import (
    CacheMissError,
    EntityNotFoundError,
    Mode,
    OTAdapter,
    OTGraphQLError,
    OTHTTPError,
    RateLimiter,
    RetryExhaustedError,
    drift_report,
)
from tests.conftest import forbid_network

EGFR = "ENSG00000146648"


def target_body(symbol="EGFR"):
    return {
        "data": {
            "target": {
                "id": EGFR,
                "approvedSymbol": symbol,
                "approvedName": "epidermal growth factor receptor",
                "biotype": "protein_coding",
                "functionDescriptions": [],
                "symbolSynonyms": [],
                "nameSynonyms": [],
            }
        }
    }


def adapter(tmp_path, fake, mode=Mode.LIVE, **kw):
    kw.setdefault("rate_limit_per_s", None)
    return OTAdapter(tmp_path / "c.sqlite", mode=mode, transport=fake.transport, **kw)


def test_live_fetches_once_then_serves_from_cache(tmp_path, fake_ot):
    fake_ot.bodies["Target"] = target_body()
    with adapter(tmp_path, fake_ot) as ot:
        assert ot.get_target(EGFR).approved_symbol == "EGFR"
        assert ot.network_call_count == 2  # Meta + Target
        assert ot.last_call.cache_hit is False
        assert ot.get_target(EGFR).approved_symbol == "EGFR"
        assert ot.network_call_count == 2
        assert ot.last_call.cache_hit is True
        assert (ot.call_count, ot.cache_hit_count) == (3, 1)  # Meta counts as a call
        assert ot.data_version == "26.06"
        assert ot.tokens > 0


def test_replay_serves_cache_without_network_and_raises_on_miss(tmp_path, fake_ot):
    fake_ot.bodies["Target"] = target_body()
    with adapter(tmp_path, fake_ot) as ot:
        ot.get_target(EGFR)
    with OTAdapter(
        tmp_path / "c.sqlite", mode=Mode.REPLAY, transport=httpx.MockTransport(forbid_network)
    ) as replay:
        assert replay.get_target(EGFR).id == EGFR
        assert replay.get_meta().data_version.month == "06"
        with pytest.raises(CacheMissError):
            replay.get_target("ENSG_UNSEEN")
        assert replay.network_call_count == 0


def test_refresh_appends_and_drift_report_flags_changed_hashes(tmp_path, fake_ot):
    fake_ot.bodies["Target"] = target_body("EGFR")
    with adapter(tmp_path, fake_ot) as ot:
        ot.get_target(EGFR)
    with adapter(tmp_path, fake_ot, mode=Mode.REFRESH) as ot:
        ot.get_target(EGFR)  # same payload: new row, same hash
        assert ot.drift_report() == []
        fake_ot.bodies["Target"] = target_body("ERBB1")
        assert ot.get_target(EGFR).approved_symbol == "ERBB1"
        key = ot.last_call.cache_key
        assert len(ot.cache.history(key)) == 3
    [entry] = drift_report(tmp_path / "c.sqlite")
    assert entry.cache_key == key
    assert (entry.operation, entry.n_fetches, entry.n_distinct_hashes, entry.n_changes) == (
        "Target",
        3,
        2,
        1,
    )


def test_null_entity_is_cached_and_raises_not_found(tmp_path, fake_ot):
    fake_ot.bodies["Target"] = {"data": {"target": None}}
    with adapter(tmp_path, fake_ot) as ot:
        for _ in range(2):
            with pytest.raises(EntityNotFoundError):
                ot.get_target("ENSG_NOPE")
        assert fake_ot.by_operation["Target"] == 1


def test_graphql_errors_raise_and_are_not_cached(tmp_path, fake_ot):
    fake_ot.bodies["Target"] = {"errors": [{"message": "boom"}], "data": None}
    with adapter(tmp_path, fake_ot) as ot:
        with pytest.raises(OTGraphQLError, match="boom"):
            ot.get_target(EGFR)
        assert ot.cache.count() == 1  # only the Meta row


def test_retries_with_backoff_then_succeeds(tmp_path, fake_ot, no_sleep):
    fake_ot.bodies["Target"] = target_body()
    fake_ot.scripted["Target"] = [
        httpx.Response(503),
        httpx.Response(429, headers={"Retry-After": "2"}),
    ]
    with adapter(tmp_path, fake_ot, sleep=no_sleep, backoff_base_s=0.1, seed=0) as ot:
        ot.get_target(EGFR)
        assert ot.retry_count == 2
        assert ot.network_call_count == 4  # Meta + 3 Target attempts
        assert ot.last_call.network_attempts == 3
    assert 0.05 <= no_sleep.calls[0] <= 0.1  # jittered first backoff
    assert no_sleep.calls[1] == 2.0  # Retry-After honoured


def test_retry_exhaustion_and_non_retryable_status(tmp_path, fake_ot, no_sleep):
    fake_ot.scripted["Target"] = [httpx.Response(503)] * 3
    with adapter(tmp_path, fake_ot, sleep=no_sleep, max_retries=2) as ot:
        with pytest.raises(RetryExhaustedError):
            ot.get_target(EGFR)
        fake_ot.scripted["Target"] = [httpx.Response(404, text="nope")]
        with pytest.raises(OTHTTPError):
            ot.get_target(EGFR)


def test_rate_limiter_spaces_requests():
    now = [0.0]
    sleeps = []

    def sleep(s):
        sleeps.append(s)
        now[0] += s

    limiter = RateLimiter(2.0, clock=lambda: now[0], sleep=sleep)
    for _ in range(3):
        limiter.acquire()
    assert sleeps == [0.5, 0.5]


def test_resolve_falls_back_to_search(tmp_path, fake_ot):
    fake_ot.bodies["MapIds"] = {"data": {"mapIds": {"mappings": [{"term": "egfrr", "hits": []}]}}}
    fake_ot.bodies["Search"] = {
        "data": {
            "search": {
                "total": 1,
                "hits": [{"id": EGFR, "entity": "target", "name": "EGFR", "score": 9.0}],
            }
        }
    }
    with adapter(tmp_path, fake_ot) as ot:
        res = ot.resolve_entity("egfrr", ["target"])
    assert (res.method, res.id, res.entity) == ("search", EGFR, "target")
    assert fake_ot.requests[-1]["variables"]["entityNames"] == ["target"]


def test_usd_per_network_call_accumulates(tmp_path, fake_ot):
    fake_ot.bodies["Target"] = target_body()
    with adapter(tmp_path, fake_ot, usd_per_network_call=0.01) as ot:
        ot.get_target(EGFR)
        ot.get_target(EGFR)
        assert ot.cost_usd == pytest.approx(0.02)
