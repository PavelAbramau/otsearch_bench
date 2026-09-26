import sqlite3

import pytest

from otsearch_bench.env.cache import CacheStore, cache_key, response_hash


def _append(store: CacheStore, response: dict, variables: dict | None = None):
    return store.append(
        query="query T($id: String!) { target(ensemblId: $id) { id } }",
        variables=variables or {"id": "ENSG1"},
        request={},
        response=response,
        endpoint="http://fake",
        data_version="26.06",
        operation="T",
    )


def test_cache_key_ignores_whitespace_and_variable_order():
    a = cache_key("query X { a  b }", {"x": 1, "y": [1, 2]})
    b = cache_key("query X {\n  a\n  b\n}", {"y": [1, 2], "x": 1})
    assert a == b
    assert a != cache_key("query X { a b }", {"x": 2, "y": [1, 2]})


def test_response_hash_is_canonical():
    assert response_hash({"a": 1, "b": 2}) == response_hash({"b": 2, "a": 1})


def test_append_only_rows_cannot_be_updated_or_deleted(tmp_path):
    with CacheStore(tmp_path / "c.sqlite") as store:
        row = _append(store, {"data": {"target": {"id": "ENSG1"}}})
        with pytest.raises(sqlite3.DatabaseError, match="append-only"):
            store._conn.execute("UPDATE cache_entries SET response_json = '{}'")
        with pytest.raises(sqlite3.DatabaseError, match="append-only"):
            store._conn.execute("DELETE FROM cache_entries WHERE row_id = ?", (row.row_id,))
        assert store.count() == 1


def test_history_keeps_every_fetch_and_latest_wins(tmp_path):
    with CacheStore(tmp_path / "c.sqlite") as store:
        first = _append(store, {"data": {"v": 1}})
        second = _append(store, {"data": {"v": 2}})
        _append(store, {"data": {"v": 3}}, variables={"id": "OTHER"})
        assert [r.row_id for r in store.history(first.cache_key)] == [first.row_id, second.row_id]
        assert store.latest(first.cache_key) == second
        assert store.drifted_keys() == [first.cache_key]
