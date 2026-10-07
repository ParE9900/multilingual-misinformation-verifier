"""Tests for SQLite persistent disk cache module."""

import time
from src.cache import DiskCache


def test_cache_set_and_get(tmp_path):
    db_path = tmp_path / "cache.db"
    cache = DiskCache(path=str(db_path))

    namespace = "test_ns"
    key = "sample_key"
    value = "sample_string_value"

    cache.set(namespace, key, value)
    retrieved = cache.get(namespace, key)

    assert retrieved == value
    cache.close()


def test_cache_complex_types(tmp_path):
    db_path = tmp_path / "cache.db"
    with DiskCache(path=str(db_path)) as cache:
        payload = {
            "claim": "Earth is flat",
            "score": 0.01,
            "sub_claims": ["claim 1", "claim 2"],
            "verified": False,
        }
        cache.set("claims", "flat_earth", payload)
        assert cache.get("claims", "flat_earth") == payload


def test_cache_missing_key_returns_none(tmp_path):
    db_path = tmp_path / "cache.db"
    with DiskCache(path=str(db_path)) as cache:
        assert cache.get("unknown_ns", "missing_key") is None


def test_cache_ttl_expiration(tmp_path):
    db_path = tmp_path / "cache.db"
    with DiskCache(path=str(db_path)) as cache:
        namespace = "ttl_ns"
        key = "short_lived_key"
        value = "temporary_data"

        cache.set(namespace, key, value, ttl=1)
        assert cache.get(namespace, key) == value

        time.sleep(2)
        assert cache.get(namespace, key) is None
