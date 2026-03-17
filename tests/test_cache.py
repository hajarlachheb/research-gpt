"""Tests for the cache manager."""

from __future__ import annotations

import os
import time
import pytest
from unittest.mock import patch, MagicMock

os.environ.setdefault("LLM_API_KEY", "test-key-not-real")
os.environ.setdefault("CHROMA_PERSIST_DIR", "./data/test_chroma_cache")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/15")

from cache.redis_cache import CacheManager


@pytest.fixture
def cache():
    with patch.object(CacheManager, "_connect_redis"):
        cm = CacheManager()
        cm._redis = None
        return cm


class TestCacheManager:
    def test_set_and_get(self, cache):
        cache.set("key1", {"answer": "test"})
        result = cache.get("key1")
        assert result == {"answer": "test"}

    def test_get_missing_key(self, cache):
        assert cache.get("nonexistent") is None

    def test_invalidate(self, cache):
        cache.set("key2", "value")
        assert cache.get("key2") is not None
        cache.invalidate("key2")
        assert cache.get("key2") is None

    def test_ttl_expiration(self, cache):
        cache.set("expiring", "data", ttl=1)
        assert cache.get("expiring") == "data"
        time.sleep(1.1)
        assert cache.get("expiring") is None

    def test_eviction(self, cache):
        for i in range(1010):
            cache.set(f"k{i}", f"v{i}")
        assert len(cache._memory) <= 1000

    def test_hash_key_deterministic(self):
        key1 = CacheManager._hash_key("test")
        key2 = CacheManager._hash_key("test")
        assert key1 == key2
        assert key1.startswith("rgpt:")

    def test_hash_key_different_inputs(self):
        key1 = CacheManager._hash_key("test1")
        key2 = CacheManager._hash_key("test2")
        assert key1 != key2
