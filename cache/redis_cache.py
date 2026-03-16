"""Caching layer with Redis backend and in-memory fallback."""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any, Optional

from config.settings import get_settings
from monitoring.logger import get_logger

logger = get_logger(__name__)


class CacheManager:
    """Two-tier cache: tries Redis first, falls back to in-memory LRU."""

    def __init__(self):
        self.settings = get_settings()
        self._redis = None
        self._memory: dict[str, tuple[Any, float]] = {}  # key -> (value, expiry)
        self._connect_redis()

    def _connect_redis(self) -> None:
        if not self.settings.redis_url:
            logger.info("redis_not_configured_using_memory_cache")
            return

        try:
            import redis
            self._redis = redis.from_url(
                self.settings.redis_url,
                decode_responses=True,
                socket_connect_timeout=2,
            )
            self._redis.ping()
            logger.info("redis_connected", url=self.settings.redis_url)
        except Exception as exc:
            logger.warning("redis_unavailable_using_memory_cache", error=str(exc))
            self._redis = None

    def get(self, key: str) -> Optional[Any]:
        cache_key = self._hash_key(key)

        if self._redis:
            try:
                val = self._redis.get(cache_key)
                if val is not None:
                    logger.debug("cache_hit_redis", key=key[:50])
                    return json.loads(val)
            except Exception:
                pass

        if cache_key in self._memory:
            value, expiry = self._memory[cache_key]
            if time.time() < expiry:
                logger.debug("cache_hit_memory", key=key[:50])
                return value
            else:
                del self._memory[cache_key]

        return None

    def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        cache_key = self._hash_key(key)
        ttl = ttl or self.settings.cache_ttl
        serialized = json.dumps(value)

        if self._redis:
            try:
                self._redis.setex(cache_key, ttl, serialized)
            except Exception:
                pass

        self._memory[cache_key] = (value, time.time() + ttl)
        self._evict_if_needed()

    def invalidate(self, key: str) -> None:
        cache_key = self._hash_key(key)
        if self._redis:
            try:
                self._redis.delete(cache_key)
            except Exception:
                pass
        self._memory.pop(cache_key, None)

    def _evict_if_needed(self, max_size: int = 1000) -> None:
        if len(self._memory) > max_size:
            sorted_keys = sorted(self._memory, key=lambda k: self._memory[k][1])
            for k in sorted_keys[: len(self._memory) - max_size]:
                del self._memory[k]

    @staticmethod
    def _hash_key(key: str) -> str:
        return f"rgpt:{hashlib.sha256(key.encode()).hexdigest()[:24]}"
