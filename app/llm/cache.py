"""
LLM response caching to reduce GPU usage.

TTL: 45 minutes — keeps sentiment fresh intraday without hammering Qwen3.
Stale entries from previous trading days are auto-purged on import.
"""

import os
import json
import hashlib
from pathlib import Path
from datetime import datetime, timedelta
from ..monitor.logger import get_logger

logger = get_logger(__name__)

CACHE_DIR     = Path("data/llm_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_TTL_HOURS = 0.75   # 45 minutes


class LLMCache:
    """File-based cache for LLM responses with TTL."""

    def __init__(self, cache_dir: Path = CACHE_DIR, ttl_hours: float = CACHE_TTL_HOURS):
        self.cache_dir = cache_dir
        self.ttl       = timedelta(hours=ttl_hours)
        # Always purge stale entries on construction so old sessions
        # never bleed bullish/bearish bias into a new trading day.
        self.clear_expired()

    def _key(self, prompt: str, model: str) -> str:
        return hashlib.sha256(f"{model}:{prompt}".encode()).hexdigest()

    def _path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    def get(self, prompt: str, model: str):
        path = self._path(self._key(prompt, model))
        if not path.exists():
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                entry = json.load(f)
            cached_time = datetime.fromisoformat(entry["timestamp"])
            if datetime.utcnow() - cached_time > self.ttl:
                path.unlink(missing_ok=True)
                return None
            return entry["response"]
        except Exception as e:
            logger.warning("Cache read error: %s", e)
            return None

    def set(self, prompt: str, model: str, response):
        path = self._path(self._key(prompt, model))
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump({
                    "timestamp": datetime.utcnow().isoformat(),
                    "model":     model,
                    "response":  response,
                }, f, indent=2)
        except Exception as e:
            logger.warning("Cache write error: %s", e)

    def clear_expired(self):
        """Remove all entries older than TTL."""
        cleared = 0
        for p in self.cache_dir.glob("*.json"):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    entry = json.load(f)
                cached_time = datetime.fromisoformat(entry["timestamp"])
                if datetime.utcnow() - cached_time > self.ttl:
                    p.unlink(missing_ok=True)
                    cleared += 1
            except Exception:
                # Corrupt or unreadable — delete it
                try:
                    p.unlink(missing_ok=True)
                    cleared += 1
                except Exception:
                    pass
        if cleared:
            logger.info("LLM cache: cleared %d stale entries.", cleared)

    def clear_all(self):
        for p in self.cache_dir.glob("*.json"):
            p.unlink(missing_ok=True)
        logger.info("LLM cache: fully cleared.")


# Global singleton — purges stale entries on every process start
cache = LLMCache()
