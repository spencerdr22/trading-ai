"""
LLM response caching to reduce GPU usage and costs.
"""

import os
import json
import hashlib
from pathlib import Path
from datetime import datetime, timedelta
from ..monitor.logger import get_logger

logger = get_logger(__name__)

CACHE_DIR = Path("data/llm_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Cache expiration — 45 minutes keeps sentiment fresh during the trading day
# without hammering Qwen3 on every single refresh cycle
CACHE_TTL_HOURS = 0.75  # 45 minutes


class LLMCache:
    """
    File-based cache for LLM responses with TTL.
    """
    
    def __init__(self, cache_dir: Path = CACHE_DIR, ttl_hours: int = CACHE_TTL_HOURS):
        self.cache_dir = cache_dir
        self.ttl = timedelta(hours=ttl_hours)
    
    def _get_cache_key(self, prompt: str, model: str) -> str:
        """Generate cache key from prompt + model."""
        content = f"{model}:{prompt}"
        return hashlib.sha256(content.encode()).hexdigest()
    
    def _get_cache_path(self, cache_key: str) -> Path:
        """Get cache file path."""
        return self.cache_dir / f"{cache_key}.json"
    
    def get(self, prompt: str, model: str) -> dict | None:
        """
        Retrieve cached response if valid.
        """
        cache_key = self._get_cache_key(prompt, model)
        cache_path = self._get_cache_path(cache_key)
        
        if not cache_path.exists():
            return None
        
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cached = json.load(f)
            
            # Check TTL
            cached_time = datetime.fromisoformat(cached["timestamp"])
            if datetime.utcnow() - cached_time > self.ttl:
                logger.debug(f"Cache expired: {cache_key}")
                cache_path.unlink()
                return None
            
            logger.debug(f"Cache hit: {cache_key}")
            return cached["response"]
        
        except Exception as e:
            logger.warning(f"Cache read error: {e}")
            return None
    
    def set(self, prompt: str, model: str, response: dict):
        """
        Store response in cache.
        """
        cache_key = self._get_cache_key(prompt, model)
        cache_path = self._get_cache_path(cache_key)
        
        try:
            cache_entry = {
                "timestamp": datetime.utcnow().isoformat(),
                "model": model,
                "prompt_hash": cache_key,
                "response": response
            }
            
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(cache_entry, f, indent=2)
            
            logger.debug(f"Cached response: {cache_key}")
        
        except Exception as e:
            logger.warning(f"Cache write error: {e}")
    
    def clear_expired(self):
        """
        Remove expired cache entries.
        """
        cleared = 0
        for cache_file in self.cache_dir.glob("*.json"):
            try:
                with open(cache_file, "r") as f:
                    cached = json.load(f)
                
                cached_time = datetime.fromisoformat(cached["timestamp"])
                if datetime.utcnow() - cached_time > self.ttl:
                    cache_file.unlink()
                    cleared += 1
            except Exception:
                continue
        
        if cleared:
            logger.info(f"Cleared {cleared} expired cache entries")
    
    def clear_all(self):
        """Clear entire cache."""
        for cache_file in self.cache_dir.glob("*.json"):
            cache_file.unlink()
        logger.info("Cleared all LLM cache")


# Global cache instance
cache = LLMCache()
