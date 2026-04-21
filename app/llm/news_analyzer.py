"""
news_analyzer.py — Qwen3 sentiment analysis via Ollama.

Routes through the AI Orchestrator (/trading/analyze) for LLM analysis.
Falls back to direct Ollama (port 11434) if orchestrator is not running.

Orchestrator endpoint: http://localhost:8000/trading/analyze
Direct Ollama fallback: http://localhost:11434
"""

import os
import json
import re
import requests
import pandas as pd
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import List, Optional

from ..monitor.logger import get_logger
from .prompts import SENTIMENT_ANALYSIS_PROMPT
from .cache import cache
from .gpu_scheduler import gpu_scheduler

logger = get_logger(__name__)

from . import DEFAULT_MODEL, FALLBACK_MODEL

MIN_ANALYZED_FOR_SIGNAL = 3

# ── Endpoint resolution ───────────────────────────────────────────────────────
# Try AI Orchestrator first (http://localhost:8000) — gives priority routing,
# model management, and metrics tracking.
# Fall back to direct Ollama (port 11434) if orchestrator is not running.

_ORCHESTRATOR_URL = os.getenv("ORCHESTRATOR_URL", "http://localhost:8000")
_DIRECT_URL       = os.getenv("OLLAMA_BASE_URL",   "http://localhost:11434")
_USE_ORCHESTRATOR = False  # resolved at import time below

def _check_orchestrator() -> bool:
    """Return True if the AI Orchestrator is reachable."""
    try:
        r = requests.get(f"{_ORCHESTRATOR_URL}/health", timeout=2)
        return r.status_code == 200
    except Exception:
        return False

_USE_ORCHESTRATOR = _check_orchestrator()

if _USE_ORCHESTRATOR:
    logger.info("Routing LLM calls through AI Orchestrator at %s", _ORCHESTRATOR_URL)
else:
    logger.warning(
        "AI Orchestrator not reachable — falling back to direct Ollama at %s",
        _DIRECT_URL,
    )

_TRADING_HEADERS = {"Content-Type": "application/json"}


@dataclass
class SentimentSignal:
    headline:         str
    timestamp:        datetime
    sentiment:        str
    confidence:       float
    relevance:        float
    urgency:          str
    category:         str
    affected_symbols: List[str]
    key_entities:     List[str]
    summary:          str

    def to_dict(self) -> dict:
        return {
            "headline":         self.headline,
            "timestamp":        self.timestamp.isoformat()
                                if isinstance(self.timestamp, datetime)
                                else self.timestamp,
            "sentiment":        self.sentiment,
            "confidence":       self.confidence,
            "relevance":        self.relevance,
            "urgency":          self.urgency,
            "category":         self.category,
            "affected_symbols": self.affected_symbols,
            "key_entities":     self.key_entities,
            "summary":          self.summary,
        }

    @property
    def numeric_sentiment(self) -> float:
        return {"bearish": -1.0, "neutral": 0.0, "bullish": 1.0}.get(
            self.sentiment, 0.0
        )

    @property
    def weighted_score(self) -> float:
        return self.numeric_sentiment * self.confidence * self.relevance


class NewsFlowAnalyzer:
    """Qwen3 sentiment analysis for financial headlines via Ollama."""

    def __init__(
        self,
        model:     str  = DEFAULT_MODEL,
        use_cache: bool = True,
    ):
        self.model     = model
        self.use_cache = use_cache
        self._verify_model()

    def _verify_model(self):
        """Check which models are available via direct Ollama (for fallback info only)."""
        try:
            r = requests.get(f"{_DIRECT_URL}/api/tags", timeout=5)
            r.raise_for_status()
            raw = r.json().get("models", [])
            available = [
                m.get("name", m.get("model", "")) for m in raw
            ]
            logger.info("Ollama models available: %s", available)
            if self.model not in available:
                if FALLBACK_MODEL in available:
                    logger.warning(
                        "Model %s not found — using %s. Pull: ollama pull %s",
                        self.model, FALLBACK_MODEL, self.model,
                    )
                    self.model = FALLBACK_MODEL
                elif available:
                    self.model = available[0]
                    logger.warning("Using first available model: %s", self.model)
                else:
                    logger.error("No Ollama models available.")
            else:
                logger.info("Using model: %s", self.model)
        except Exception as e:
            logger.warning("Could not verify Ollama models: %s", e)

    def _call_ollama(self, prompt: str) -> str:
        """
        Call LLM via AI Orchestrator if running, else direct Ollama.
        Orchestrator uses the 8B model with priority routing.
        Direct Ollama fallback uses the configured model.
        """
        if _USE_ORCHESTRATOR:
            # Route through orchestrator /trading/analyze
            payload = {"prompt": prompt, "max_tokens": 512}
            r = requests.post(
                f"{_ORCHESTRATOR_URL}/trading/analyze",
                headers=_TRADING_HEADERS,
                json=payload,
                timeout=60,
            )
            r.raise_for_status()
            return r.json()["result"].get("response", "")
        else:
            # Direct Ollama fallback
            payload = {
                "model":    self.model,
                "messages": [{"role": "user", "content": prompt}],
                "stream":   False,
            }
            r = requests.post(
                f"{_DIRECT_URL}/api/chat",
                headers=_TRADING_HEADERS,
                json=payload,
                timeout=60,
            )
            r.raise_for_status()
            return r.json()["message"]["content"]

    def analyze_headline(
        self,
        headline:  str,
        symbol:    str               = "SPY",
        timestamp: Optional[datetime] = None,
    ) -> Optional[SentimentSignal]:
        """Analyze a single headline. Returns None on error."""
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)

        # Cache check — no GPU needed
        if self.use_cache:
            cached = cache.get(f"{headline}:{symbol}", self.model)
            if cached:
                try:
                    return self._parse(cached, headline, timestamp)
                except Exception:
                    pass

        prompt = SENTIMENT_ANALYSIS_PROMPT.format(
            headline  = headline,
            symbol    = symbol,
            timestamp = timestamp.isoformat() if hasattr(timestamp, "isoformat")
                        else str(timestamp),
        )

        try:
            with gpu_scheduler.analysis_inference(timeout=30.0) as available:
                if not available:
                    logger.debug("Skipping headline — GPU busy: %.50s", headline)
                    return None

                content = self._call_ollama(prompt)
                logger.debug("LLM response (200 chars): %.200s", content)

                if self.use_cache:
                    cache.set(f"{headline}:{symbol}", self.model, content)

                return self._parse(content, headline, timestamp)

        except Exception as e:
            logger.error("LLM analysis failed for '%.50s': %s", headline, e)
            return None

    def _parse(self, response: str, headline: str, timestamp: datetime) -> SentimentSignal:
        cleaned = response.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned.split("```json", 1)[1].split("```", 1)[0]
        elif cleaned.startswith("```"):
            cleaned = cleaned.split("```", 1)[1].split("```", 1)[0]
        cleaned = cleaned.strip()

        if not cleaned.startswith("{"):
            m = re.search(r"\{.*?\}", cleaned, re.DOTALL)
            cleaned = m.group(0) if m else "{}"

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as e:
            logger.warning("JSON parse failed for '%.50s': %s", headline, e)
            return SentimentSignal(
                headline=headline, timestamp=timestamp,
                sentiment="neutral", confidence=0.0, relevance=0.0,
                urgency="low", category="parse_error",
                affected_symbols=[], key_entities=[],
                summary="Parse error — defaulted to neutral.",
            )

        return SentimentSignal(
            headline         = headline,
            timestamp        = timestamp,
            sentiment        = data.get("sentiment", "neutral"),
            confidence       = float(data.get("confidence", 0.5)),
            relevance        = float(data.get("relevance",  0.5)),
            urgency          = data.get("urgency",   "medium"),
            category         = data.get("category",  "general"),
            affected_symbols = data.get("affected_symbols", []),
            key_entities     = data.get("key_entities",     []),
            summary          = data.get("summary",   ""),
        )

    def analyze_batch(
        self,
        headlines: List[str],
        symbol:    str = "SPY",
    ) -> pd.DataFrame:
        """Analyze headlines sequentially with trading priority routing."""
        logger.info(
            "Batch analyzing %d headlines for %s (via %s)...",
            len(headlines), symbol,
            "orchestrator" if _USE_ORCHESTRATOR else "direct Ollama",
        )
        results = []
        for h in headlines:
            try:
                sig = self.analyze_headline(h, symbol)
                if sig:
                    results.append(sig.to_dict())
            except Exception as e:
                logger.error("Batch error for '%.30s': %s", h, e)

        logger.info("Analyzed %d / %d headlines.", len(results), len(headlines))
        return pd.DataFrame(results) if results else pd.DataFrame()

    def get_aggregated_sentiment(self, df: pd.DataFrame) -> dict:
        """Compute weighted aggregate. Returns neutral if too few analyzed."""
        NEUTRAL = {
            "overall_sentiment": "neutral",
            "sentiment_score":   0.0,
            "confidence":        0.0,
            "bullish_pct":       0.0,
            "bearish_pct":       0.0,
            "neutral_pct":       0.0,
        }

        if df.empty or len(df) < MIN_ANALYZED_FOR_SIGNAL:
            if not df.empty:
                logger.warning(
                    "Only %d headline(s) analyzed (min %d) — returning neutral.",
                    len(df), MIN_ANALYZED_FOR_SIGNAL,
                )
            return NEUTRAL

        df = df.copy()
        mapping = {"bullish": 1, "neutral": 0, "bearish": -1}
        df["weighted_score"] = (
            df["sentiment"].map(mapping).fillna(0)
            * df["confidence"]
            * df["relevance"]
        )

        avg    = float(df["weighted_score"].mean())
        total  = len(df)
        counts = df["sentiment"].value_counts()

        result = {
            "overall_sentiment": (
                "bullish" if avg >  0.15 else
                "bearish" if avg < -0.15 else
                "neutral"
            ),
            "sentiment_score": avg,
            "confidence":      float(df["confidence"].mean()),
            "bullish_pct":     float(counts.get("bullish", 0) / total),
            "bearish_pct":     float(counts.get("bearish", 0) / total),
            "neutral_pct":     float(counts.get("neutral", 0) / total),
        }

        logger.info(
            "Aggregated sentiment: %s (score=%.3f, n=%d, bull=%.0f%%, bear=%.0f%%)",
            result["overall_sentiment"].upper(),
            result["sentiment_score"],
            total,
            result["bullish_pct"] * 100,
            result["bearish_pct"] * 100,
        )
        return result
