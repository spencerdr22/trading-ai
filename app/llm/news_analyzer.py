"""
news_analyzer.py — Qwen3-30B sentiment analysis via Ollama.

Key fix: analysis_inference timeout raised to 120s so the lock is held
for the full LLM call, not just the 5s acquisition window. Previously
only 1 headline per batch was processed because the context manager
exited after 5s while Qwen3 was still generating the response.

All emoji stripped for Windows cp1252 compatibility.
"""

import json
import re
import pandas as pd
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..monitor.logger import get_logger
from .prompts import SENTIMENT_ANALYSIS_PROMPT
from .cache import cache
from .gpu_scheduler import gpu_scheduler

logger = get_logger(__name__)

DEFAULT_MODEL  = "qwen3:30b-a3b-q4_K_M"
FALLBACK_MODEL = "qwen3:14b"

# Minimum successful analyses before aggregation is trusted.
# Below this threshold, return neutral to avoid single-headline bias.
MIN_ANALYZED_FOR_SIGNAL = 3


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
    """LLM-powered news sentiment analysis using Qwen3 via Ollama."""

    def __init__(
        self,
        model:       str  = DEFAULT_MODEL,
        use_cache:   bool = True,
        max_workers: int  = 4,   # reduced: Qwen3-30B is single-threaded on GPU
    ):
        self.model       = model
        self.use_cache   = use_cache
        self.max_workers = max_workers
        self._verify_model()

    def _verify_model(self):
        try:
            import ollama
            resp = ollama.list()
            available = (
                [m.model for m in resp.models]
                if hasattr(resp, "models")
                else [m.get("name", m.get("model", "")) for m in resp.get("models", [])]
            )
            logger.info("Ollama available models: %s", available)
            if self.model not in available:
                if FALLBACK_MODEL in available:
                    logger.warning("Model %s not found — using %s.", self.model, FALLBACK_MODEL)
                    self.model = FALLBACK_MODEL
                else:
                    logger.warning("Neither primary nor fallback model found — will attempt anyway.")
            else:
                logger.info("Using model: %s", self.model)
        except Exception as e:
            logger.warning("Could not verify Ollama models: %s — will attempt anyway.", e)

    def analyze_headline(
        self,
        headline:  str,
        symbol:    str              = "SPY",
        timestamp: Optional[datetime] = None,
    ) -> Optional[SentimentSignal]:
        """
        Analyze a single headline. Returns None on error or GPU unavailability.

        The GPU scheduler lock is held for the FULL Ollama call duration
        (up to 120 seconds) so the context manager does not exit mid-inference.
        """
        import ollama

        if timestamp is None:
            timestamp = datetime.now(timezone.utc)

        # Cache check — no GPU needed for cache hits
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
            # Hold the lock for the FULL inference call (not just acquisition).
            # timeout=120s gives Qwen3-30B plenty of room on the 4070 Super.
            with gpu_scheduler.analysis_inference(timeout=120.0) as available:
                if not available:
                    logger.debug("Skipping headline — GPU busy: %.50s", headline)
                    return None

                response = ollama.chat(
                    model    = self.model,
                    messages = [{"role": "user", "content": prompt}],
                )
                content = response["message"]["content"]
                logger.debug("Raw LLM response (200 chars): %.200s", content)

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
        """
        Analyze headlines sequentially (Qwen3-30B is GPU-bound; parallelism
        doesn't help and causes lock contention). Returns a DataFrame.
        """
        logger.info("Batch analyzing %d headlines for %s...", len(headlines), symbol)
        results = []

        # Sequential: one headline at a time so the GPU lock is never contested
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
        """
        Compute weighted aggregate sentiment from a batch DataFrame.

        If fewer than MIN_ANALYZED_FOR_SIGNAL headlines were successfully
        analyzed, returns neutral rather than letting a single headline
        dominate the signal. This prevents the "1-of-34 bullish" problem.
        """
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
                    "Only %d headline(s) analyzed — below minimum %d for signal. "
                    "Returning neutral to avoid single-headline bias.",
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

        avg   = float(df["weighted_score"].mean())
        total = len(df)
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
