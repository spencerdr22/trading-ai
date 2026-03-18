"""
news_analyzer.py — Qwen3-30B sentiment analysis via Ollama.

All emoji stripped for Windows cp1252 compatibility.
"""

import json
import re
import pandas as pd
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..monitor.logger import get_logger
from .prompts import SENTIMENT_ANALYSIS_PROMPT
from .cache import cache
from .gpu_scheduler import gpu_scheduler

logger = get_logger(__name__)

DEFAULT_MODEL  = "qwen3:30b-a3b-q4_K_M"
FALLBACK_MODEL = "qwen3:14b"


# ── Data structure ────────────────────────────────────────────────────────────

@dataclass
class SentimentSignal:
    headline:         str
    timestamp:        datetime
    sentiment:        str   # bullish | bearish | neutral
    confidence:       float
    relevance:        float
    urgency:          str   # low | medium | high
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


# ── Analyzer ─────────────────────────────────────────────────────────────────

class NewsFlowAnalyzer:
    """LLM-powered news sentiment analysis using Qwen3 via Ollama."""

    def __init__(
        self,
        model:       str  = DEFAULT_MODEL,
        use_cache:   bool = True,
        max_workers: int  = 8,
    ):
        self.model       = model
        self.use_cache   = use_cache
        self.max_workers = max_workers
        self._verify_model()

    def _verify_model(self):
        """Check Ollama availability and select best model."""
        try:
            import ollama
            resp = ollama.list()
            if hasattr(resp, "models"):
                available = [m.model for m in resp.models]
            elif isinstance(resp, dict):
                available = [m.get("name", m.get("model", "")) for m in resp.get("models", [])]
            else:
                available = [str(m) for m in resp]

            logger.info("Ollama available models: %s", available)

            if self.model not in available:
                if FALLBACK_MODEL in available:
                    logger.warning(
                        "Model %s not found — using fallback %s.",
                        self.model, FALLBACK_MODEL,
                    )
                    self.model = FALLBACK_MODEL
                else:
                    logger.warning(
                        "Neither %s nor %s found. Will attempt %s anyway.",
                        self.model, FALLBACK_MODEL, self.model,
                    )
            else:
                logger.info("Using model: %s", self.model)
        except Exception as e:
            logger.warning("Could not verify Ollama models: %s — will attempt anyway.", e)

    # ------------------------------------------------------------------
    def analyze_headline(
        self,
        headline:  str,
        symbol:    str              = "SPY",
        timestamp: Optional[datetime] = None,
    ) -> Optional[SentimentSignal]:
        """Analyze a single headline.  Returns None if GPU busy or on error."""
        import ollama

        if timestamp is None:
            timestamp = datetime.now(timezone.utc)

        # Cache check
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
            with gpu_scheduler.analysis_inference(timeout=5.0) as available:
                if not available:
                    logger.debug("Skipping headline (GPU busy): %.50s...", headline)
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

    # ------------------------------------------------------------------
    def _parse(
        self,
        response:  str,
        headline:  str,
        timestamp: datetime,
    ) -> SentimentSignal:
        """Parse JSON from LLM response into a SentimentSignal."""
        cleaned = response.strip()

        # Strip markdown fences
        if cleaned.startswith("```json"):
            cleaned = cleaned.split("```json", 1)[1].split("```", 1)[0]
        elif cleaned.startswith("```"):
            cleaned = cleaned.split("```", 1)[1].split("```", 1)[0]
        cleaned = cleaned.strip()

        # Extract first JSON object if surrounded by text
        if not cleaned.startswith("{"):
            m = re.search(r"\{.*?\}", cleaned, re.DOTALL)
            if m:
                cleaned = m.group(0)
            else:
                raise json.JSONDecodeError("No JSON object found", cleaned, 0)

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as e:
            logger.warning("JSON parse failed for '%.50s': %s", headline, e)
            return SentimentSignal(
                headline         = headline,
                timestamp        = timestamp,
                sentiment        = "neutral",
                confidence       = 0.0,
                relevance        = 0.0,
                urgency          = "low",
                category         = "parse_error",
                affected_symbols = [],
                key_entities     = [],
                summary          = "Parse error — defaulted to neutral.",
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

    # ------------------------------------------------------------------
    def analyze_batch(
        self,
        headlines: List[str],
        symbol:    str = "SPY",
    ) -> pd.DataFrame:
        """Analyze headlines in parallel.  Returns a DataFrame."""
        logger.info("Batch analyzing %d headlines for %s...", len(headlines), symbol)
        results = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {
                pool.submit(self.analyze_headline, h, symbol): h
                for h in headlines
            }
            for future in as_completed(futures):
                try:
                    sig = future.result()
                    if sig:
                        results.append(sig.to_dict())
                except Exception as e:
                    logger.error("Batch error for '%.30s': %s", futures[future], e)

        logger.info("Analyzed %d / %d headlines.", len(results), len(headlines))
        return pd.DataFrame(results) if results else pd.DataFrame()

    # ------------------------------------------------------------------
    def get_aggregated_sentiment(self, df: pd.DataFrame) -> dict:
        """Compute weighted aggregate sentiment from a batch DataFrame."""
        if df.empty:
            return {
                "overall_sentiment": "neutral",
                "sentiment_score":   0.0,
                "confidence":        0.0,
                "bullish_pct":       0.0,
                "bearish_pct":       0.0,
                "neutral_pct":       0.0,
            }

        df = df.copy()
        mapping = {"bullish": 1, "neutral": 0, "bearish": -1}
        df["weighted_score"] = (
            df["sentiment"].map(mapping).fillna(0)
            * df["confidence"]
            * df["relevance"]
        )

        avg   = float(df["weighted_score"].mean())
        total = len(df)
        counts= df["sentiment"].value_counts()

        return {
            "overall_sentiment": (
                "bullish" if avg > 0.15 else
                "bearish" if avg < -0.15 else
                "neutral"
            ),
            "sentiment_score": avg,
            "confidence":      float(df["confidence"].mean()),
            "bullish_pct":     float(counts.get("bullish", 0) / total),
            "bearish_pct":     float(counts.get("bearish", 0) / total),
            "neutral_pct":     float(counts.get("neutral", 0) / total),
        }
