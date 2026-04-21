"""
app/llm/orchestrator_bridge.py
──────────────────────────────
Optional bridge from trading-ai to the AI Orchestrator running on
C:\Users\spenc\Documents\ai-orchestrator (FastAPI on port 8000).

When the orchestrator is running, sentiment analysis requests are routed
through its /trading/analyze endpoint instead of calling Ollama directly.
This gives the orchestrator visibility into trading requests and lets it
apply priority routing (trading > content > coding).

If the orchestrator is unreachable the bridge falls back transparently —
trading never blocks because the orchestrator is down.

Usage:
    from .orchestrator_bridge import orchestrator_sentiment, orchestrator_available

    if orchestrator_available():
        score, label = orchestrator_sentiment(headlines, symbol="MES")
"""

import os
import json
import requests
from typing import List, Tuple, Optional
from ..monitor.logger import get_logger

logger = get_logger(__name__)

ORCHESTRATOR_URL     = os.getenv("AI_ORCHESTRATOR_URL",      "http://localhost:8000")
ORCHESTRATOR_TIMEOUT = int(os.getenv("AI_ORCHESTRATOR_TIMEOUT", "10"))


def orchestrator_available() -> bool:
    """Ping the orchestrator /health endpoint. Returns True if reachable."""
    try:
        r = requests.get(f"{ORCHESTRATOR_URL}/health", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def orchestrator_analyze(
    prompt: str,
    context: str = "",
    max_tokens: int = 300,
) -> Optional[str]:
    """
    Send a prompt to the orchestrator /trading/analyze endpoint.
    Returns the raw response string or None on failure.
    Always uses the 8B model with trading-priority routing.
    """
    payload = {
        "prompt":     prompt,
        "context":    context,
        "max_tokens": max_tokens,
    }
    try:
        r = requests.post(
            f"{ORCHESTRATOR_URL}/trading/analyze",
            json=payload,
            timeout=ORCHESTRATOR_TIMEOUT,
        )
        if r.status_code == 200:
            data = r.json()
            result = data.get("result", "")
            if isinstance(result, dict):
                return result.get("response") or result.get("message", "")
            return str(result)
        logger.warning(
            "Orchestrator /trading/analyze returned %d: %s",
            r.status_code, r.text[:200],
        )
        return None
    except requests.exceptions.ConnectionError:
        logger.debug("Orchestrator not reachable at %s", ORCHESTRATOR_URL)
        return None
    except Exception as e:
        logger.debug("Orchestrator call failed: %s", e)
        return None


def orchestrator_sentiment(
    headlines: List[str],
    symbol: str = "MES",
) -> Tuple[float, str]:
    """
    Request sentiment analysis from the orchestrator.

    Returns (score, label):
        score  : float in [-1.0, 1.0]  (-1=bearish, 0=neutral, +1=bullish)
        label  : "bullish" | "bearish" | "neutral"

    Falls back to (0.0, "neutral") on any failure.
    """
    if not headlines:
        return 0.0, "neutral"

    sample         = headlines[:10]  # keep prompt short for fast 8B inference
    headlines_text = "\n".join(f"- {h}" for h in sample)

    prompt = (
        f"Analyze market sentiment for {symbol} from these headlines. "
        f'Reply ONLY with valid JSON: {{ "score": <float -1 to 1>, "label": "bullish"|"bearish"|"neutral" }}\n\n'
        f"{headlines_text}"
    )

    raw = orchestrator_analyze(prompt, context=f"Symbol: {symbol}", max_tokens=100)
    if not raw:
        return 0.0, "neutral"

    try:
        start = raw.find("{")
        end   = raw.rfind("}") + 1
        if start >= 0 and end > start:
            data  = json.loads(raw[start:end])
            score = float(data.get("score", 0.0))
            label = str(data.get("label", "neutral")).lower().strip("\"'  ")
            score = max(-1.0, min(1.0, score))
            if label not in ("bullish", "bearish", "neutral"):
                label = "bullish" if score > 0.05 else "bearish" if score < -0.05 else "neutral"
            logger.info(
                "Orchestrator sentiment [%s]: %s (%.3f)",
                symbol, label.upper(), score,
            )
            return score, label
    except Exception as e:
        logger.warning(
            "Could not parse orchestrator sentiment: %s | raw=%s",
            e, raw[:200],
        )

    return 0.0, "neutral"
