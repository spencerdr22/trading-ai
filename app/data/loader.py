# File: app/data/loader.py
"""
Data loader for Trading-AI.

Priority order for training data:
  1. Real SPY bar data from Alpaca (last N days) — best quality
  2. Existing sim_MES.csv if it has enough rows — acceptable
  3. Synthetic random-walk data — last resort

The RF model trains on whichever source is available at the top
of this priority chain.
"""

import os
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv

load_dotenv()

ALPACA_API_KEY    = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
ALPACA_DATA_URL   = os.getenv("ALPACA_DATA_URL", "https://data.alpaca.markets")

SIM_PATH     = os.path.join(os.getcwd(), "data", "sim_MES.csv")
REAL_PATH    = os.path.join(os.getcwd(), "data", "real_SPY.csv")
MIN_ROWS     = 500
TARGET_ROWS  = 1440   # ~1 trading day of 1-min bars


def fetch_real_bars(symbol: str = "SPY", days: int = 5) -> pd.DataFrame:
    """
    Pull recent 1-minute OHLCV bars from Alpaca market data API.
    Returns empty DataFrame if credentials missing or request fails.
    """
    if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
        return pd.DataFrame()

    end   = datetime.now(timezone.utc)
    start = end - timedelta(days=days)

    headers = {
        "APCA-API-KEY-ID":     ALPACA_API_KEY,
        "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY,
    }
    params = {
        "timeframe": "1Min",
        "start":     start.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "end":       end.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "limit":     10000,
        "feed":      "iex",
    }

    try:
        r = requests.get(
            f"{ALPACA_DATA_URL}/v2/stocks/{symbol}/bars",
            headers=headers,
            params=params,
            timeout=15,
        )
        r.raise_for_status()
        bars = r.json().get("bars", [])
        if not bars:
            return pd.DataFrame()

        df = pd.DataFrame(bars)
        df = df.rename(columns={
            "t": "timestamp",
            "o": "open",
            "h": "high",
            "l": "low",
            "c": "close",
            "v": "volume",
        })[["timestamp", "open", "high", "low", "close", "volume"]]

        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_localize(None)

        # Save for reuse
        os.makedirs(os.path.dirname(REAL_PATH), exist_ok=True)
        df.to_csv(REAL_PATH, index=False)
        print(f"[INFO] Fetched {len(df)} real SPY bars from Alpaca ({days} days)")
        return df

    except Exception as e:
        print(f"[WARN] Alpaca bar fetch failed: {e}")
        return pd.DataFrame()


def load_real_bars() -> pd.DataFrame:
    """
    Load cached real SPY bars from disk if recent enough (under 2 hours old).
    Otherwise re-fetch from Alpaca.
    """
    if os.path.exists(REAL_PATH):
        age_seconds = (
            datetime.now(timezone.utc).timestamp()
            - os.path.getmtime(REAL_PATH)
        )
        if age_seconds < 7200:   # under 2 hours — use cache
            try:
                df = pd.read_csv(REAL_PATH, parse_dates=["timestamp"])
                if len(df) >= MIN_ROWS:
                    print(f"[INFO] Using cached real SPY bars: {len(df)} rows "
                          f"({age_seconds/60:.0f} min old)")
                    return df
            except Exception:
                pass

    return fetch_real_bars()


def load_sample(min_rows: int = MIN_ROWS, target_rows: int = TARGET_ROWS) -> pd.DataFrame:
    """
    Return OHLCV data for model training.

    Priority:
      1. Real Alpaca SPY bars (best — real price dynamics)
      2. sim_MES.csv (acceptable — synthetic but stable)
      3. Fresh synthetic data (last resort)
    """
    # ── Priority 1: Real Alpaca bars ──────────────────────────────────
    real_df = load_real_bars()
    if len(real_df) >= min_rows:
        return real_df

    # ── Priority 2: Existing sim file ─────────────────────────────────
    if os.path.exists(SIM_PATH):
        try:
            df = pd.read_csv(SIM_PATH, parse_dates=["timestamp"])
            if (not df.empty
                    and all(c in df.columns
                            for c in ["open", "high", "low", "close", "volume"])
                    and len(df) >= min_rows):
                print(f"[INFO] Using sim_MES.csv: {len(df)} rows")
                return df
            else:
                print(f"[WARN] sim_MES.csv too short or missing columns "
                      f"({len(df)} rows) — generating synthetic data")
        except Exception as e:
            print(f"[WARN] Could not read sim_MES.csv: {e}")

    # ── Priority 3: Synthetic fallback ────────────────────────────────
    now = datetime.utcnow()
    timestamps = [now - timedelta(minutes=i) for i in range(target_rows)][::-1]
    price = np.cumsum(np.random.randn(target_rows) * 0.5) + 5750
    df = pd.DataFrame({
        "timestamp": timestamps,
        "open":      price + np.random.randn(target_rows) * 0.5,
        "high":      price + np.random.rand(target_rows) * 1.0,
        "low":       price - np.random.rand(target_rows) * 1.0,
        "close":     price + np.random.randn(target_rows) * 0.3,
        "volume":    np.random.randint(100, 1000, size=target_rows),
    })
    print(f"[INFO] Generated synthetic fallback data ({target_rows} rows)")
    return df
