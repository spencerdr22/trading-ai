# File: app/data/loader.py
"""
Data loader for Trading-AI.

Priority order for training data:
  1. Real SPY bar data freshly fetched from Alpaca (always on startup)
  2. Cached real_SPY.csv if under 30 minutes old (intraday retrain)
  3. sim_MES.csv if real fetch fails and sim has enough rows
  4. Synthetic random-walk data — absolute last resort

Cache age reduced from 2 hours -> 30 minutes so intraday retrains
always use recent market data.
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

SIM_PATH        = os.path.join(os.getcwd(), "data", "sim_MES.csv")
REAL_PATH       = os.path.join(os.getcwd(), "data", "real_SPY.csv")
MIN_ROWS        = 500
TARGET_ROWS     = 1440          # ~1 trading day of 1-min bars
CACHE_MAX_AGE_S = 30 * 60      # 30 minutes — stale after this


def fetch_real_bars(symbol: str = "SPY", days: int = 30) -> pd.DataFrame:
    """
    Pull recent 1-minute OHLCV bars from Alpaca market data API.
    Returns empty DataFrame if credentials missing or request fails.
    """
    if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
        print("[WARN] Alpaca API keys not set — cannot fetch real bars.")
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
        "adjustment": "raw",
    }

    all_bars = []
    next_token = None

    try:
        while True:
            if next_token:
                params["page_token"] = next_token
            r = requests.get(
                f"{ALPACA_DATA_URL}/v2/stocks/{symbol}/bars",
                headers=headers,
                params=params,
                timeout=30,
            )
            r.raise_for_status()
            data = r.json()
            bars = data.get("bars", [])
            all_bars.extend(bars)
            next_token = data.get("next_page_token")
            if not next_token or not bars:
                break

        if not all_bars:
            print(f"[WARN] Alpaca returned 0 bars for {symbol}.")
            return pd.DataFrame()

        df = pd.DataFrame(all_bars)
        df = df.rename(columns={
            "t": "timestamp",
            "o": "open",
            "h": "high",
            "l": "low",
            "c": "close",
            "v": "volume",
        })[["timestamp", "open", "high", "low", "close", "volume"]]

        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_localize(None)

        os.makedirs(os.path.dirname(REAL_PATH), exist_ok=True)
        df.to_csv(REAL_PATH, index=False)
        print(f"[INFO] Fetched {len(df)} real {symbol} bars from Alpaca ({days} days)")
        return df

    except Exception as e:
        print(f"[WARN] Alpaca bar fetch failed: {e}")
        return pd.DataFrame()


def load_real_bars(force_refresh: bool = False) -> pd.DataFrame:
    """
    Return real SPY bars.

    If force_refresh=True (used at startup), always re-fetch from Alpaca.
    Otherwise use cache if under CACHE_MAX_AGE_S old.
    """
    if not force_refresh and os.path.exists(REAL_PATH):
        age_seconds = (
            datetime.now(timezone.utc).timestamp()
            - os.path.getmtime(REAL_PATH)
        )
        if age_seconds < CACHE_MAX_AGE_S:
            try:
                df = pd.read_csv(REAL_PATH, parse_dates=["timestamp"])
                if len(df) >= MIN_ROWS:
                    print(
                        f"[INFO] Using cached real SPY bars: {len(df)} rows "
                        f"({age_seconds/60:.0f} min old)"
                    )
                    return df
            except Exception:
                pass

    return fetch_real_bars()


def load_sample(
    min_rows: int = MIN_ROWS,
    target_rows: int = TARGET_ROWS,
    force_refresh: bool = True,   # always fetch fresh data at startup
) -> pd.DataFrame:
    """
    Return OHLCV data for model training.

    Priority:
      1. Real Alpaca SPY bars (always attempted first)
      2. sim_MES.csv (acceptable — synthetic but stable)
      3. Fresh synthetic data (last resort — uses current price level)
    """
    # ── Priority 1: Real Alpaca bars ──────────────────────────────────
    real_df = load_real_bars(force_refresh=force_refresh)
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
                print(
                    f"[WARN] sim_MES.csv too short or missing columns "
                    f"({len(df)} rows) — generating synthetic data"
                )
        except Exception as e:
            print(f"[WARN] Could not read sim_MES.csv: {e}")

    # ── Priority 3: Synthetic fallback ────────────────────────────────
    # Use a starting price near current SPY level (~$560) so features
    # are at least in the right order of magnitude
    print("[WARN] Using synthetic fallback data — real bar fetch failed.")
    now = datetime.utcnow()
    timestamps = [now - timedelta(minutes=i) for i in range(target_rows)][::-1]
    price = np.cumsum(np.random.randn(target_rows) * 0.3) + 560.0
    df = pd.DataFrame({
        "timestamp": timestamps,
        "open":      price + np.random.randn(target_rows) * 0.2,
        "high":      price + np.abs(np.random.randn(target_rows)) * 0.5,
        "low":       price - np.abs(np.random.randn(target_rows)) * 0.5,
        "close":     price + np.random.randn(target_rows) * 0.2,
        "volume":    np.random.randint(500, 5000, size=target_rows),
    })
    print(f"[INFO] Generated synthetic fallback data ({target_rows} rows @ ~$560)")
    return df
