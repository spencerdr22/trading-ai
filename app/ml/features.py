"""
features.py — Technical indicator feature engineering for MES / SPY.

Key design principles for profitability:
- Normalise all price-based features so the model sees patterns, not levels
- Include trend-following and mean-reversion signals
- Include regime/volatility context so the model knows when NOT to trade
- Keep lag count low (5) to avoid overfitting to price history
"""

import pandas as pd
import numpy as np


# ── Primitives ────────────────────────────────────────────────────────────────

def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    if len(series) < window:
        return pd.Series(50.0, index=series.index)
    delta = series.diff()
    up    = delta.clip(lower=0).ewm(alpha=1 / window, adjust=False).mean()
    down  = (-delta.clip(upper=0)).ewm(alpha=1 / window, adjust=False).mean()
    rs    = up / (down + 1e-8)
    return 100 - (100 / (1 + rs))


def atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    if len(df) < 2:
        return pd.Series(1.0, index=df.index)
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"]  - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(window=window, min_periods=1).mean()


def macd(series: pd.Series, short: int = 12, long: int = 26, signal_span: int = 9):
    short_ema   = ema(series, short)
    long_ema    = ema(series, long)
    macd_line   = short_ema - long_ema
    signal_line = macd_line.ewm(span=signal_span, adjust=False).mean()
    hist        = macd_line - signal_line
    return macd_line, signal_line, hist


def bollinger_bands(series: pd.Series, window: int = 20, n_std: float = 2.0):
    mid   = series.rolling(window, min_periods=1).mean()
    std   = series.rolling(window, min_periods=1).std().fillna(0)
    upper = mid + n_std * std
    lower = mid - n_std * std
    return upper, mid, lower


# ── Main feature builder ──────────────────────────────────────────────────────

def make_features(df: pd.DataFrame, window: int = 30) -> pd.DataFrame:
    """
    Build a normalised, regime-aware feature DataFrame.

    Every price-based feature is divided by ATR so the model sees
    pattern strength rather than absolute price levels. This makes
    features comparable across different price regimes and symbols.

    Args:
        df:     OHLCV DataFrame (timestamp, open, high, low, close, volume)
        window: rolling window for volume / volatility features

    Returns:
        DataFrame with OHLCV columns + engineered features, fully filled.
    """
    df = df.copy()
    close = df["close"]

    # ── ATR normalisation denominator ────────────────────────────────
    atr_val = atr(df, 14).replace(0, np.nan).ffill().fillna(1.0)

    # ── Returns (normalised by ATR so cross-asset comparable) ────────
    ret1  = close.pct_change().fillna(0)
    ret5  = close.pct_change(5).fillna(0)
    ret15 = close.pct_change(15).fillna(0)
    df["ret_1"]  = ret1
    df["ret_5"]  = ret5
    df["ret_15"] = ret15

    # ── Short lag features (only 5 — avoids overfitting) ─────────────
    for i in range(1, 6):
        df[f"lag_{i}"] = ret1.shift(i).fillna(0)

    # ── EMA trend signals (normalised) ───────────────────────────────
    ema9  = ema(close, 9)
    ema21 = ema(close, 21)
    ema50 = ema(close, 50)
    df["ema_cross_fast"] = (ema9  - ema21) / atr_val   # fast cross
    df["ema_cross_slow"] = (ema21 - ema50) / atr_val   # slow trend
    df["ema_slope"]      = ema9.diff(3) / atr_val       # momentum of fast EMA

    # ── RSI ───────────────────────────────────────────────────────────
    df["rsi"]         = rsi(close, 14) / 100.0          # normalise to [0,1]
    df["rsi_slope"]   = df["rsi"].diff(3).fillna(0)     # RSI momentum

    # ── MACD (normalised) ─────────────────────────────────────────────
    macd_line, macd_sig, macd_hist = macd(close)
    df["macd_hist_norm"] = macd_hist / atr_val

    # ── Bollinger Band position ───────────────────────────────────────
    bb_upper, bb_mid, bb_lower = bollinger_bands(close, 20)
    bb_width = (bb_upper - bb_lower).replace(0, np.nan).ffill().fillna(1.0)
    df["bb_pos"]   = (close - bb_lower) / bb_width      # 0=at lower, 1=at upper
    df["bb_width"] = bb_width / close                   # relative band width

    # ── ATR regime (how volatile is the current bar vs recent average) ─
    atr_slow = atr(df, 50).replace(0, np.nan).ffill().fillna(1.0)
    df["atr_ratio"] = atr_val / atr_slow                # >1 = expanding vol

    # ── Volume features ───────────────────────────────────────────────
    vol_ma = df["volume"].rolling(min(window, len(df)), min_periods=1).mean()
    df["vol_ratio"]  = df["volume"] / (vol_ma + 1e-9)   # relative volume spike
    df["vol_trend"]  = vol_ma.pct_change(5).fillna(0)   # volume trend

    # ── Bar structure ─────────────────────────────────────────────────
    bar_range = (df["high"] - df["low"]).replace(0, 1e-9)
    df["close_pct"]  = (close - df["low"]) / bar_range  # where close sits in bar
    df["bar_norm"]   = bar_range / atr_val               # bar size relative to ATR

    # ── Trend strength ────────────────────────────────────────────────
    # ADX-lite: ratio of directional move to total range over N bars
    high_diff = df["high"].diff(1).clip(lower=0)
    low_diff  = (-df["low"].diff(1)).clip(lower=0)
    df["dm_plus"]  = (high_diff / (atr_val + 1e-9)).rolling(14, min_periods=1).mean()
    df["dm_minus"] = (low_diff  / (atr_val + 1e-9)).rolling(14, min_periods=1).mean()
    df["trend_str"] = df["dm_plus"] - df["dm_minus"]     # + = uptrend, - = downtrend

    # ── Fill any remaining NaN ────────────────────────────────────────
    df = df.ffill().bfill().fillna(0)

    return df
