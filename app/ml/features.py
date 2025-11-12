import pandas as pd
import numpy as np

def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    if len(series) < window:
        return pd.Series(np.nan, index=series.index)
    delta = series.diff()
    up = delta.clip(lower=0).ewm(alpha=1/window).mean()
    down = -delta.clip(upper=0).ewm(alpha=1/window).mean()
    rs = up / (down + 1e-8)
    return 100 - (100 / (1 + rs))

def atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    if len(df) < window:
        return pd.Series(np.nan, index=df.index)
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(window=window, min_periods=1).mean()

def macd(series: pd.Series, short: int = 12, long: int = 26, signal: int = 9):
    short_ema = ema(series, short)
    long_ema = ema(series, long)
    macd_line = short_ema - long_ema
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def make_features(df: pd.DataFrame, window: int = 30) -> pd.DataFrame:
    df = df.copy()

    # Basic returns and lags
    df["return_1"] = df["close"].pct_change().fillna(0)
    for i in range(1, min(window, len(df))):
        df[f"lag_{i}"] = df["close"].shift(i)

    # Core indicators
    df["ema_short"] = ema(df["close"], 12)
    df["ema_long"] = ema(df["close"], 26)
    df["rsi"] = rsi(df["close"], 14)
    df["atr"] = atr(df, 14)
    df["macd"], df["macd_signal"], df["macd_hist"] = macd(df["close"])
    df["vol_mean_30"] = df["volume"].rolling(min(30, len(df)), min_periods=1).mean()

    # Instead of dropping everything, just fill missing indicators for small sets
    df = df.fillna(method="bfill").fillna(method="ffill").fillna(0)

    return df
