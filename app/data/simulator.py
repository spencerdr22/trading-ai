import numpy as np
import pandas as pd
import datetime as dt
import pytz
from typing import Generator, Optional
from ..config import load_config

cfg = load_config()

def generate_walk_prices(
    start_price: float = 1000.0,
    n: int = 1440,
    volatility: float = None,
    seed: Optional[int] = None,
    start_time: Optional[dt.datetime] = None,
) -> pd.DataFrame:
    """
    Generate 1-minute OHLCV bars using a simple geometric random walk.
    Returns a DataFrame with timezone-aware UTC timestamps.
    """
    np.random.seed(seed or cfg["simulator"]["seed"])
    vol = volatility if volatility is not None else cfg["simulator"]["volatility"]
    if start_time is None:
        start_time = dt.datetime(2023, 1, 1, 9, 30, tzinfo=pytz.UTC)
    # generate log returns
    returns = np.random.normal(loc=0, scale=vol, size=n)
    prices = start_price * np.exp(np.cumsum(returns))
    times = [start_time + dt.timedelta(minutes=i) for i in range(n)]
    # build OHLCV by adding intrabar noise
    high = prices * (1 + np.abs(np.random.normal(0, vol*2, size=n)))
    low = prices * (1 - np.abs(np.random.normal(0, vol*2, size=n)))
    open_p = np.concatenate([[start_price], prices[:-1]])
    volume = np.random.poisson(lam=120, size=n) + np.random.randint(1, 50, size=n)
    df = pd.DataFrame({
        "timestamp": times,
        "open": open_p,
        "high": np.maximum(open_p, high),
        "low": np.minimum(open_p, low),
        "close": prices,
        "volume": volume
    })
    return df

def stream_bars(
    symbol: str = "MES",
    minutes: int = 1440,
    fast: bool = False,
    start_price: float = 1000.0,
    volatility: float = None,
    seed: Optional[int] = None,
) -> Generator[pd.Series, None, None]:
    """
    Yields bars one by one. If fast=True yields quickly (no sleep).
    """
    df = generate_walk_prices(start_price=start_price, n=minutes, volatility=volatility, seed=seed)
    for _, row in df.iterrows():
        yield row
