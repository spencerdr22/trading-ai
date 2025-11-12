import pandas as pd

def resample_ticks_to_1m(ticks: pd.DataFrame) -> pd.DataFrame:
    """
    ticks must have 'timestamp' as index or column. Produces OHLCV 1m.
    """
    df = ticks.copy()
    if "timestamp" in df.columns:
        df = df.set_index("timestamp")
    o = df["price"].resample("1T").first()
    h = df["price"].resample("1T").max()
    l = df["price"].resample("1T").min()
    c = df["price"].resample("1T").last()
    v = df["size"].resample("1T").sum()
    res = pd.concat([o, h, l, c, v], axis=1)
    res.columns = ["open", "high", "low", "close", "volume"]
    return res.dropna()
