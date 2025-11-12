# File: app/data/loader.py
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def load_sample(min_rows: int = 500, target_rows: int = 1440) -> pd.DataFrame:
    """
    Loads OHLCV data from disk if available.
    If not enough rows are present, generate synthetic data automatically.
    """
    data_path = os.path.join(os.getcwd(), "data", "sim_MES.csv")

    # === Try reading existing simulation ===
    if os.path.exists(data_path):
        try:
            df = pd.read_csv(data_path, parse_dates=["timestamp"])
            if not df.empty and all(c in df.columns for c in ["open", "high", "low", "close", "volume"]):
                if len(df) >= min_rows:
                    print(f"[INFO] Loaded existing data file: {data_path} ({len(df)} rows)")
                    return df
                else:
                    print(f"[WARN] Data file too short ({len(df)} rows). Generating synthetic data...")
            else:
                print(f"[WARN] File exists but missing required columns.")
        except Exception as e:
            print(f"[WARN] Could not load {data_path}: {e}")

    # === Generate synthetic dataset ===
    now = datetime.utcnow()
    timestamps = [now - timedelta(minutes=i) for i in range(target_rows)][::-1]
    price = np.cumsum(np.random.randn(target_rows)) + 4200
    df = pd.DataFrame({
        "timestamp": timestamps,
        "open": price + np.random.randn(target_rows) * 0.5,
        "high": price + np.random.rand(target_rows) * 1.0,
        "low": price - np.random.rand(target_rows) * 1.0,
        "close": price + np.random.randn(target_rows) * 0.3,
        "volume": np.random.randint(100, 1000, size=target_rows)
    })
    print(f"[INFO] Generated synthetic data ({target_rows} rows)")
    return df
