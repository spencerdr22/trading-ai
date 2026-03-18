# File: app/data/live_feed.py

import os
import requests
import pandas as pd
from datetime import datetime, timedelta

DATA_DIR = os.path.join(os.getcwd(), "data")
os.makedirs(DATA_DIR, exist_ok=True)

DATA_FILE = os.path.join(DATA_DIR, "sp500_mini.csv")

API_KEY = os.getenv("ALPACA_API_KEY")
API_SECRET = os.getenv("ALPACA_SECRET_KEY")  # fixed: was ALPACA_API_SECRET
BASE_URL = "https://data.alpaca.markets/v2"

# ---------------------------
# Fetch Historical Data
# ---------------------------
def fetch_week_snapshot(symbol="SPY", timeframe="1Min", days=5):  # SPY proxy for MES via Alpaca
    """
    Fetch last `days` worth of historical bars (default 5 days, 1m bars).
    """
    end = datetime.utcnow()
    start = end - timedelta(days=days)

    url = f"{BASE_URL}/stocks/{symbol}/bars"
    headers = {
        "APCA-API-KEY-ID": API_KEY,
        "APCA-API-SECRET-KEY": API_SECRET,
    }
    params = {
        "timeframe": timeframe,
        "start": start.isoformat() + "Z",
        "end": end.isoformat() + "Z",
        "limit": 10000,
    }

    resp = requests.get(url, headers=headers, params=params)
    resp.raise_for_status()
    data = resp.json().get("bars", [])

    if not data:
        print("[WARN] No data returned from API.")
        return pd.DataFrame()

    df = pd.DataFrame(data)
    df.to_csv(DATA_FILE, index=False)
    print(f"[INFO] Saved snapshot to {DATA_FILE}")
    return df

# ---------------------------
# Load / Update Local Data
# ---------------------------
def load_data():
    if os.path.exists(DATA_FILE):
        return pd.read_csv(DATA_FILE)
    return pd.DataFrame()

def append_new_bar(bar):
    df = load_data()
    df = pd.concat([df, pd.DataFrame([bar])], ignore_index=True)
    df.to_csv(DATA_FILE, index=False)
    print("[INFO] New bar appended to local dataset.")

def get_last_n_bars(n=100):
    df = load_data()
    if df.empty:
        return pd.DataFrame()
    return df.tail(n)


class LiveFeed:
    """
    Stub live feed class.
    Streams bars from Alpaca data API for the given symbol.
    Requires a tradovate_client (or any client with place_order) for execution.
    """

    def __init__(self, symbol: str = "SPY", tradovate_client=None):
        self.symbol   = symbol
        self.client   = tradovate_client

    async def stream(self):
        """Async bar stream — placeholder for WebSocket integration."""
        import asyncio
        print(f"[LiveFeed] Streaming {self.symbol} (stub — no live WebSocket yet).")
        while True:
            bar = fetch_week_snapshot(symbol=self.symbol, days=1)
            if not bar.empty and self.client:
                latest = bar.iloc[-1].to_dict()
                print(f"[LiveFeed] Latest bar: {latest}")
            await asyncio.sleep(60)
