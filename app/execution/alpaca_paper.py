"""
Alpaca Paper Trading Integration — MES via SPY Proxy

Alpaca paper trading supports equities/ETFs, not CME futures directly.
Strategy: trade SPY shares as a 1:1 proxy for MES signals.
  - MES BUY  → buy  N shares of SPY
  - MES SELL → sell N shares of SPY (or short if margin enabled)
  - Position sizing based on MES tick value equivalent

Key endpoints:
  Paper trading: https://paper-api.alpaca.markets
  Market data:   https://data.alpaca.markets
"""

import os
import asyncio
import requests
from datetime import datetime, timezone
from typing import Dict, Optional, List
from dotenv import load_dotenv

from ..monitor.logger import get_logger

load_dotenv()
logger = get_logger(__name__)

ALPACA_API_KEY   = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
ALPACA_BASE_URL  = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
ALPACA_DATA_URL  = os.getenv("ALPACA_DATA_URL", "https://data.alpaca.markets")

# MES proxy: we trade SPY shares to simulate MES exposure.
# SPY ~= S&P 500 / 10.  MES point value = $5.  SPY ~$580 → 1 share ≈ $580.
# Using 1 SPY share per signal keeps risk minimal for paper testing.
MES_PROXY_SYMBOL  = "SPY"
MES_PROXY_QTY     = 1   # shares per signal; scale up once comfortable


class AlpacaPaperTrading:
    """
    Synchronous Alpaca paper trading client.

    Translates MES BUY/SELL signals into SPY equity orders on Alpaca paper.
    All HTTP calls use requests (sync) so it drops cleanly into forward_mode.
    """

    def __init__(self):
        if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
            raise ValueError(
                "ALPACA_API_KEY / ALPACA_SECRET_KEY missing from .env"
            )
        self.headers = {
            "APCA-API-KEY-ID":     ALPACA_API_KEY,
            "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY,
            "Content-Type":        "application/json",
        }
        self.base_url = ALPACA_BASE_URL
        self.data_url = ALPACA_DATA_URL
        self._open_order_id: Optional[str] = None  # track last order
        logger.info("✅ AlpacaPaperTrading initialized → %s", self.base_url)

    # ------------------------------------------------------------------
    # ACCOUNT
    # ------------------------------------------------------------------

    def get_account(self) -> Dict:
        """Return account dict with buying_power, portfolio_value, etc."""
        r = requests.get(f"{self.base_url}/v2/account", headers=self.headers, timeout=10)
        if r.status_code == 200:
            data = r.json()
            logger.info(
                "💰 Portfolio: $%s  |  Buying Power: $%s",
                f"{float(data['portfolio_value']):,.2f}",
                f"{float(data['buying_power']):,.2f}",
            )
            return data
        logger.error("get_account failed: %s", r.text)
        return {}

    def is_market_open(self) -> bool:
        """Check Alpaca clock — returns True if market is currently open."""
        r = requests.get(f"{self.base_url}/v2/clock", headers=self.headers, timeout=10)
        if r.status_code == 200:
            return r.json().get("is_open", False)
        return False

    # ------------------------------------------------------------------
    # POSITIONS
    # ------------------------------------------------------------------

    def get_positions(self) -> List[Dict]:
        """Return list of open positions."""
        r = requests.get(f"{self.base_url}/v2/positions", headers=self.headers, timeout=10)
        if r.status_code == 200:
            return r.json()
        logger.error("get_positions failed: %s", r.text)
        return []

    def get_position(self, symbol: str = MES_PROXY_SYMBOL) -> Optional[Dict]:
        """Return single position for symbol, or None if flat."""
        r = requests.get(
            f"{self.base_url}/v2/positions/{symbol}",
            headers=self.headers, timeout=10
        )
        if r.status_code == 200:
            return r.json()
        return None  # 404 = no position

    def current_side(self) -> Optional[str]:
        """Return 'long', 'short', or None based on current SPY position."""
        pos = self.get_position()
        if not pos:
            return None
        qty = float(pos.get("qty", 0))
        return "long" if qty > 0 else "short" if qty < 0 else None

    # ------------------------------------------------------------------
    # ORDERS
    # ------------------------------------------------------------------

    def place_order(
        self,
        side: str,                          # "buy" or "sell"
        qty: int = MES_PROXY_QTY,
        symbol: str = MES_PROXY_SYMBOL,
        order_type: str = "market",
        time_in_force: str = "day",
        limit_price: Optional[float] = None,
    ) -> Optional[Dict]:
        """
        Place a paper order.  Returns order dict or None on failure.
        Automatically skips duplicate-direction orders (already long/short).
        """
        current = self.current_side()
        if side == "buy"  and current == "long":
            logger.debug("Already long %s — skipping BUY", symbol)
            return None
        if side == "sell" and current == "short":
            logger.debug("Already short %s — skipping SELL", symbol)
            return None

        payload: Dict = {
            "symbol":        symbol,
            "qty":           str(qty),
            "side":          side,
            "type":          order_type,
            "time_in_force": time_in_force,
        }
        if order_type in ("limit", "stop_limit") and limit_price:
            payload["limit_price"] = str(round(limit_price, 2))

        r = requests.post(
            f"{self.base_url}/v2/orders",
            headers=self.headers,
            json=payload,
            timeout=10,
        )
        if r.status_code in (200, 201):
            order = r.json()
            self._open_order_id = order.get("id")
            logger.info(
                "📤 Order placed: %s %s x%s | id=%s",
                side.upper(), symbol, qty, self._open_order_id
            )
            return order
        logger.error("❌ Order failed (%s): %s", r.status_code, r.text)
        return None

    def cancel_all_orders(self) -> bool:
        """Cancel every open order."""
        r = requests.delete(f"{self.base_url}/v2/orders", headers=self.headers, timeout=10)
        if r.status_code in (200, 207):
            logger.info("🗑️  All orders cancelled")
            return True
        logger.error("cancel_all_orders failed: %s", r.text)
        return False

    def close_all_positions(self) -> bool:
        """Flatten all open positions (EOD cleanup)."""
        r = requests.delete(f"{self.base_url}/v2/positions", headers=self.headers, timeout=10)
        if r.status_code in (200, 207):
            logger.info("🏳️  All positions closed")
            return True
        logger.error("close_all_positions failed: %s", r.text)
        return False

    # ------------------------------------------------------------------
    # MES SIGNAL HANDLER  (called from forward_mode)
    # ------------------------------------------------------------------

    def execute_mes_signal(self, signal: Dict, bar: Dict) -> Optional[Dict]:
        """
        Translate a StrategyEngine signal dict into an Alpaca SPY order.

        Args:
            signal: {"side": "BUY"|"SELL"|"HOLD", "strength": float, ...}
            bar:    current price bar (used for logging)

        Returns:
            Alpaca order dict, or None for HOLD / blocked orders
        """
        side_raw = signal.get("side", "HOLD").upper()
        close_px = bar.get("close", 0)

        if side_raw == "HOLD":
            return None

        alpaca_side = "buy" if side_raw == "BUY" else "sell"

        logger.info(
            "🎯 MES signal: %s @ %.2f  →  Alpaca %s %s x%d",
            side_raw, close_px, alpaca_side.upper(), MES_PROXY_SYMBOL, MES_PROXY_QTY
        )
        return self.place_order(side=alpaca_side)

    # ------------------------------------------------------------------
    # LIVE BAR FETCH  (for real-time forward loop)
    # ------------------------------------------------------------------

    def get_latest_bar(self, symbol: str = MES_PROXY_SYMBOL) -> Optional[Dict]:
        """
        Fetch the most recent 1-minute bar for `symbol` from Alpaca market data.
        Returns a bar dict with open/high/low/close/volume/timestamp.
        """
        url = f"{self.data_url}/v2/stocks/{symbol}/bars/latest"
        r = requests.get(url, headers=self.headers, timeout=10)
        if r.status_code == 200:
            bar = r.json().get("bar", {})
            return {
                "timestamp": bar.get("t", datetime.now(timezone.utc).isoformat()),
                "open":   bar.get("o", 0.0),
                "high":   bar.get("h", 0.0),
                "low":    bar.get("l", 0.0),
                "close":  bar.get("c", 0.0),
                "volume": bar.get("v", 0),
            }
        logger.warning("get_latest_bar failed (%s): %s", r.status_code, r.text)
        return None


def get_alpaca_client() -> AlpacaPaperTrading:
    """Convenience factory — returns a ready AlpacaPaperTrading instance."""
    return AlpacaPaperTrading()
