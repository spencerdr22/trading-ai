"""
Alpaca Paper Trading Integration — MES via SPY Proxy

Alpaca paper trading supports equities/ETFs, not CME futures directly.
Strategy: trade SPY shares as a 1:1 proxy for MES signals.
  - MES BUY  -> buy  N shares of SPY
  - MES SELL -> sell N shares of SPY
  - 1 share per signal — scale up once system is proven profitable

Key fix over previous version:
  In-memory position state (_position_side) prevents duplicate orders
  firing while Alpaca processes the previous fill. The API position
  check has ~5s lag; the local flag is instant.
"""

import os
import requests
from datetime import datetime, timezone
from typing import Dict, Optional, List
from dotenv import load_dotenv

from ..monitor.logger import get_logger

load_dotenv()
logger = get_logger(__name__)

ALPACA_API_KEY    = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
ALPACA_BASE_URL   = os.getenv("ALPACA_BASE_URL",  "https://paper-api.alpaca.markets")
ALPACA_DATA_URL   = os.getenv("ALPACA_DATA_URL",  "https://data.alpaca.markets")

MES_PROXY_SYMBOL = "SPY"
MES_PROXY_QTY    = 1


class AlpacaPaperTrading:
    """
    Synchronous Alpaca paper trading client with in-memory position tracking.

    The _position_side flag is the single source of truth for whether we
    are flat / long / short.  It is set immediately when an order is placed
    and cleared when positions are closed — no waiting for Alpaca to confirm.
    """

    def __init__(self):
        if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
            raise ValueError("ALPACA_API_KEY / ALPACA_SECRET_KEY missing from .env")

        self.headers = {
            "APCA-API-KEY-ID":     ALPACA_API_KEY,
            "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY,
            "Content-Type":        "application/json",
        }
        self.base_url = ALPACA_BASE_URL
        self.data_url = ALPACA_DATA_URL

        # In-memory position state — set on order placement, cleared on flatten
        # Values: None (flat), "long", "short"
        self._position_side: Optional[str] = None
        self._last_order_id: Optional[str] = None

        # Sync state with Alpaca on startup
        self._sync_position_state()
        logger.info("AlpacaPaperTrading initialized -> %s | position=%s",
                    self.base_url, self._position_side or "flat")

    # ------------------------------------------------------------------
    # STARTUP SYNC
    # ------------------------------------------------------------------

    def _sync_position_state(self):
        """
        On startup, read actual Alpaca position to initialise local state.
        This handles restarts mid-day where we may already have a position.
        """
        try:
            r = requests.get(
                f"{self.base_url}/v2/positions/{MES_PROXY_SYMBOL}",
                headers=self.headers, timeout=10,
            )
            if r.status_code == 200:
                qty = float(r.json().get("qty", 0))
                if qty > 0:
                    self._position_side = "long"
                elif qty < 0:
                    self._position_side = "short"
                else:
                    self._position_side = None
            else:
                self._position_side = None   # 404 = flat
        except Exception as e:
            logger.warning("Could not sync position state on startup: %s", e)
            self._position_side = None

    # ------------------------------------------------------------------
    # ACCOUNT
    # ------------------------------------------------------------------

    def get_account(self) -> Dict:
        r = requests.get(f"{self.base_url}/v2/account",
                         headers=self.headers, timeout=10)
        if r.status_code == 200:
            data = r.json()
            logger.info(
                "Portfolio: $%s  |  Buying Power: $%s",
                f"{float(data['portfolio_value']):,.2f}",
                f"{float(data['buying_power']):,.2f}",
            )
            return data
        logger.error("get_account failed: %s", r.text)
        return {}

    def is_market_open(self) -> bool:
        r = requests.get(f"{self.base_url}/v2/clock",
                         headers=self.headers, timeout=10)
        if r.status_code == 200:
            return r.json().get("is_open", False)
        return False

    # ------------------------------------------------------------------
    # POSITIONS
    # ------------------------------------------------------------------

    def get_positions(self) -> List[Dict]:
        r = requests.get(f"{self.base_url}/v2/positions",
                         headers=self.headers, timeout=10)
        if r.status_code == 200:
            return r.json()
        logger.error("get_positions failed: %s", r.text)
        return []

    def current_side(self) -> Optional[str]:
        """Return local position state — instant, no API call."""
        return self._position_side

    # ------------------------------------------------------------------
    # ORDERS
    # ------------------------------------------------------------------

    def place_order(
        self,
        side: str,
        qty:            int           = MES_PROXY_QTY,
        symbol:         str           = MES_PROXY_SYMBOL,
        order_type:     str           = "market",
        time_in_force:  str           = "day",
        limit_price:    Optional[float] = None,
    ) -> Optional[Dict]:
        """
        Place a paper order.

        Uses in-memory _position_side to block duplicate-direction orders
        instantly without waiting for Alpaca to confirm the previous fill.
        """
        # -- Duplicate direction guard (in-memory, instant) ----------------
        if side == "buy"  and self._position_side == "long":
            logger.debug("Already long %s — skipping BUY", symbol)
            return None
        if side == "sell" and self._position_side == "short":
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
            self._last_order_id = order.get("id")

            # Update local state immediately
            if side == "buy":
                self._position_side = "long"
            elif side == "sell":
                self._position_side = "short"

            logger.info(
                "Order placed: %s %s x%d | id=%s | local_state=%s",
                side.upper(), symbol, qty,
                self._last_order_id, self._position_side,
            )
            return order

        logger.error("Order failed (%s): %s", r.status_code, r.text)
        return None

    def cancel_all_orders(self) -> bool:
        r = requests.delete(f"{self.base_url}/v2/orders",
                            headers=self.headers, timeout=10)
        if r.status_code in (200, 207):
            logger.info("All orders cancelled")
            return True
        logger.error("cancel_all_orders failed: %s", r.text)
        return False

    def close_all_positions(self) -> bool:
        """Flatten all positions and reset local state."""
        r = requests.delete(f"{self.base_url}/v2/positions",
                            headers=self.headers, timeout=10)
        if r.status_code in (200, 207):
            self._position_side = None   # reset immediately
            logger.info("All positions closed | local_state=flat")
            return True
        logger.error("close_all_positions failed: %s", r.text)
        return False

    # ------------------------------------------------------------------
    # MES SIGNAL HANDLER
    # ------------------------------------------------------------------

    def execute_mes_signal(self, signal: Dict, bar: Dict) -> Optional[Dict]:
        """
        Translate a StrategyEngine signal into an Alpaca SPY order.
        Only fires when the signal represents a position change:
          flat -> long  : BUY signal
          flat -> short : SELL signal
          long -> flat  : SELL signal (close long)
          short -> flat : BUY signal (close short)
        """
        side_raw = signal.get("side", "HOLD").upper()
        if side_raw == "HOLD":
            return None

        close_px    = bar.get("close", 0)
        current     = self._position_side
        alpaca_side = "buy" if side_raw == "BUY" else "sell"

        logger.info(
            "MES signal: %s @ %.2f  current_pos=%s  -> Alpaca %s %s x%d",
            side_raw, close_px,
            current or "flat",
            alpaca_side.upper(), MES_PROXY_SYMBOL, MES_PROXY_QTY,
        )
        return self.place_order(side=alpaca_side)

    # ------------------------------------------------------------------
    # LIVE BAR FETCH
    # ------------------------------------------------------------------

    def get_latest_bar(self, symbol: str = MES_PROXY_SYMBOL) -> Optional[Dict]:
        url = f"{self.data_url}/v2/stocks/{symbol}/bars/latest"
        r = requests.get(url, headers=self.headers, timeout=10)
        if r.status_code == 200:
            bar = r.json().get("bar", {})
            return {
                "timestamp": bar.get("t", datetime.now(timezone.utc).isoformat()),
                "open":      bar.get("o", 0.0),
                "high":      bar.get("h", 0.0),
                "low":       bar.get("l", 0.0),
                "close":     bar.get("c", 0.0),
                "volume":    bar.get("v", 0),
            }
        logger.warning("get_latest_bar failed (%s): %s", r.status_code, r.text)
        return None


def get_alpaca_client() -> AlpacaPaperTrading:
    return AlpacaPaperTrading()
