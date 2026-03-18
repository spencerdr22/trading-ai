"""
Tradovate Live Trading Client — MES Futures

Full REST + WebSocket integration for Tradovate.
Handles OAuth token refresh, order placement, position tracking,
and real-time bar streaming for MES.

Credentials required in .env:
    TRADOVATE_CLIENT_ID
    TRADOVATE_CLIENT_SECRET
    TRADOVATE_ACCESS_TOKEN   (optional — auto-fetched if blank)
    TRADOVATE_ACCOUNT_ID     (numeric account ID from Tradovate)
    TRADOVATE_ENV            live | demo  (default: demo)
"""

import os
import json
import time
import threading
import requests
from datetime import datetime, timezone
from typing import Dict, Optional, List
from dotenv import load_dotenv

from ..monitor.logger import get_logger

load_dotenv()
logger = get_logger(__name__)

# ── Endpoints ────────────────────────────────────────────────────────────────
_ENV = os.getenv("TRADOVATE_ENV", "demo").lower()
BASE_URL  = "https://live.tradovateapi.com/v1"  if _ENV == "live" else "https://demo.tradovateapi.com/v1"
WS_URL    = "wss://live.tradovateapi.com/v1/websocket" if _ENV == "live" else "wss://demo.tradovateapi.com/v1/websocket"

# MES contract details
MES_SYMBOL      = "MESH5"   # update to current front-month each quarter
MES_TICK_SIZE   = 0.25
MES_TICK_VALUE  = 1.25      # $ per tick
MES_POINT_VALUE = 5.0       # $ per full point


class TradovateAPI:
    """
    Full Tradovate REST client for live/demo MES trading.

    Features:
    - OAuth2 token auto-refresh
    - Place / cancel market and limit orders
    - Real-time position tracking
    - Account PnL reporting
    - Execute MES signals from StrategyEngine
    """

    def __init__(self):
        self.client_id     = os.getenv("TRADOVATE_CLIENT_ID", "")
        self.client_secret = os.getenv("TRADOVATE_CLIENT_SECRET", "")
        self.account_id    = os.getenv("TRADOVATE_ACCOUNT_ID", "")
        self._access_token = os.getenv("TRADOVATE_ACCESS_TOKEN", "")
        self._token_expiry = 0.0
        self._lock         = threading.Lock()

        self.ready = bool(self.client_id and self.client_secret)
        if not self.ready:
            logger.warning(
                "Tradovate credentials not set — live mode disabled. "
                "Add TRADOVATE_CLIENT_ID and TRADOVATE_CLIENT_SECRET to .env"
            )
            return

        # Validate / refresh token on init
        try:
            self._ensure_token()
            logger.info("TradovateAPI initialized [%s] account=%s", _ENV.upper(), self.account_id)
        except Exception as e:
            logger.error("Tradovate token init failed: %s", e)
            self.ready = False

    # ------------------------------------------------------------------
    # AUTH
    # ------------------------------------------------------------------

    def _ensure_token(self):
        """Refresh access token if expired or missing."""
        with self._lock:
            if self._access_token and time.time() < self._token_expiry - 60:
                return  # still valid

            logger.info("Tradovate: requesting new access token...")
            payload = {
                "name":       self.client_id,
                "password":   self.client_secret,
                "appId":      "TradingAI",
                "appVersion": "1.0",
                "cid":        int(self.client_id) if self.client_id.isdigit() else 0,
                "sec":        self.client_secret,
            }
            r = requests.post(f"{BASE_URL}/auth/accesstokenrequest", json=payload, timeout=15)
            r.raise_for_status()
            data = r.json()

            token = data.get("accessToken") or data.get("access_token")
            if not token:
                raise ValueError(f"No token in response: {data}")

            self._access_token = token
            # Tradovate tokens expire in 80 minutes
            self._token_expiry = time.time() + 80 * 60
            logger.info("Tradovate: token refreshed, expires in 80 min")

    def _headers(self) -> Dict[str, str]:
        self._ensure_token()
        return {
            "Authorization": f"Bearer {self._access_token}",
            "Content-Type":  "application/json",
        }

    def _get(self, path: str, params: dict = None) -> dict:
        r = requests.get(f"{BASE_URL}{path}", headers=self._headers(),
                         params=params, timeout=10)
        r.raise_for_status()
        return r.json()

    def _post(self, path: str, payload: dict) -> dict:
        r = requests.post(f"{BASE_URL}{path}", headers=self._headers(),
                          json=payload, timeout=10)
        r.raise_for_status()
        return r.json()

    # ------------------------------------------------------------------
    # ACCOUNT
    # ------------------------------------------------------------------

    def get_account(self) -> Dict:
        """Return account info dict."""
        if not self.ready:
            return {}
        try:
            accts = self._get("/account/list")
            if accts:
                acct = accts[0]
                logger.info(
                    "Tradovate account: id=%s  balance=$%.2f",
                    acct.get("id"), acct.get("cashBalance", 0)
                )
                return acct
        except Exception as e:
            logger.error("get_account failed: %s", e)
        return {}

    def get_cash_balance(self) -> float:
        """Return current account cash balance."""
        acct = self.get_account()
        return float(acct.get("cashBalance", 0))

    # ------------------------------------------------------------------
    # POSITIONS
    # ------------------------------------------------------------------

    def get_positions(self) -> List[Dict]:
        """Return all open positions."""
        if not self.ready:
            return []
        try:
            return self._get("/position/list") or []
        except Exception as e:
            logger.error("get_positions failed: %s", e)
            return []

    def get_mes_position(self) -> Optional[Dict]:
        """Return current MES position, or None if flat."""
        for pos in self.get_positions():
            if "MES" in str(pos.get("contractId", "")):
                return pos
        return None

    def current_side(self) -> Optional[str]:
        """Return 'long', 'short', or None."""
        pos = self.get_mes_position()
        if not pos:
            return None
        net = int(pos.get("netPos", 0))
        return "long" if net > 0 else "short" if net < 0 else None

    # ------------------------------------------------------------------
    # ORDERS
    # ------------------------------------------------------------------

    def _get_contract_id(self, symbol: str = MES_SYMBOL) -> Optional[int]:
        """Look up numeric contract ID for a symbol."""
        try:
            contracts = self._get("/contract/find", {"name": symbol})
            if isinstance(contracts, list) and contracts:
                return contracts[0].get("id")
            if isinstance(contracts, dict):
                return contracts.get("id")
        except Exception as e:
            logger.error("Contract lookup failed for %s: %s", symbol, e)
        return None

    def place_order(
        self,
        side: str,                      # "Buy" or "Sell"
        qty: int = 1,
        order_type: str = "Market",     # Market | Limit | Stop | StopLimit
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        symbol: str = MES_SYMBOL,
    ) -> Optional[Dict]:
        """
        Place an order on Tradovate.

        Args:
            side:        "Buy" or "Sell"
            qty:         Number of contracts
            order_type:  Market | Limit | Stop | StopLimit
            limit_price: Required for Limit / StopLimit orders
            stop_price:  Required for Stop / StopLimit orders
            symbol:      Contract symbol (default MES front month)

        Returns:
            Order response dict or None on failure
        """
        if not self.ready:
            logger.error("Tradovate not ready — order blocked")
            return None

        contract_id = self._get_contract_id(symbol)
        if not contract_id:
            logger.error("Cannot place order — contract ID not found for %s", symbol)
            return None

        payload = {
            "accountSpec":  self.account_id,
            "accountId":    int(self.account_id) if self.account_id.isdigit() else 0,
            "action":       side,           # "Buy" or "Sell"
            "symbol":       symbol,
            "orderQty":     qty,
            "orderType":    order_type,
            "isAutomated":  True,
        }
        if limit_price and order_type in ("Limit", "StopLimit"):
            payload["price"] = limit_price
        if stop_price and order_type in ("Stop", "StopLimit"):
            payload["stopPrice"] = stop_price

        try:
            result = self._post("/order/placeorder", payload)
            order_id = result.get("orderId") or result.get("id")
            logger.info(
                "Tradovate order placed: %s %d %s | id=%s",
                side, qty, symbol, order_id
            )
            return result
        except Exception as e:
            logger.error("Tradovate place_order failed: %s", e)
            return None

    def cancel_order(self, order_id: int) -> bool:
        """Cancel a specific order by ID."""
        if not self.ready:
            return False
        try:
            self._post("/order/cancelorder", {"orderId": order_id})
            logger.info("Tradovate order %d cancelled", order_id)
            return True
        except Exception as e:
            logger.error("cancel_order failed: %s", e)
            return False

    def cancel_all_orders(self) -> bool:
        """Cancel all open orders for the account."""
        if not self.ready:
            return False
        try:
            open_orders = self._get("/order/list") or []
            cancelled = 0
            for o in open_orders:
                if o.get("ordStatus") in ("Working", "Accepted", "PendingNew"):
                    if self.cancel_order(o["id"]):
                        cancelled += 1
            logger.info("Tradovate: cancelled %d orders", cancelled)
            return True
        except Exception as e:
            logger.error("cancel_all_orders failed: %s", e)
            return False

    def close_all_positions(self) -> bool:
        """Flatten all open positions with market orders."""
        if not self.ready:
            return False
        try:
            for pos in self.get_positions():
                net = int(pos.get("netPos", 0))
                if net == 0:
                    continue
                close_side = "Sell" if net > 0 else "Buy"
                self.place_order(side=close_side, qty=abs(net))
            logger.info("Tradovate: all positions closed")
            return True
        except Exception as e:
            logger.error("close_all_positions failed: %s", e)
            return False

    # ------------------------------------------------------------------
    # MES SIGNAL HANDLER
    # ------------------------------------------------------------------

    def execute_mes_signal(self, signal: Dict, bar: Dict) -> Optional[Dict]:
        """
        Translate a StrategyEngine signal into a live Tradovate MES order.

        Args:
            signal: {"side": "BUY"|"SELL"|"HOLD", "strength": float}
            bar:    current price bar (for logging)

        Returns:
            Order dict or None for HOLD / blocked
        """
        if not self.ready:
            return None

        side_raw = signal.get("side", "HOLD").upper()
        if side_raw == "HOLD":
            return None

        current = self.current_side()
        if side_raw == "BUY" and current == "long":
            logger.debug("Already long MES — skipping BUY")
            return None
        if side_raw == "SELL" and current == "short":
            logger.debug("Already short MES — skipping SELL")
            return None

        tradovate_side = "Buy" if side_raw == "BUY" else "Sell"
        close_px = bar.get("close", 0)

        logger.info(
            "LIVE MES signal: %s @ %.2f  ->  Tradovate %s %s x1",
            side_raw, close_px, tradovate_side, MES_SYMBOL
        )
        return self.place_order(side=tradovate_side, qty=1)

    # ------------------------------------------------------------------
    # MARKET DATA
    # ------------------------------------------------------------------

    def get_latest_bar(self, symbol: str = MES_SYMBOL) -> Optional[Dict]:
        """
        Fetch the most recent quote for a contract.
        Returns a bar-compatible dict.
        """
        if not self.ready:
            return None
        try:
            quote = self._get("/md/getQuote", {"symbol": symbol})
            px = float(quote.get("trade", {}).get("price", 0) or
                       quote.get("bid", 0))
            return {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "open":   px,
                "high":   px,
                "low":    px,
                "close":  px,
                "volume": quote.get("trade", {}).get("size", 0),
            }
        except Exception as e:
            logger.warning("get_latest_bar failed: %s", e)
            return None


# ── Mock for local testing ────────────────────────────────────────────────────

class MockTradovate:
    """Simulates Tradovate ack/fill for unit tests and dry runs."""

    def __init__(self):
        self.ready = True
        self._orders = []

    def place_order(self, order: Dict) -> Dict:
        filled = {"status": "ACK", "order": order,
                  "filled_price": order.get("price")}
        self._orders.append(filled)
        return filled

    def execute_mes_signal(self, signal: Dict, bar: Dict) -> Optional[Dict]:
        side = signal.get("side", "HOLD")
        if side == "HOLD":
            return None
        return self.place_order({"side": side, "price": bar.get("close", 0), "qty": 1})

    def cancel_all_orders(self) -> bool:
        self._orders.clear()
        return True

    def close_all_positions(self) -> bool:
        return True

    def get_account(self) -> Dict:
        return {"cashBalance": 50000.0, "id": "mock"}

    def get_positions(self) -> List:
        return []

    def current_side(self) -> None:
        return None
