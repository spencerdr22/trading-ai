from .simulator_executor import simulate_fill
from typing import Dict
import logging

logger = logging.getLogger(__name__)

class PaperExecutor:
    def __init__(self, config):
        self.config = config
        self.positions = []

    def place_order(self, bar, signal):
        if signal["side"] == "HOLD":
            return None
        fill_price, slip = simulate_fill(float(bar["close"]), signal["side"], 1, slippage_std=0.05)
        trade = {
            "timestamp": bar["timestamp"],
            "side": signal["side"],
            "price": fill_price,
            "size": 1,
            "slippage": slip,
            "commission": self.config["risk"]["contract_cost"]
        }
        logger.info("Paper trade executed: %s", trade)
        self.positions.append(trade)
        return trade
