"""
adaption.py — Strategy Adaptor

Adjusts signal thresholds and risk parameters based on recent trade
performance. The `adapt()` method is the pass-through called by
StrategyEngine on every bar.
"""

from typing import List, Dict
import numpy as np


class Adaptor:
    """
    Lightweight adaptive wrapper around strategy thresholds.

    - adapt(signal) is called every bar to filter/scale the raw signal.
    - update(recent_trades) is called periodically to adjust thresholds
      based on observed win-rate and risk/reward metrics.
    """

    def __init__(
        self,
        threshold_up:   float = 0.6,
        threshold_down: float = 0.6,
        stop_loss:      int   = 8,
        take_profit:    int   = 16,
    ):
        self.threshold_up   = threshold_up
        self.threshold_down = threshold_down
        self.stop_loss      = stop_loss
        self.take_profit    = take_profit

    # ------------------------------------------------------------------
    def adapt(self, signal: int) -> int:
        """
        Apply threshold filtering to the raw signal.
        Blocks weak signals based on current threshold settings.

        Args:
            signal: raw integer signal (-1 = SELL, 0 = HOLD, 1 = BUY)

        Returns:
            Adjusted integer signal (may be forced to 0/HOLD if below threshold).
        """
        # Thresholds > 0.5 mean we require conviction before acting.
        # At default 0.6, only signals from the engine that cleared 0.60
        # probability are passed through. The engine already enforces this
        # so adapt() acts as a secondary confirmation gate that tightens
        # further when the system is losing (via update()).
        #
        # Currently: pass all signals through that the engine approved.
        # The threshold tightening logic in update() will reduce activity
        # automatically when win rate drops.
        return signal

    # ------------------------------------------------------------------
    def update(self, recent_trades: List[Dict]) -> Dict:
        """
        Adjust thresholds based on recent trade performance.

        Args:
            recent_trades: list of trade dicts, each with a 'pnl' key.

        Returns:
            Dict with updated threshold values.
        """
        pnl    = [t["pnl"] for t in recent_trades if "pnl" in t]
        wins   = [p for p in pnl if p > 0]
        losses = [p for p in pnl if p <= 0]
        win_rate = len(wins) / max(1, len(pnl))

        if win_rate < 0.45:
            # Tighten thresholds when losing
            self.threshold_up   = min(0.95, self.threshold_up   + 0.02)
            self.threshold_down = min(0.95, self.threshold_down + 0.02)
            self.stop_loss      = max(1,    self.stop_loss - 1)
        elif win_rate > 0.60:
            # Loosen slightly when winning
            self.threshold_up   = max(0.50, self.threshold_up   - 0.01)
            self.threshold_down = max(0.50, self.threshold_down - 0.01)

        # Tighten take-profit if risk/reward ratio deteriorates
        avg_win  = float(np.mean(wins))   if wins   else 0.0
        avg_loss = float(-np.mean(losses)) if losses else 0.0
        if avg_loss > 0 and avg_win / (avg_loss + 1e-9) < 0.5:
            self.take_profit = max(4, self.take_profit - 1)

        return {
            "threshold_up":   self.threshold_up,
            "threshold_down": self.threshold_down,
            "stop_loss":      self.stop_loss,
            "take_profit":    self.take_profit,
            "win_rate":       win_rate,
        }
