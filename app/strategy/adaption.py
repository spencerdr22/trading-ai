from typing import List, Dict
import numpy as np

class Adaptor:
    """
    Updates strategy thresholds and stop levels based on recent trade performance.
    Simple conservative rules: small incremental adjustments.
    """
    def __init__(self, threshold_up=0.6, threshold_down=0.6, stop_loss=8, take_profit=12):
        self.threshold_up = threshold_up
        self.threshold_down = threshold_down
        self.stop_loss = stop_loss
        self.take_profit = take_profit

    def update(self, recent_trades: List[Dict]) -> Dict:
        """
        recent_trades: list of trades with keys ['pnl', 'side']
        """
        pnl = [t["pnl"] for t in recent_trades if "pnl" in t]
        wins = [p for p in pnl if p > 0]
        losses = [p for p in pnl if p <= 0]
        win_rate = len(wins) / max(1, len(pnl))
        # Adjust thresholds conservatively
        if win_rate < 0.45:
            # make thresholds stricter
            self.threshold_up = min(0.95, self.threshold_up + 0.02)
            self.threshold_down = min(0.95, self.threshold_down + 0.02)
            self.stop_loss = max(1, self.stop_loss - 1)
        elif win_rate > 0.6:
            # loosen thresholds slightly to capture more trades
            self.threshold_up = max(0.5, self.threshold_up - 0.01)
            self.threshold_down = max(0.5, self.threshold_down - 0.01)
            self.stop_loss = self.stop_loss + 0
        # Keep take_profit related to reward/risk seen
        avg_win = np.mean(wins) if wins else 0
        avg_loss = -np.mean(losses) if losses else 0
        if avg_loss > 0 and avg_win / (avg_loss + 1e-9) < 0.5:
            self.take_profit = max(4, self.take_profit - 1)
        return {
            "threshold_up": self.threshold_up,
            "threshold_down": self.threshold_down,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "win_rate": win_rate
        }
