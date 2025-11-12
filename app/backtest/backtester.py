# File: app/backtest/backtester.py

from typing import Dict, Any, List
import pandas as pd
from ..ml.trainer import Trainer   # ✅ fixed import
from ..strategy.engine import StrategyEngine
from ..strategy.adaption import Adaptor
from ..execution.simulator_executor import execute_trade, exit_with_sl_tp
from .metrics import compute_win_rate, equity_curve_from_trades, max_drawdown
from ..ml.features import make_features
from ..db import get_session
from ..models.schema import Trade, Metric
import logging

logger = logging.getLogger(__name__)


class Backtester:
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.trainer = Trainer()  # ✅ fixed from Training() to Trainer()

    def _persist_trades_metrics(self, trades: List[Dict], win_rate: float, drawdown: float):
        with get_session() as s:
            for t in trades:
                s.add(Trade(
                    symbol=self.cfg["symbol"],
                    timestamp=t["timestamp"],
                    side=t["side"],
                    price=t["price"],
                    size=t["size"],
                    pnl=t["pnl"],
                    commission=t["commission"],
                    slippage=t["slippage"],
                    status="FILLED",
                    meta={"reason": t.get("reason", "EOD")}
                ))
            s.add(Metric(name="backtest_win_rate", value=win_rate))
            s.add(Metric(name="backtest_max_drawdown", value=drawdown))
            s.commit()

    def run(self, df: pd.DataFrame):
        n = len(df)
        if n < 120:
            raise ValueError("Not enough data to backtest (need >=120 bars).")

        split = int(n * 0.5)
        train_df = df.iloc[:split]
        test_df = df.iloc[split:]

        logger.info("Training on %d bars, testing on %d bars", len(train_df), len(test_df))

        predictor = self.trainer.train(train_df)

        adaptor = Adaptor(
            threshold_up=0.62, threshold_down=0.62,
            stop_loss=self.cfg["risk"]["stop_loss_ticks"],
            take_profit=self.cfg["risk"]["take_profit_ticks"]
        )

        strategy = StrategyEngine(predictor, adaptor, classes=[-1, 0, 1], cfg=self.cfg)

        feat = make_features(df, window=self.cfg["model"]["window"])
        test_feat = feat[feat["timestamp"] >= test_df["timestamp"].min()].reset_index(drop=True)
        X = test_feat.drop(columns=["timestamp", "open", "high", "low", "close", "volume"], errors="ignore")

        trades: List[Dict] = []
        for idx in range(len(test_feat) - 1):
            row = test_feat.iloc[idx]
            X_row = X.loc[[idx]]

            signal = strategy.on_bar(X_row)
            exec_info = execute_trade(row, signal, self.cfg)

            if not exec_info:
                continue

            next_bar = test_feat.iloc[idx + 1]
            exit_price, pnl, reason = exit_with_sl_tp(exec_info, next_bar, self.cfg)

            trade = {
                "timestamp": exec_info["timestamp"],
                "side": exec_info["side"],
                "price": exec_info["price"],
                "size": exec_info["size"],
                "pnl": float(pnl),
                "commission": exec_info["commission"],
                "slippage": exec_info["slippage"],
                "reason": reason
            }
            trades.append(trade)

            # Let strategy adapt periodically
            if len(trades) % 50 == 0:
                strategy.adapt(trades[-100:])

        win_rate = compute_win_rate(trades)
        equity = equity_curve_from_trades(trades)
        dd = max_drawdown(equity)

        try:
            self._persist_trades_metrics(trades, win_rate, dd)
        except Exception as e:
            logger.warning("Failed to persist trades/metrics: %s", e)

        return {"trades": trades, "win_rate": win_rate, "equity_curve": equity, "max_drawdown": dd}
