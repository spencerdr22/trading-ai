# ============================================================
# vbt_engine.py — Vectorbt-Based Backtest Engine
# Version: 2.0
# ============================================================

from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Union

try:
    import vectorbt as vbt
except ImportError:
    raise ImportError(
        "vectorbt is required for VectorbtEngine. "
        "Install it with: pip install vectorbt"
    )

from .entities import Trade
from .util import compute_week_bucket, compute_drawdown, expectancy_R, expectancy_pnl


class VectorbtEngine:
    """
    Vectorbt-based backtest engine for the research pipeline.
    
    Replaces the custom BacktestEngine with industry-standard vectorbt
    for accurate, well-tested backtesting.
    
    Features:
      - Vectorized portfolio simulation via vectorbt
      - Support for long/short positions
      - Accurate slippage, fees, and transaction costs
      - Comprehensive performance metrics
      - Compatible with existing strategy interface
      - Produces same output format as legacy engine
      
    Key Differences from Legacy Engine:
      - Uses vectorbt.Portfolio for simulation
      - More accurate position sizing and exits
      - Better handling of complex scenarios
      - Industry-standard metrics calculation
    """

    def __init__(
        self,
        label_spec=None,
        strategy=None,
        sir_week_cfg: Optional[Dict[str, Any]] = None,
        return_col: str = "close",
        initial_cash: float = 10000.0,
        fees: float = 0.0,
        slippage: float = 0.0,
        size_type: str = "amount",
        size: float = 1.0,
        freq: str = "1h",
    ):
        """
        Initialize VectorbtEngine.
        
        Args:
            label_spec: Label specification (for compatibility)
            strategy: Strategy interface instance (for compatibility)
            sir_week_cfg: SIR week configuration for metrics grouping
            return_col: Column name for price data (default: "close")
            initial_cash: Initial capital for portfolio
            fees: Transaction fees as percentage (0.001 = 0.1%)
            slippage: Slippage as percentage (0.001 = 0.1%)
            size_type: Position sizing type ("amount", "percent", "shares")
            size: Position size (interpretation depends on size_type)
            freq: Data frequency for time-based calculations
        """
        self.label_spec = label_spec
        self.strategy = strategy
        self.sir_week_cfg = sir_week_cfg or {"week_start_day": "Monday", "week_end_day": "Friday"}
        self.return_col = return_col
        self.initial_cash = initial_cash
        self.fees = fees
        self.slippage = slippage
        self.size_type = size_type
        self.size = size
        self.freq = freq
        
        # Cache for portfolio results
        self._portfolio = None
        self._trades_df = None

    def generate_signals(
        self,
        df: pd.DataFrame,
        model=None,
        features=None
    ) -> pd.Series:
        """
        Generate trading signals using strategy.
        
        Args:
            df: Price data DataFrame with index as timestamps
            model: Optional ML model for predictions
            features: Optional feature array for model
            
        Returns:
            pd.Series with signals: 1 (long), -1 (short), 0 (flat)
        """
        signals = []
        
        if model is None:
            # Rules-based strategy
            for i in range(len(df)):
                signal = self.strategy.generate_signal(df.iloc[i])
                signals.append(signal)
        else:
            # Model-based strategy
            has_proba = hasattr(model, "predict_proba")
            
            if has_proba:
                proba_all = model.predict_proba(features)
                for i in range(len(df)):
                    signal = self.strategy.apply_model_proba(df.iloc[i], proba_all[i])
                    signals.append(signal)
            else:
                preds_all = model.predict(features)
                for i in range(len(df)):
                    signal = self.strategy.apply_model_prediction(df.iloc[i], preds_all[i])
                    signals.append(signal)
        
        return pd.Series(signals, index=df.index, name="signals")

    def run(
        self,
        df: pd.DataFrame,
        model=None,
        features=None
    ) -> Dict[str, Any]:
        """
        Run backtest using vectorbt.
        
        Args:
            df: Price data DataFrame (must have DatetimeIndex)
            model: Optional trained model
            features: Optional feature array for model predictions
            
        Returns:
            Dictionary containing:
                - metrics: Performance metrics dict
                - trades: DataFrame of individual trades
                - portfolio: vectorbt Portfolio object
        """
        # Ensure datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            if "datetime" in df.columns:
                df = df.set_index("datetime")
            else:
                warnings.warn("DataFrame lacks DatetimeIndex, using range index")
                df.index = pd.date_range(start="2020-01-01", periods=len(df), freq=self.freq)
        
        # Get price data
        prices = df[self.return_col]
        
        # Generate signals
        signals = self.generate_signals(df, model, features)
        
        # Convert signals to entries and exits for vectorbt
        # Entry: signal changes from 0 to 1/-1 or changes sign
        entries_long = (signals == 1) & (signals.shift(1, fill_value=0) != 1)
        entries_short = (signals == -1) & (signals.shift(1, fill_value=0) != -1)
        
        # Exit: signal changes from 1/-1 to 0 or opposite direction
        exits_long = ((signals == 0) | (signals == -1)) & (signals.shift(1, fill_value=0) == 1)
        exits_short = ((signals == 0) | (signals == 1)) & (signals.shift(1, fill_value=0) == -1)
        
        # Create portfolio using vectorbt
        portfolio = vbt.Portfolio.from_signals(
            close=prices,
            entries=entries_long,
            exits=exits_long,
            short_entries=entries_short,
            short_exits=exits_short,
            init_cash=self.initial_cash,
            fees=self.fees,
            slippage=self.slippage,
            size=self.size,
            size_type=self.size_type,
            freq=self.freq,
        )
        
        self._portfolio = portfolio
        
        # Extract trades
        trades_df = self._extract_trades(portfolio, df)
        self._trades_df = trades_df
        
        # Compute metrics
        metrics = self._compute_metrics(portfolio, trades_df)
        
        return {
            "metrics": metrics,
            "trades": trades_df,
            "portfolio": portfolio,
        }

    def _extract_trades(
        self,
        portfolio: vbt.Portfolio,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Extract individual trades from vectorbt portfolio.
        
        Args:
            portfolio: vectorbt Portfolio object
            df: Original price DataFrame
            
        Returns:
            DataFrame with trade information compatible with legacy format
        """
        try:
            # Get trades from vectorbt
            vbt_trades = portfolio.trades.records_readable
            
            if len(vbt_trades) == 0:
                # No trades executed
                return pd.DataFrame(columns=[
                    "entry_index", "exit_index", "entry_price", "exit_price",
                    "direction", "volatility", "pnl", "R", 
                    "timestamp_entry", "timestamp_exit", "week_bucket"
                ])
            
            # Build trades list
            trades = []
            
            for _, trade in vbt_trades.iterrows():
                entry_idx = trade["Entry Index"]
                exit_idx = trade["Exit Index"]
                
                # Get volatility from label_spec if available
                if self.label_spec and hasattr(self.label_spec, "volatility_col"):
                    vol_col = self.label_spec.volatility_col
                    volatility = df.iloc[entry_idx][vol_col] if vol_col in df.columns else 0.0001
                else:
                    volatility = 0.0001
                
                # Calculate R-multiple
                pnl = float(trade["PnL"])
                R = (pnl / volatility) if volatility != 0 else 0.0
                
                # Get timestamps
                ts_entry = df.index[entry_idx]
                ts_exit = df.index[exit_idx]
                
                # Compute week bucket
                week_bucket = compute_week_bucket(ts_entry, self.sir_week_cfg)
                
                # Direction: 1 for long, -1 for short
                direction = 1 if trade["Direction"] == "Long" else -1
                
                trade_dict = {
                    "entry_index": int(entry_idx),
                    "exit_index": int(exit_idx),
                    "entry_price": float(trade["Avg Entry Price"]),
                    "exit_price": float(trade["Avg Exit Price"]),
                    "direction": direction,
                    "volatility": float(volatility),
                    "pnl": pnl,
                    "R": float(R),
                    "timestamp_entry": ts_entry,
                    "timestamp_exit": ts_exit,
                    "week_bucket": week_bucket,
                }
                
                trades.append(trade_dict)
            
            return pd.DataFrame(trades)
            
        except Exception as e:
            warnings.warn(f"Error extracting trades: {e}")
            return pd.DataFrame(columns=[
                "entry_index", "exit_index", "entry_price", "exit_price",
                "direction", "volatility", "pnl", "R", 
                "timestamp_entry", "timestamp_exit", "week_bucket"
            ])

    def _compute_metrics(
        self,
        portfolio: vbt.Portfolio,
        trades_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Compute comprehensive performance metrics.
        
        Args:
            portfolio: vectorbt Portfolio object
            trades_df: DataFrame of trades
            
        Returns:
            Dictionary of performance metrics
        """
        # Use vectorbt's built-in metrics
        stats = portfolio.stats()
        
        # Custom metrics from trades
        if len(trades_df) > 0:
            R_values = trades_df["R"].values
            pnl_values = trades_df["pnl"].values
            
            # Per-trade expectancy
            per_trade_expectancy_R = expectancy_R(R_values)
            per_trade_expectancy_pnl = expectancy_pnl(pnl_values)
            
            # Weekly expectancy
            weekly_group = trades_df.groupby("week_bucket")
            weekly_exp_R = weekly_group["R"].mean().to_dict()
            weekly_exp_pnl = weekly_group["pnl"].mean().to_dict()
            
            # Overall expectancy
            overall_exp_R = expectancy_R(R_values)
            overall_exp_pnl = expectancy_pnl(pnl_values)
            
            # Drawdown
            dd_curve = compute_drawdown(pnl_values)
            max_dd_pnl = float(dd_curve.min()) if len(dd_curve) else 0.0
        else:
            per_trade_expectancy_R = 0.0
            per_trade_expectancy_pnl = 0.0
            weekly_exp_R = {}
            weekly_exp_pnl = {}
            overall_exp_R = 0.0
            overall_exp_pnl = 0.0
            max_dd_pnl = 0.0
        
        # Extract key vectorbt metrics
        total_trades = int(stats.get("Total Trades", 0))
        win_rate_pct = stats.get("Win Rate [%]", 0)
        # Handle NaN for win rate when no trades
        if pd.isna(win_rate_pct) or total_trades == 0:
            win_rate_pct = 0
        
        metrics = {
            # Trade counts
            "num_trades": total_trades,
            "num_winning_trades": int(win_rate_pct * total_trades / 100),
            "num_losing_trades": int((100 - win_rate_pct) * total_trades / 100),
            
            # Win rate and profit factor
            "win_rate": float(stats.get("Win Rate [%]", 0)) / 100.0,
            "profit_factor": float(stats.get("Profit Factor", 0)),
            
            # Returns
            "total_return": float(stats.get("Total Return [%]", 0)) / 100.0,
            "sharpe_ratio": float(stats.get("Sharpe Ratio", 0)),
            "sortino_ratio": float(stats.get("Sortino Ratio", 0)),
            "calmar_ratio": float(stats.get("Calmar Ratio", 0)),
            
            # Drawdown (vectorbt's version)
            "max_drawdown": float(stats.get("Max Drawdown [%]", 0)) / 100.0,
            "max_drawdown_duration": str(stats.get("Max Drawdown Duration", "0")),
            
            # Custom R-based metrics
            "per_trade_expectancy_R": float(per_trade_expectancy_R),
            "per_trade_expectancy_pnl": float(per_trade_expectancy_pnl),
            "weekly_expectancy_R": weekly_exp_R,
            "weekly_expectancy_pnl": weekly_exp_pnl,
            "overall_expectancy_R": float(overall_exp_R),
            "overall_expectancy_pnl": float(overall_exp_pnl),
            "max_drawdown_pnl": float(max_dd_pnl),
            
            # Additional metrics
            "avg_winning_trade": float(stats.get("Avg Winning Trade [%]", 0)) / 100.0,
            "avg_losing_trade": float(stats.get("Avg Losing Trade [%]", 0)) / 100.0,
            "expectancy": float(stats.get("Expectancy", 0)),
        }
        
        return metrics

    def get_portfolio_stats(self) -> pd.Series:
        """
        Get full vectorbt portfolio statistics.
        
        Returns:
            pd.Series with all available statistics
        """
        if self._portfolio is None:
            raise ValueError("Must run backtest first using .run()")
        
        return self._portfolio.stats()

    def plot_portfolio(self, **kwargs):
        """
        Plot portfolio performance using vectorbt's visualization.
        
        Args:
            **kwargs: Additional arguments passed to portfolio.plot()
        """
        if self._portfolio is None:
            raise ValueError("Must run backtest first using .run()")
        
        return self._portfolio.plot(**kwargs)

    def get_returns(self) -> pd.Series:
        """
        Get portfolio returns series.
        
        Returns:
            pd.Series of returns
        """
        if self._portfolio is None:
            raise ValueError("Must run backtest first using .run()")
        
        return self._portfolio.returns()

    def get_equity_curve(self) -> pd.Series:
        """
        Get portfolio equity curve.
        
        Returns:
            pd.Series of cumulative equity
        """
        if self._portfolio is None:
            raise ValueError("Must run backtest first using .run()")
        
        return self._portfolio.value()


# Convenience function for backward compatibility
def run_backtest(
    df: pd.DataFrame,
    strategy,
    label_spec=None,
    sir_week_cfg: Optional[Dict[str, Any]] = None,
    model=None,
    features=None,
    **kwargs
) -> Dict[str, Any]:
    """
    Convenience function to run backtest with vectorbt engine.
    
    Args:
        df: Price data DataFrame
        strategy: Strategy instance
        label_spec: Label specification
        sir_week_cfg: SIR week configuration
        model: Optional ML model
        features: Optional feature array
        **kwargs: Additional arguments for VectorbtEngine
        
    Returns:
        Dictionary with backtest results
    """
    engine = VectorbtEngine(
        label_spec=label_spec,
        strategy=strategy,
        sir_week_cfg=sir_week_cfg,
        **kwargs
    )
    
    return engine.run(df, model=model, features=features)


if __name__ == "__main__":
    """Demo usage of VectorbtEngine."""
    print("\n📊 VECTORBT BACKTEST ENGINE")
    print("=" * 60)
    print("Vectorbt-based backtesting with industry-standard metrics")
    print("\nUsage:")
    print("  from backtests.vbt_engine import VectorbtEngine")
    print("  engine = VectorbtEngine(strategy=my_strategy)")
    print("  results = engine.run(df)")
    print("\nFeatures:")
    print("  ✓ Accurate portfolio simulation")
    print("  ✓ Proper handling of slippage and fees")
    print("  ✓ Comprehensive performance metrics")
    print("  ✓ Compatible with existing strategy interface")
    print("  ✓ Rich visualization capabilities")
    print("\n" + "=" * 60)
