"""
vbt_engine.py — Vectorbt-Based Backtest Engine

Wraps vectorbt.Portfolio for accurate backtesting.
Replaces the legacy custom engine while producing a compatible output dict.

Missing helpers (entities.py / util.py) are stubbed inline so the file
is self-contained and importable even if those modules don't exist yet.
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional

try:
    import vectorbt as vbt
    VBT_AVAILABLE = True
except ImportError:
    VBT_AVAILABLE = False
    warnings.warn(
        "vectorbt not installed — VectorbtEngine will raise if .run() is called. "
        "Install with: pip install vectorbt",
        ImportWarning,
        stacklevel=2,
    )


# ── Inline stubs for missing util functions ───────────────────────────────────

def _compute_week_bucket(ts, cfg: dict) -> str:
    """Return ISO-week string like '2026-W11'."""
    try:
        return ts.strftime("%G-W%V")
    except Exception:
        return "unknown"


def _compute_drawdown(pnl_values: np.ndarray) -> np.ndarray:
    cumulative = np.cumsum(pnl_values)
    running_max = np.maximum.accumulate(cumulative)
    return cumulative - running_max


def _expectancy_R(r_values: np.ndarray) -> float:
    return float(np.mean(r_values)) if len(r_values) else 0.0


def _expectancy_pnl(pnl_values: np.ndarray) -> float:
    return float(np.mean(pnl_values)) if len(pnl_values) else 0.0


# Try to import real helpers; fall back to stubs silently
try:
    from .util import compute_week_bucket, compute_drawdown, expectancy_R, expectancy_pnl  # type: ignore
except ImportError:
    compute_week_bucket = _compute_week_bucket
    compute_drawdown    = _compute_drawdown
    expectancy_R        = _expectancy_R
    expectancy_pnl      = _expectancy_pnl

try:
    from .entities import Trade  # type: ignore  # noqa: F401
except ImportError:
    Trade = None  # not used directly at runtime


# ── Main engine ───────────────────────────────────────────────────────────────

class VectorbtEngine:
    """
    Vectorbt-backed backtest engine.

    Usage:
        engine  = VectorbtEngine(strategy=my_strategy)
        results = engine.run(df)
        # results = {"metrics": {...}, "trades": DataFrame, "portfolio": vbt.Portfolio}
    """

    def __init__(
        self,
        label_spec=None,
        strategy=None,
        sir_week_cfg: Optional[Dict[str, Any]] = None,
        return_col: str = "close",
        initial_cash: float = 10_000.0,
        fees: float = 0.0,
        slippage: float = 0.0,
        size_type: str = "amount",
        size: float = 1.0,
        freq: str = "1min",
    ):
        self.label_spec   = label_spec
        self.strategy     = strategy
        self.sir_week_cfg = sir_week_cfg or {"week_start_day": "Monday"}
        self.return_col   = return_col
        self.initial_cash = initial_cash
        self.fees         = fees
        self.slippage     = slippage
        self.size_type    = size_type
        self.size         = size
        self.freq         = freq

        self._portfolio: Optional[Any] = None
        self._trades_df: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    def generate_signals(
        self,
        df: pd.DataFrame,
        model=None,
        features: Optional[np.ndarray] = None,
    ) -> pd.Series:
        """Return a Series of signals: 1 (long), -1 (short), 0 (flat)."""
        if self.strategy is None:
            return pd.Series(0, index=df.index, name="signals")

        signals = []
        if model is None:
            for i in range(len(df)):
                signals.append(self.strategy.generate_signal(df.iloc[i]))
        else:
            has_proba = hasattr(model, "predict_proba")
            if has_proba:
                proba_all = model.predict_proba(features)
                for i in range(len(df)):
                    signals.append(self.strategy.apply_model_proba(df.iloc[i], proba_all[i]))
            else:
                preds_all = model.predict(features)
                for i in range(len(df)):
                    signals.append(self.strategy.apply_model_prediction(df.iloc[i], preds_all[i]))

        return pd.Series(signals, index=df.index, name="signals")

    # ------------------------------------------------------------------
    def run(
        self,
        df: pd.DataFrame,
        model=None,
        features: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Run the backtest and return a results dict."""
        if not VBT_AVAILABLE:
            raise ImportError(
                "vectorbt is required. Install with: pip install vectorbt"
            )

        # Ensure DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            if "datetime" in df.columns:
                df = df.set_index("datetime")
            elif "timestamp" in df.columns:
                df = df.set_index("timestamp")
            else:
                df.index = pd.date_range(
                    start="2020-01-01", periods=len(df), freq=self.freq
                )

        prices  = df[self.return_col]
        signals = self.generate_signals(df, model, features)

        prev = signals.shift(1, fill_value=0)
        entries_long  = (signals ==  1) & (prev != 1)
        entries_short = (signals == -1) & (prev != -1)
        exits_long    = ((signals == 0) | (signals == -1)) & (prev == 1)
        exits_short   = ((signals == 0) | (signals ==  1)) & (prev == -1)

        portfolio = vbt.Portfolio.from_signals(
            close        = prices,
            entries      = entries_long,
            exits        = exits_long,
            short_entries= entries_short,
            short_exits  = exits_short,
            init_cash    = self.initial_cash,
            fees         = self.fees,
            slippage     = self.slippage,
            size         = self.size,
            size_type    = self.size_type,
            freq         = self.freq,
        )
        self._portfolio = portfolio

        trades_df       = self._extract_trades(portfolio, df)
        self._trades_df = trades_df
        metrics         = self._compute_metrics(portfolio, trades_df)

        return {"metrics": metrics, "trades": trades_df, "portfolio": portfolio}

    # ------------------------------------------------------------------
    def _extract_trades(self, portfolio, df: pd.DataFrame) -> pd.DataFrame:
        COLS = [
            "entry_index", "exit_index", "entry_price", "exit_price",
            "direction", "volatility", "pnl", "R",
            "timestamp_entry", "timestamp_exit", "week_bucket",
        ]
        empty = pd.DataFrame(columns=COLS)

        try:
            vbt_trades = portfolio.trades.records_readable
            if len(vbt_trades) == 0:
                return empty

            rows = []
            for _, trade in vbt_trades.iterrows():
                ei  = int(trade["Entry Index"])
                xi  = int(trade["Exit Index"])
                vol = 0.0001
                pnl = float(trade["PnL"])
                R   = pnl / vol
                ts_entry = df.index[ei]
                ts_exit  = df.index[xi]
                rows.append({
                    "entry_index":   ei,
                    "exit_index":    xi,
                    "entry_price":   float(trade["Avg Entry Price"]),
                    "exit_price":    float(trade["Avg Exit Price"]),
                    "direction":     1 if trade["Direction"] == "Long" else -1,
                    "volatility":    vol,
                    "pnl":           pnl,
                    "R":             float(R),
                    "timestamp_entry": ts_entry,
                    "timestamp_exit":  ts_exit,
                    "week_bucket":   compute_week_bucket(ts_entry, self.sir_week_cfg),
                })
            return pd.DataFrame(rows)
        except Exception as e:
            warnings.warn(f"VectorbtEngine: trade extraction failed: {e}")
            return empty

    # ------------------------------------------------------------------
    def _compute_metrics(self, portfolio, trades_df: pd.DataFrame) -> Dict[str, Any]:
        try:
            stats = portfolio.stats()
        except Exception:
            stats = {}

        def _s(key, default=0.0):
            v = stats.get(key, default)
            return 0.0 if (v is None or (isinstance(v, float) and np.isnan(v))) else float(v)

        n_trades = int(_s("Total Trades"))
        wr_pct   = _s("Win Rate [%]")

        if len(trades_df):
            R_vals  = trades_df["R"].values
            pnl_vals= trades_df["pnl"].values
            wkly    = trades_df.groupby("week_bucket")
            wkly_R  = wkly["R"].mean().to_dict()
            wkly_pnl= wkly["pnl"].mean().to_dict()
            dd_curve= compute_drawdown(pnl_vals)
            max_dd_pnl = float(dd_curve.min()) if len(dd_curve) else 0.0
        else:
            R_vals  = np.array([])
            pnl_vals= np.array([])
            wkly_R  = {}
            wkly_pnl= {}
            max_dd_pnl = 0.0

        return {
            "num_trades":              n_trades,
            "win_rate":                wr_pct / 100.0,
            "profit_factor":           _s("Profit Factor"),
            "total_return":            _s("Total Return [%]") / 100.0,
            "sharpe_ratio":            _s("Sharpe Ratio"),
            "sortino_ratio":           _s("Sortino Ratio"),
            "calmar_ratio":            _s("Calmar Ratio"),
            "max_drawdown":            _s("Max Drawdown [%]") / 100.0,
            "per_trade_expectancy_R":  expectancy_R(R_vals),
            "per_trade_expectancy_pnl":expectancy_pnl(pnl_vals),
            "weekly_expectancy_R":     wkly_R,
            "weekly_expectancy_pnl":   wkly_pnl,
            "overall_expectancy_R":    expectancy_R(R_vals),
            "overall_expectancy_pnl":  expectancy_pnl(pnl_vals),
            "max_drawdown_pnl":        max_dd_pnl,
        }

    # ------------------------------------------------------------------
    def get_portfolio_stats(self) -> pd.Series:
        if self._portfolio is None:
            raise ValueError("Run .run() first.")
        return self._portfolio.stats()

    def get_equity_curve(self) -> pd.Series:
        if self._portfolio is None:
            raise ValueError("Run .run() first.")
        return self._portfolio.value()

    def get_returns(self) -> pd.Series:
        if self._portfolio is None:
            raise ValueError("Run .run() first.")
        return self._portfolio.returns()

    def plot_portfolio(self, **kwargs):
        if self._portfolio is None:
            raise ValueError("Run .run() first.")
        return self._portfolio.plot(**kwargs)


# ── Convenience wrapper ───────────────────────────────────────────────────────

def run_backtest(
    df: pd.DataFrame,
    strategy,
    label_spec=None,
    sir_week_cfg: Optional[Dict[str, Any]] = None,
    model=None,
    features: Optional[np.ndarray] = None,
    **kwargs,
) -> Dict[str, Any]:
    engine = VectorbtEngine(
        label_spec   = label_spec,
        strategy     = strategy,
        sir_week_cfg = sir_week_cfg,
        **kwargs,
    )
    return engine.run(df, model=model, features=features)
