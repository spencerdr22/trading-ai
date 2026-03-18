"""
multi_backtest.py — Multi-parameter backtest sweep with visualization.

Runs a grid of EMA / RSI / volatility combinations and aggregates results.

Usage:
    python -m app.analysis.multi_backtest
    python -m app.analysis.multi_backtest --compare-retrain
"""

import itertools
import os
import glob
import argparse
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe on headless machines
import matplotlib.pyplot as plt
import joblib

from app.config import load_config
from app.data.simulator import stream_bars
from app.backtest.backtester import Backtester
from app.monitor.logger import get_logger
from app.db import get_session
from app.models.schema import Metric

logger = get_logger(__name__)
cfg    = load_config()

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT_DIR  = os.path.join(ROOT_DIR, "data", "multi_backtests")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Parameter grid ────────────────────────────────────────────────────────────
EMA_SHORTS   = [8,  12, 16]
EMA_LONGS    = [20, 26, 32]
RSI_PERIODS  = [10, 14, 18]
VOLATILITIES = [0.0006, 0.0008, 0.0010]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _calculate_ratios(equity_curve: list):
    """
    Compute annualised Sharpe and Sortino from an equity curve list.
    Returns (sharpe, sortino) — both float, NaN if not computable.
    """
    if len(equity_curve) < 2:
        return np.nan, np.nan

    arr     = np.array(equity_curve, dtype=float)
    returns = np.diff(arr) / (arr[:-1] + 1e-9)
    mean_r  = np.mean(returns)
    std_r   = np.std(returns)
    ann     = np.sqrt(252 * 390)   # minute-bar annualisation factor

    sharpe = float(mean_r / std_r * ann) if std_r > 0 else np.nan

    downside     = returns[returns < 0]
    downside_std = np.std(downside) if len(downside) > 1 else 0.0
    sortino = float(mean_r / downside_std * ann) if downside_std > 0 else np.nan

    return sharpe, sortino


def _log_to_db(win_rate: float, label: str):
    try:
        with get_session() as s:
            s.add(Metric(
                name      = "multi_backtest",
                value     = win_rate,
                timestamp = datetime.now(timezone.utc),
                meta      = {"label": label},
            ))
            s.commit()
    except Exception as e:
        logger.warning("DB log failed: %s", e)


# ── Single run ────────────────────────────────────────────────────────────────

def run_single_backtest(ema_short, ema_long, rsi_period, volatility):
    """Run one backtest configuration.  Returns the results dict."""
    label = f"EMA({ema_short},{ema_long}) RSI({rsi_period}) Vol={volatility}"
    logger.info("Running backtest: %s", label)

    # Shallow-copy cfg so grid changes don't bleed between runs
    import copy
    run_cfg = copy.deepcopy(cfg)
    run_cfg["model"]["features"]["ema_short"] = ema_short
    run_cfg["model"]["features"]["ema_long"]  = ema_long
    run_cfg["model"]["features"]["rsi"]       = rsi_period
    run_cfg["simulator"]["volatility"]        = volatility

    bars = list(stream_bars(
        symbol     = run_cfg.get("symbol", "MES"),
        minutes    = 1440,
        fast       = True,
        seed       = 42,
        volatility = volatility,
    ))
    df = pd.DataFrame(bars)

    bt      = Backtester(run_cfg)
    results = bt.run(df)

    sharpe, sortino = _calculate_ratios(results.get("equity_curve", []))

    results.update({
        "ema_short":  ema_short,
        "ema_long":   ema_long,
        "rsi":        rsi_period,
        "volatility": volatility,
        "sharpe":     sharpe,
        "sortino":    sortino,
        "total_pnl":  results["equity_curve"][-1] - results["equity_curve"][0]
                      if results.get("equity_curve") else 0.0,
    })

    out_path = os.path.join(OUT_DIR, f"bt_{ema_short}_{ema_long}_{rsi_period}_{volatility}.pkl")
    joblib.dump(results, out_path)
    logger.info("Saved: %s", out_path)

    _log_to_db(results.get("win_rate", 0.0), label)
    return results


# ── Aggregation & visualisation ───────────────────────────────────────────────

def aggregate_results(all_results: list):
    """Aggregate results list, save summary CSV, and produce charts."""
    if not all_results:
        logger.error("No backtest results to aggregate.")
        return pd.DataFrame()

    df = pd.DataFrame(all_results)

    expected = ["ema_short", "ema_long", "rsi", "volatility",
                "win_rate", "max_drawdown", "total_pnl", "sharpe", "sortino"]
    for col in expected:
        if col not in df.columns:
            df[col] = np.nan

    summary      = df[expected]
    summary_path = os.path.join(OUT_DIR, "summary.csv")
    summary.to_csv(summary_path, index=False)
    logger.info("Summary saved: %s", summary_path)

    try:
        _plot_results(df)
    except Exception as e:
        logger.warning("Visualisation failed: %s", e)

    return summary


def _plot_results(df: pd.DataFrame):
    """Generate three performance charts."""
    # 1. Sharpe vs Sortino scatter
    fig, ax = plt.subplots(figsize=(10, 6))
    sc = ax.scatter(df["sharpe"], df["sortino"],
                    c=df["win_rate"], cmap="viridis", s=100, edgecolor="k")
    plt.colorbar(sc, ax=ax, label="Win Rate")
    ax.set_xlabel("Sharpe Ratio")
    ax.set_ylabel("Sortino Ratio")
    ax.set_title("Risk-Adjusted Performance Across Configurations")
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "risk_adjusted_performance.png"), dpi=150)
    plt.close(fig)

    # 2. Win rate vs EMA short, grouped by volatility
    fig, ax = plt.subplots(figsize=(10, 6))
    for vol, sub in df.groupby("volatility"):
        ax.plot(sub["ema_short"], sub["win_rate"], marker="o", label=f"Vol {vol}")
    ax.set_xlabel("EMA Short Period")
    ax.set_ylabel("Win Rate")
    ax.set_title("Win Rate vs EMA Period by Volatility")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "win_rate_comparison.png"), dpi=150)
    plt.close(fig)

    # 3. PnL vs Sharpe
    fig, ax = plt.subplots(figsize=(10, 6))
    sc = ax.scatter(df["total_pnl"], df["sharpe"],
                    c=df["volatility"], cmap="coolwarm", s=80, edgecolor="k")
    plt.colorbar(sc, ax=ax, label="Volatility")
    ax.set_xlabel("Total PnL")
    ax.set_ylabel("Sharpe Ratio")
    ax.set_title("PnL vs Sharpe (coloured by volatility)")
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "pnl_vs_sharpe.png"), dpi=150)
    plt.close(fig)

    logger.info("Charts saved to %s", OUT_DIR)


# ── Retrain comparison ────────────────────────────────────────────────────────

def compare_retrain_vs_baseline():
    """Load all pkl files in multi_backtests/ and print a comparison table."""
    files = glob.glob(os.path.join(OUT_DIR, "*.pkl"))
    if not files:
        print(f"[WARN] No backtest files found in {OUT_DIR}")
        return

    rows = []
    for f in files:
        try:
            data = joblib.load(f)
            if isinstance(data, dict):
                row = data
            elif isinstance(data, pd.DataFrame):
                row = data.iloc[0].to_dict() if len(data) else {}
            else:
                continue

            row.setdefault("model_version", os.path.basename(f).replace(".pkl", ""))

            eq = row.get("equity_curve", [])
            if "total_pnl" not in row:
                row["total_pnl"] = (eq[-1] - eq[0]) if len(eq) >= 2 else 0.0

            for col in ["win_rate", "max_drawdown", "sharpe", "sortino"]:
                row.setdefault(col, None)

            rows.append(row)
        except Exception as e:
            print(f"[ERROR] {f}: {e}")

    if not rows:
        print("[WARN] No valid results loaded.")
        return

    df   = pd.DataFrame(rows)
    keep = [c for c in ["model_version", "total_pnl", "win_rate",
                         "max_drawdown", "sharpe", "sortino"] if c in df.columns]
    df   = df[keep]

    print("\n=== Model Comparison Summary ===")
    print(df.to_string(index=False))

    out = os.path.join(OUT_DIR, "comparison_summary.csv")
    df.to_csv(out, index=False)
    print(f"\n[INFO] Saved comparison to {out}")


# ── Entrypoint ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Multi-parameter backtest sweep")
    parser.add_argument("--compare-retrain", action="store_true",
                        help="Compare stored backtest results instead of running new ones")
    args = parser.parse_args()

    if args.compare_retrain:
        compare_retrain_vs_baseline()
        return

    all_results = []
    for ema_short, ema_long, rsi, vol in itertools.product(
        EMA_SHORTS, EMA_LONGS, RSI_PERIODS, VOLATILITIES
    ):
        try:
            res = run_single_backtest(ema_short, ema_long, rsi, vol)
            all_results.append(res)
        except Exception as e:
            logger.error("Backtest failed (%s,%s,%s,%s): %s",
                         ema_short, ema_long, rsi, vol, e)

    aggregate_results(all_results)


if __name__ == "__main__":
    main()
