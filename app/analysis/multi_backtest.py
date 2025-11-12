"""
Multi-parameter backtest validation suite with performance metrics.

This script runs multiple backtests with different configs and
summarizes the results to evaluate model robustness.
"""

import itertools
import logging
import os
import glob
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import argparse

from datetime import datetime
from app.config import load_config
from app.data.simulator import stream_bars
from app.backtest.backtester import Backtester
from app.monitor.logger import get_logger
from app.db import get_session
from app.models.schema import Metric
from app.ml.training import retrain_on_recent

# === Ensure output folder exists ===
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT_DIR = os.path.join(ROOT_DIR, "data", "multi_backtests")
os.makedirs(OUT_DIR, exist_ok=True)

# Logging setup
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

logger = get_logger(__name__)
cfg = load_config()

# === Parameter grid for testing ===
EMA_SHORTS = [8, 12, 16]
EMA_LONGS = [20, 26, 32]
RSI_PERIODS = [10, 14, 18]
VOLATILITIES = [0.0006, 0.0008, 0.0010]

OUT_DIR = os.path.join(os.getcwd(), "data", "multi_backtests")
os.makedirs(OUT_DIR, exist_ok=True)


def calculate_ratios(pnl_series):
    """
    Compute Sharpe and Sortino ratios from PnL series.
    """
    if pnl_series.empty:
        return np.nan, np.nan

    returns = pnl_series.pct_change().dropna()
    mean_ret = returns.mean()
    std_ret = returns.std()

    sharpe = (mean_ret / std_ret) * np.sqrt(252 * 24 * 60) if std_ret > 0 else np.nan

    downside = returns[returns < 0]
    downside_std = downside.std()
    sortino = (mean_ret / downside_std) * np.sqrt(252 * 24 * 60) if downside_std > 0 else np.nan

    return sharpe, sortino


def run_single_backtest(ema_short, ema_long, rsi_period, volatility):
    """Run a single backtest with given parameters."""
    logger.info(f"Running backtest: EMA({ema_short},{ema_long}) RSI({rsi_period}) Vol={volatility}")

    # Update config for this run
    cfg["model"]["features"]["ema_short"] = ema_short
    cfg["model"]["features"]["ema_long"] = ema_long
    cfg["model"]["features"]["rsi"] = rsi_period
    cfg["simulator"]["volatility"] = volatility

    # Generate simulated bars
    bars = list(stream_bars(symbol=cfg["symbol"], minutes=1440, fast=True, seed=42, volatility=volatility))
    df = pd.DataFrame(bars)

    # Run backtest
    bt = Backtester(cfg)
    results = bt.run(df)
    pnl_series = results.get("pnl_series", pd.Series(dtype=float))

    sharpe, sortino = calculate_ratios(pnl_series)

    results["ema_short"] = ema_short
    results["ema_long"] = ema_long
    results["rsi"] = rsi_period
    results["volatility"] = volatility
    results["sharpe"] = sharpe
    results["sortino"] = sortino

    # Save results
    out_path = os.path.join(OUT_DIR, f"bt_{ema_short}_{ema_long}_{rsi_period}_{volatility}.pkl")
    joblib.dump(results, out_path)
    logger.info(f"Saved: {out_path}")

    # Log to DB (optional)
    try:
        with get_session() as s:
            s.add(Metric(
                name="multi_backtest",
                value=results.get("win_rate", 0),
                timestamp=datetime.utcnow(),
                meta=f"{ema_short},{ema_long},{rsi_period},{volatility}",
            ))
            s.commit()
    except Exception as e:
        logger.warning(f"Could not log to DB: {e}")

    return results


def aggregate_results(all_results):
    """Aggregate all test results, handle missing fields gracefully, and visualize."""
    import numpy as np

    df = pd.DataFrame(all_results)
    if df.empty:
        logger.error("No backtest results found!")
        return

    logger.info(f"Aggregating {len(df)} backtest result entries...")

    # === Handle missing columns safely ===
    expected_cols = [
        "ema_short",
        "ema_long",
        "rsi",
        "volatility",
        "win_rate",
        "max_drawdown",
        "total_pnl",
        "sharpe",
        "sortino"
    ]

    for col in expected_cols:
        if col not in df.columns:
            logger.warning(f"Missing column '{col}' in results — filling with NaN.")
            df[col] = np.nan

    # === Save summary CSV ===
    summary = df[expected_cols]
    summary_path = os.path.join(OUT_DIR, "summary.csv")
    summary.to_csv(summary_path, index=False)
    logger.info(f"Saved summary CSV to: {summary_path}")

    # === Visualization Section ===
    try:
        # Risk-adjusted performance
        plt.figure(figsize=(10, 6))
        plt.scatter(
            df["sharpe"],
            df["sortino"],
            c=df["win_rate"],
            cmap="viridis",
            s=100,
            edgecolor="k"
        )
        plt.xlabel("Sharpe Ratio")
        plt.ylabel("Sortino Ratio")
        plt.title("Risk-Adjusted Performance Across Configurations")
        plt.colorbar(label="Win Rate")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, "risk_adjusted_performance.png"))
        logger.info("Saved: risk_adjusted_performance.png")

        # Win rate vs volatility
        plt.figure(figsize=(10, 6))
        for vol, sub in df.groupby("volatility"):
            plt.plot(sub["ema_short"], sub["win_rate"], marker="o", label=f"Vol {vol}")
        plt.xlabel("EMA Short Period")
        plt.ylabel("Win Rate")
        plt.title("Win Rate Across EMA Periods and Volatility Levels")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, "win_rate_comparison.png"))
        logger.info("Saved: win_rate_comparison.png")

        # PnL vs Sharpe
        plt.figure(figsize=(10, 6))
        plt.scatter(
            df["total_pnl"],
            df["sharpe"],
            c=df["volatility"],
            cmap="coolwarm",
            s=80,
            edgecolor="k"
        )
        plt.xlabel("Total PnL")
        plt.ylabel("Sharpe Ratio")
        plt.title("PnL vs Sharpe (colored by volatility)")
        plt.colorbar(label="Volatility")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, "pnl_vs_sharpe.png"))
        logger.info("Saved: pnl_vs_sharpe.png")

    except Exception as e:
        logger.error(f"Visualization failed: {e}")

    logger.info("Aggregation and visualization complete.")
    return summary



def compare_retrain_vs_baseline():
    """
    Compare baseline vs retrained model performance based on stored backtest results.
    This version ensures missing columns like total_pnl are handled gracefully.
    """
    base_dir = os.path.join("data", "multi_backtests")
    files = glob.glob(os.path.join(base_dir, "*.pkl"))

    if not files:
        print("[WARN] No backtest files found in", base_dir)
        return

    results = []
    for f in files:
        try:
            data = pd.read_pickle(f)

            # Some backtest results might be dicts, others DataFrames
            if isinstance(data, dict):
                df = pd.DataFrame([data])
            elif isinstance(data, pd.DataFrame):
                df = data
            else:
                print(f"[WARN] Unsupported data type in {f}: {type(data)}")
                continue

            # Add model version if missing
            if "model_version" not in df.columns:
                df["model_version"] = os.path.basename(f).replace(".pkl", "")

            # Handle missing total_pnl column
            if "total_pnl" not in df.columns:
                if "pnl" in df.columns:
                    df["total_pnl"] = df["pnl"].sum()
                elif "pnl_series" in df.columns:
                    df["total_pnl"] = sum(df["pnl_series"])
                else:
                    df["total_pnl"] = 0.0  # fallback

            # Add missing optional metrics
            for col in ["win_rate", "max_drawdown", "sharpe", "sortino"]:
                if col not in df.columns:
                    df[col] = None

            results.append(df)
        except Exception as e:
            print(f"[ERROR] Failed to process {f}: {e}")

    if not results:
        print("[WARN] No valid results loaded.")
        return

    df = pd.concat(results, ignore_index=True)

    # Only select columns that exist
    keep_cols = [c for c in ["model_version", "total_pnl", "win_rate", "max_drawdown", "sharpe", "sortino"] if c in df.columns]
    df = df[keep_cols]

    print("\n=== Model Comparison Summary ===")
    print(df)

    summary_path = os.path.join(base_dir, "comparison_summary.csv")
    df.to_csv(summary_path, index=False)
    print(f"\n[INFO] Saved comparison summary to {summary_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--compare-retrain", action="store_true", help="Compare retrained vs baseline model performance")
    args = parser.parse_args()

    if args.compare_retrain:
        compare_retrain_vs_baseline()
        return

    all_results = []

    for ema_short, ema_long, rsi, vol in itertools.product(EMA_SHORTS, EMA_LONGS, RSI_PERIODS, VOLATILITIES):
        try:
            res = run_single_backtest(ema_short, ema_long, rsi, vol)
            all_results.append(res)
        except Exception as e:
            logger.error(f"Backtest failed for config {ema_short},{ema_long},{rsi},{vol}: {e}")

    aggregate_results(all_results)


if __name__ == "__main__":
    main()
