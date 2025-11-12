# File: app/analysis/view_forward.py
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def load_forward_results(symbol: str = "MES"):
    """
    Loads the forward trading results from CSV or PKL file.
    """
    data_dir = os.path.join(os.getcwd(), "data")
    csv_path = os.path.join(data_dir, f"forward_{symbol}.csv")
    pkl_path = os.path.join(data_dir, f"forward_{symbol}.pkl")

    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    elif os.path.exists(pkl_path):
        df = pd.read_pickle(pkl_path)
    else:
        raise FileNotFoundError(f"No forward results found for symbol {symbol} in {data_dir}")

    return df


def compute_forward_stats(df: pd.DataFrame):
    """
    Compute win rate, total/avg pnl, and drawdown metrics.
    """
    if df.empty:
        return {"trades": 0, "win_rate": 0.0, "total_pnl": 0.0, "max_drawdown": 0.0}

    wins = df[df["pnl"] > 0]
    losses = df[df["pnl"] <= 0]

    total_pnl = df["pnl"].sum()
    avg_pnl = df["pnl"].mean()
    win_rate = len(wins) / len(df)
    cumulative_pnl = df["pnl"].cumsum()
    rolling_max = cumulative_pnl.cummax()
    drawdown = cumulative_pnl - rolling_max
    max_dd = drawdown.min()

    return {
        "trades": len(df),
        "win_rate": round(win_rate, 4),
        "avg_pnl": round(avg_pnl, 4),
        "total_pnl": round(total_pnl, 4),
        "max_drawdown": round(max_dd, 4)
    }


def plot_forward_results(df: pd.DataFrame, symbol: str = "MES"):
    """
    Generate a visual summary of forward results: equity curve, PnL histogram, and trade counts.
    """
    os.makedirs(os.path.join("data", "plots"), exist_ok=True)

    stats = compute_forward_stats(df)
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    plt.subplots_adjust(hspace=0.4)

    # --- Equity Curve ---
    df["equity"] = df["pnl"].cumsum()
    axes[0].plot(df["timestamp"], df["equity"], color="blue", linewidth=2)
    axes[0].set_title(f"Forward Equity Curve — {symbol}")
    axes[0].set_ylabel("Cumulative PnL ($)")
    axes[0].grid(True)

    # --- PnL Distribution ---
    sns.histplot(df["pnl"], bins=40, ax=axes[1], kde=True, color="orange")
    axes[1].set_title("PnL Distribution per Trade")
    axes[1].set_xlabel("PnL ($)")
    axes[1].set_ylabel("Count")

    # --- Trade Count by Side ---
    if "side" in df.columns:
        side_counts = df["side"].value_counts()
        axes[2].bar(side_counts.index, side_counts.values, color=["green", "red", "gray"])
        axes[2].set_title("Trade Count by Side")
        axes[2].set_xlabel("Side")
        axes[2].set_ylabel("Trades")

    # --- Annotate Stats ---
    summary_text = (
        f"Trades: {stats['trades']}\n"
        f"Win Rate: {stats['win_rate']*100:.2f}%\n"
        f"Total PnL: {stats['total_pnl']:.2f}\n"
        f"Avg PnL: {stats['avg_pnl']:.4f}\n"
        f"Max Drawdown: {stats['max_drawdown']:.2f}"
    )
    plt.figtext(0.02, 0.02, summary_text, fontsize=10, bbox=dict(facecolor='white', alpha=0.6))

    out_path = os.path.join("data", "plots", f"forward_{symbol}_summary.png")
    plt.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[INFO] Forward summary plot saved to {out_path}")
    print(f"[INFO] Stats: {summary_text.replace(chr(10), ' | ')}")


def main(symbol="MES"):
    try:
        df = load_forward_results(symbol)
        stats = compute_forward_stats(df)
        print(f"\n=== Forward Results Summary for {symbol} ===")
        for k, v in stats.items():
            print(f"{k:>12}: {v}")
        plot_forward_results(df, symbol)
    except Exception as e:
        print(f"[ERROR] {e}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="MES")
    args = parser.parse_args()
    main(args.symbol)
