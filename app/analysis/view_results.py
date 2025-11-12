import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

DATA_PATH = os.path.join(os.path.dirname(__file__), "..","..", "data", "backtest_MES.pkl")

def load_results():
    path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "backtest_MES.pkl")
    path = os.path.abspath(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Backtest file not found: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)

def summarize_results(res):
    """Print and visualize key metrics."""
    print("\n=== 📈 Backtest Summary ===")
    print(f"Win rate: {res.get('win_rate', 0):.2%}")
    print(f"Max drawdown: {res.get('max_drawdown', 0):.2%}")
    print(f"Total PnL: {res.get('total_pnl', 0):.2f}")
    print(f"Number of trades: {len(res.get('trades', []))}")

    trades = pd.DataFrame(res.get("trades", []))
    if trades.empty:
        print("No trades to display.")
        return

    # Plot equity curve
    trades["cum_pnl"] = trades["pnl"].cumsum()
    sns.set(style="darkgrid")
    plt.figure(figsize=(10, 5))
    plt.plot(trades["timestamp"], trades["cum_pnl"], label="Equity Curve", color="cyan")
    plt.title("Cumulative PnL Over Time")
    plt.xlabel("Time")
    plt.ylabel("PnL ($)")
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    res = load_results()
    summarize_results(res)
