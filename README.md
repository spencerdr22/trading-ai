# Trading-AI Futures System

A modular **futures trading research system** for simulation → backtest → forward/paper → live (Tradovate).

> **IMPORTANT:**  
> THIS PROJECT SIMULATES TRADING. A HIGH OR 100% WIN RATE IS NOT GUARANTEED IN REAL MARKETS.  
> The code attempts to optimize models and strategy parameters toward high win rates using online learning,  
> but it includes transaction costs, slippage, walk-forward testing, and kill-switches to prevent over-optimistic behavior.  
> Treat all results as simulations until you validate via paper/live forward tests.

---

## 📦 Features
- **Simulation**: Random-walk OHLCV generator.
- **Backtesting**: Walk-forward analysis, metrics, slippage/fees modeled.
- **ML Models**: scikit-learn (default), PyTorch optional.
- **Strategy Engine**: Adaptive thresholds & risk controls.
- **Executors**: Simulator, paper, Tradovate placeholders.
- **Monitoring**: Equity, PnL, Sharpe, drawdown.
- **Safety**: Kill-switch on large drawdown/loss.

---

## ⚡ Quickstart (Windows 11 Pro)

### 1. Clone & enter project
```powershell
git clone https://github.com/yourusername/trading-ai.git
cd trading-ai
