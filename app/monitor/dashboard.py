# File: app/monitor/dashboard.py
"""
Streamlit dashboard to visualize backtest & forward/paper results.
Auto-loads:
 - data/backtest_*.pkl      (joblib/pickled dict from backtests)
 - data/forward_<symbol>.csv
 - data/db_export_<symbol>.csv

Usage:
  # from project root (with .venv activated)
  streamlit run app/monitor/dashboard.py -- --symbol MES
"""
import streamlit as st
import os
import glob
import pandas as pd
import joblib
from datetime import datetime

st.set_page_config(page_title="Trading-AI Dashboard", layout="wide")

# CLI-like arg via Streamlit -> fallback to "MES"
import sys
_symbol = "MES"
if "--symbol" in sys.argv:
    try:
        _symbol = sys.argv[sys.argv.index("--symbol") + 1]
    except Exception:
        pass

DATA_DIR = os.path.abspath(os.path.join(os.getcwd(), "data"))
os.makedirs(DATA_DIR, exist_ok=True)

st.title("Trading-AI: Backtest & Forward Dashboard")
st.sidebar.header("Settings")
symbol = st.sidebar.text_input("Symbol", value=_symbol)
st.sidebar.markdown("Files auto-loaded from `data/` folder")

# Helper: load latest backtest pickle(s)
def load_backtests(symbol: str):
    pattern = os.path.join(DATA_DIR, f"multi_backtests", f"*{symbol}*.pkl")
    # fallback pattern
    if not glob.glob(pattern):
        pattern = os.path.join(DATA_DIR, f"backtest_{symbol}.pkl")
    files = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
    results = []
    for f in files:
        try:
            res = joblib.load(f)
            # normalize some fields for display
            res_meta = {
                "file": os.path.basename(f),
                "timestamp": datetime.fromtimestamp(os.path.getmtime(f)).isoformat(),
                "win_rate": res.get("win_rate"),
                "max_drawdown": res.get("max_drawdown"),
                "total_pnl": res.get("equity_curve")[-1] if res.get("equity_curve") else None,
                "trades": len(res.get("trades", [])),
            }
            results.append((f, res, res_meta))
        except Exception as e:
            st.sidebar.error(f"Failed to load backtest file {f}: {e}")
    return results

# Helper: load forward CSV and DB export
def load_forward(symbol: str):
    forward_file = os.path.join(DATA_DIR, f"forward_{symbol}.csv")
    db_file = os.path.join(DATA_DIR, f"db_export_{symbol}.csv")
    forward = pd.read_csv(forward_file, parse_dates=["timestamp"]) if os.path.exists(forward_file) else None
    db = pd.read_csv(db_file, parse_dates=["timestamp"]) if os.path.exists(db_file) else None
    return forward, db

# Load data
backtests = load_backtests(symbol)
forward_df, db_df = load_forward(symbol)

# Top-level summary
col1, col2 = st.columns([2, 1])
with col1:
    st.header(f"Summary for {symbol}")
    st.write(f"Data directory: `{DATA_DIR}`")
    st.markdown(f"**Backtests found:** {len(backtests)}")
    st.markdown(f"**Forward CSV:** {'Yes' if forward_df is not None else 'No'}")
    st.markdown(f"**DB export:** {'Yes' if db_df is not None else 'No'}")

with col2:
    if backtests:
        latest_meta = backtests[0][2]
        st.metric("Latest backtest", latest_meta["file"])
        st.metric("Win rate", latest_meta["win_rate"])
        st.metric("Trades", latest_meta["trades"])
    else:
        st.info("No backtests found.")

# Section: Backtest selector + equity curve
if backtests:
    st.subheader("Backtest list")
    # show a compact table of metadata
    meta_rows = [m for (_, _, m) in backtests]
    meta_df = pd.DataFrame(meta_rows)
    st.dataframe(meta_df[["file", "timestamp", "win_rate", "max_drawdown", "total_pnl", "trades"]].fillna(""))

    sel = st.selectbox("Select backtest to visualize", [m["file"] for (_, _, m) in backtests])
    sel_idx = next(i for i, t in enumerate(backtests) if t[2]["file"] == sel)
    _, sel_res, _ = backtests[sel_idx]

    st.markdown("**Equity Curve (backtest)**")
    if sel_res.get("equity_curve") is not None:
        eq = sel_res["equity_curve"]
        eq_df = pd.DataFrame({"equity": eq})
        st.line_chart(eq_df["equity"])
    else:
        st.write("No equity curve in selected backtest.")

    st.markdown("**Trades (sample)**")
    trades = sel_res.get("trades", [])
    if trades:
        trades_df = pd.DataFrame(trades)
        st.dataframe(trades_df.head(200))
    else:
        st.write("No trades recorded in this backtest.")

else:
    st.info("No backtests available to show charts.")

# Section: Forward run + DB export
st.subheader("Forward / Paper Results")
if forward_df is not None:
    st.markdown("**Forward CSV (latest)**")
    st.dataframe(forward_df.head(500))
    if "pnl" in forward_df.columns:
        st.line_chart(forward_df["pnl"].cumsum().rename("cumulative_pnl"))
    st.download_button("Download forward CSV", forward_df.to_csv(index=False), file_name=f"forward_{symbol}.csv")
else:
    st.warning("No forward CSV found (forward_{symbol}.csv)")

if db_df is not None:
    st.markdown("**DB Export**")
    st.dataframe(db_df.head(500))
    st.download_button("Download DB export CSV", db_df.to_csv(index=False), file_name=f"db_export_{symbol}.csv")
else:
    st.info("No DB export file found (db_export_{symbol}.csv)")

# Small diagnostics & quick checks
st.subheader("Quick Diagnostics")
diag_col1, diag_col2 = st.columns(2)

with diag_col1:
    st.markdown("**Basic Checks**")
    st.write(f"Backtests files: {len(backtests)}")
    st.write(f"Forward CSV present: {forward_df is not None}")
    st.write(f"DB export present: {db_df is not None}")

with diag_col2:
    st.markdown("**Recent metrics**")
    if backtests:
        win_rates = [m["win_rate"] for (_, _, m) in backtests if m["win_rate"] is not None]
        if win_rates:
            st.write("Average win rate (backtests):", round(float(pd.Series(win_rates).mean()), 4))
    if forward_df is not None and "pnl" in forward_df.columns:
        st.write("Forward total PnL:", float(forward_df["pnl"].sum()))

st.markdown("---")
st.caption(f"Last updated: {datetime.utcnow().isoformat()} (UTC)")

# Validation Results Section
st.subheader("🔬 Validation Results")

# Walk-forward results
wf_files = glob.glob("data/validation/walk_forward_*.json")
if wf_files:
    latest_wf = max(wf_files, key=os.path.getmtime)
    with open(latest_wf, 'r') as f:
        wf_data = json.load(f)
    
    st.metric("Walk-Forward Splits", wf_data['n_splits'])
    
    if 'aggregated_metrics' in wf_data:
        agg = wf_data['aggregated_metrics']
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Avg Accuracy", f"{agg.get('accuracy', {}).get('mean', 0):.2%}")
        with col2:
            st.metric("Stability", agg.get('stability', {}).get('accuracy', 'N/A'))

# Monte Carlo results
mc_files = glob.glob("data/validation/monte_carlo_*.json")
if mc_files:
    latest_mc = max(mc_files, key=os.path.getmtime)
    with open(latest_mc, 'r') as f:
        mc_data = json.load(f)
    
    st.metric("MC Sequences", mc_data['n_sequences'])
    
    if 'summary_metrics' in mc_data:
        summary = mc_data['summary_metrics']
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Expected Sharpe", f"{summary['sharpe']['mean']:.2f}")
        with col2:
            st.metric("Expected Max DD", f"{summary['max_drawdown']['mean']:.2f}")
        with col3:
            st.metric("Sharpe Stability", summary.get('stability_assessment', {}).get('sharpe_stability', 'N/A'))