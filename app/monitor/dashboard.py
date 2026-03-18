"""
dashboard.py — Streamlit dashboard for Trading-AI

Shows backtest results, forward/paper trade history, live system
status, and validation metrics.

Run with:
    streamlit run app/monitor/dashboard.py -- --symbol MES
"""

import os
import sys
import glob
import json
import pandas as pd
import joblib
import streamlit as st
from datetime import datetime

st.set_page_config(page_title="Trading-AI Dashboard", layout="wide")

# ── Symbol from CLI arg ───────────────────────────────────────────────────────
_symbol = "MES"
if "--symbol" in sys.argv:
    try:
        _symbol = sys.argv[sys.argv.index("--symbol") + 1]
    except (IndexError, ValueError):
        pass

DATA_DIR = os.path.abspath(os.path.join(os.getcwd(), "data"))
os.makedirs(DATA_DIR, exist_ok=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.title("Trading-AI: MES Dashboard")
st.sidebar.header("Settings")
symbol = st.sidebar.text_input("Symbol", value=_symbol)
st.sidebar.markdown("Files auto-loaded from `data/`")
auto_refresh = st.sidebar.checkbox("Auto-refresh (30s)", value=False)
if auto_refresh:
    import time
    time.sleep(30)
    st.rerun()

# ── Data loaders ──────────────────────────────────────────────────────────────

def load_backtests(sym: str):
    pattern = os.path.join(DATA_DIR, "multi_backtests", f"*{sym}*.pkl")
    if not glob.glob(pattern):
        pattern = os.path.join(DATA_DIR, f"backtest_{sym}.pkl")
    files   = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
    results = []
    for f in files:
        try:
            res  = joblib.load(f)
            eq   = res.get("equity_curve") or []
            meta = {
                "file":         os.path.basename(f),
                "timestamp":    datetime.fromtimestamp(os.path.getmtime(f)).isoformat(),
                "win_rate":     res.get("win_rate"),
                "max_drawdown": res.get("max_drawdown"),
                "total_pnl":    eq[-1] if eq else None,
                "trades":       len(res.get("trades", [])),
            }
            results.append((f, res, meta))
        except Exception as e:
            st.sidebar.error(f"Could not load {os.path.basename(f)}: {e}")
    return results


def load_forward(sym: str):
    fwd_file = os.path.join(DATA_DIR, f"forward_{sym}.csv")
    db_file  = os.path.join(DATA_DIR, f"db_export_{sym}.csv")
    fwd = (pd.read_csv(fwd_file, parse_dates=["timestamp"])
           if os.path.exists(fwd_file) else None)
    db  = (pd.read_csv(db_file,  parse_dates=["timestamp"])
           if os.path.exists(db_file)  else None)
    return fwd, db


# ── Load data ─────────────────────────────────────────────────────────────────
backtests  = load_backtests(symbol)
forward_df, db_df = load_forward(symbol)

# ── Summary header ────────────────────────────────────────────────────────────
c1, c2 = st.columns([2, 1])
with c1:
    st.header(f"Symbol: {symbol}")
    st.write(f"Data dir: `{DATA_DIR}`")
    st.write(f"Backtests: {len(backtests)} | Forward CSV: {'yes' if forward_df is not None else 'no'} | DB export: {'yes' if db_df is not None else 'no'}")
with c2:
    if backtests:
        m = backtests[0][2]
        st.metric("Latest backtest",  m["file"])
        st.metric("Win rate",         f"{m['win_rate']:.2%}" if m["win_rate"] else "N/A")
        st.metric("Trades",           m["trades"])
    else:
        st.info("No backtests found.")

# ── Backtests ─────────────────────────────────────────────────────────────────
if backtests:
    st.subheader("Backtest History")
    meta_df = pd.DataFrame([m for (_, _, m) in backtests])
    st.dataframe(
        meta_df[["file", "timestamp", "win_rate", "max_drawdown", "total_pnl", "trades"]].fillna(""),
        use_container_width=True,
    )

    sel     = st.selectbox("Select backtest to view", [m["file"] for (_, _, m) in backtests])
    sel_idx = next(i for i, t in enumerate(backtests) if t[2]["file"] == sel)
    _, sel_res, _ = backtests[sel_idx]

    eq = sel_res.get("equity_curve")
    if eq:
        st.line_chart(pd.DataFrame({"equity": eq}))
    else:
        st.write("No equity curve available.")

    trades = sel_res.get("trades", [])
    if trades:
        st.dataframe(pd.DataFrame(trades).head(200), use_container_width=True)
    else:
        st.write("No trades in this backtest.")
else:
    st.info("No backtests available.")

# ── Forward / Paper results ───────────────────────────────────────────────────
st.subheader("Forward / Paper Results")

if forward_df is not None:
    st.markdown("**Forward CSV**")
    st.dataframe(forward_df.head(500), use_container_width=True)
    if "pnl" in forward_df.columns:
        st.line_chart(forward_df["pnl"].cumsum().rename("cumulative_pnl"))

    # Sentiment breakdown if column exists
    if "sentiment" in forward_df.columns:
        st.markdown("**Sentiment Distribution**")
        counts = forward_df["sentiment"].value_counts()
        st.bar_chart(counts)

    st.download_button(
        "Download forward CSV",
        forward_df.to_csv(index=False),
        file_name=f"forward_{symbol}.csv",
    )
else:
    st.warning(f"No forward CSV found: forward_{symbol}.csv")

if db_df is not None:
    st.markdown("**DB Export**")
    st.dataframe(db_df.head(500), use_container_width=True)
    st.download_button(
        "Download DB export",
        db_df.to_csv(index=False),
        file_name=f"db_export_{symbol}.csv",
    )
else:
    st.info("No DB export found.")

# ── Quick diagnostics ─────────────────────────────────────────────────────────
st.subheader("Quick Diagnostics")
d1, d2 = st.columns(2)
with d1:
    st.write(f"Backtest files: {len(backtests)}")
    st.write(f"Forward CSV: {forward_df is not None}")
    st.write(f"DB export: {db_df is not None}")
with d2:
    if backtests:
        rates = [m["win_rate"] for (_, _, m) in backtests if m["win_rate"] is not None]
        if rates:
            st.metric("Avg backtest win rate", f"{sum(rates)/len(rates):.2%}")
    if forward_df is not None and "pnl" in forward_df.columns:
        st.metric("Forward total PnL", f"${forward_df['pnl'].sum():.2f}")

# ── Live system status ────────────────────────────────────────────────────────
st.subheader("Live System Status")
try:
    from app.llm.monitor import get_performance_report
    report = get_performance_report()
    gpu    = report["gpu_scheduler"]
    syscfg = report["system_config"]
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Session",          syscfg["market_session"])
    s2.metric("GPU Mode",         syscfg["gpu_mode"])
    s3.metric("CPU",              f"{syscfg['cpu_load']}%")
    s4.metric("RAM",              f"{syscfg['memory_usage']['percent']:.0f}%")
    s1.metric("Trading calls",    gpu["trading_calls"])
    s2.metric("Analysis deferred",gpu["analysis_deferred"])
    s3.metric("Avg wait (ms)",    f"{gpu['avg_wait_time_ms']:.1f}")
    s4.metric("GPU state",        gpu["current_task"])
except Exception as e:
    st.info(f"System monitor unavailable: {e}")

# ── Validation results ────────────────────────────────────────────────────────
st.subheader("Validation Results")

wf_files = glob.glob(os.path.join(DATA_DIR, "validation", "walk_forward_*.json"))
if wf_files:
    with open(max(wf_files, key=os.path.getmtime)) as f:
        wf = json.load(f)
    st.markdown("**Walk-Forward**")
    st.metric("Splits", wf.get("n_splits", wf.get("n_splits", "?")))
    agg = wf.get("aggregated_metrics", {})
    if agg:
        v1, v2 = st.columns(2)
        acc_data = agg.get("accuracy", {})
        v1.metric("Avg Accuracy",  f"{acc_data.get('mean', 0):.2%}")
        v2.metric("Stability",     agg.get("stability", {}).get("accuracy", "N/A"))
else:
    st.info("No walk-forward results found.")

mc_files = glob.glob(os.path.join(DATA_DIR, "validation", "monte_carlo_*.json"))
if mc_files:
    with open(max(mc_files, key=os.path.getmtime)) as f:
        mc = json.load(f)
    st.markdown("**Monte Carlo**")
    st.metric("Sequences", mc.get("n_sequences"))
    sm = mc.get("summary_metrics", {})
    if sm:
        m1, m2, m3 = st.columns(3)
        m1.metric("Exp. Sharpe",   f"{sm.get('sharpe', {}).get('mean', 0):.2f}")
        m2.metric("Exp. Max DD",   f"{sm.get('max_drawdown', {}).get('mean', 0):.3f}")
        sa = mc.get("stability_assessment", {})
        m3.metric("Sharpe stable", sa.get("sharpe_stability", "N/A"))
else:
    st.info("No Monte Carlo results found.")

# ── Pipeline metrics from DB ──────────────────────────────────────────────────
st.subheader("Training Pipeline History")
try:
    from app.db.init import get_engine
    from sqlalchemy import text
    eng = get_engine()
    with eng.connect() as conn:
        rows = conn.execute(
            text("SELECT name, value, timestamp FROM metrics "
                 "WHERE name LIKE 'pipeline_%' "
                 "ORDER BY timestamp DESC LIMIT 50")
        ).fetchall()
    if rows:
        pipeline_df = pd.DataFrame(rows, columns=["metric", "value", "timestamp"])
        st.dataframe(pipeline_df, use_container_width=True)
    else:
        st.info("No pipeline metrics in DB yet.")
except Exception as e:
    st.info(f"DB metrics unavailable: {e}")

st.markdown("---")
st.caption(f"Last updated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
