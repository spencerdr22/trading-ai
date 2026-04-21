"""
dashboard.py — Trading-AI Real-Time Dashboard
"""

import os
import sys
import re
import json
import glob
import time
from datetime import datetime, timezone, timedelta
from collections import deque

import joblib
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Trading-AI Live",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── ROOT / ENV SETUP ──────────────────────────────────────────────────────────
ROOT     = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR = os.path.join(ROOT, "data")
LOG_DIR  = os.path.join(DATA_DIR, "logs")

os.makedirs(DATA_DIR, exist_ok=True)
load_dotenv(os.path.join(ROOT, ".env"))

# ── Sidebar controls ──────────────────────────────────────────────────────────
st.sidebar.title("Trading-AI")
st.sidebar.caption("Real-time paper trading monitor")

refresh_sec = st.sidebar.selectbox(
    "Auto-refresh", [0, 15, 30, 60], index=2,
    format_func=lambda x: "Off" if x == 0 else f"Every {x}s"
)

symbol = st.sidebar.text_input("Symbol", value="MES")
log_lines = st.sidebar.slider("Log lines to show", 20, 200, 50, step=10)

if st.sidebar.button("Force refresh now"):
    st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# DATA FETCHERS
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=15)
def get_account() -> dict:
    import requests
    key    = os.getenv("ALPACA_API_KEY", "")
    secret = os.getenv("ALPACA_SECRET_KEY", "")
    base   = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

    if not key or not secret:
        return {}

    try:
        r = requests.get(
            f"{base}/v2/account",
            headers={"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret},
            timeout=5,
        )
        return r.json() if r.status_code == 200 else {}
    except Exception as e:
        st.error(f"Account API error: {e}")
        return {}

@st.cache_data(ttl=15)
def get_positions() -> list:
    import requests
    key = os.getenv("ALPACA_API_KEY", "")
    secret = os.getenv("ALPACA_SECRET_KEY", "")
    base = os.getenv("ALPACA_BASE_URL", "")

    if not key or not secret:
        return []

    try:
        r = requests.get(
            f"{base}/v2/positions",
            headers={"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret},
            timeout=5,
        )
        return r.json() if r.status_code == 200 else []
    except Exception as e:
        st.error(f"Positions API error: {e}")
        return []

@st.cache_data(ttl=15)
def get_orders(limit: int = 20) -> list:
    import requests
    key = os.getenv("ALPACA_API_KEY", "")
    secret = os.getenv("ALPACA_SECRET_KEY", "")
    base = os.getenv("ALPACA_BASE_URL", "")

    if not key or not secret:
        return []

    try:
        r = requests.get(
            f"{base}/v2/orders",
            headers={"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret},
            params={"status": "all", "limit": limit},
            timeout=5,
        )
        return r.json() if r.status_code == 200 else []
    except Exception as e:
        st.error(f"Orders API error: {e}")
        return []

@st.cache_data(ttl=15)
def get_latest_spy_bar() -> dict:
    import requests
    key = os.getenv("ALPACA_API_KEY", "")
    secret = os.getenv("ALPACA_SECRET_KEY", "")
    data = os.getenv("ALPACA_DATA_URL", "https://data.alpaca.markets")

    if not key or not secret:
        return {}

    try:
        r = requests.get(
            f"{data}/v2/stocks/SPY/bars/latest",
            headers={"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret},
            timeout=5,
        )
        if r.status_code == 200:
            b = r.json().get("bar", {})
            return {
                "close": b.get("c", 0),
                "volume": b.get("v", 0),
                "timestamp": b.get("t", "")
            }
    except Exception as e:
        st.error(f"Market data error: {e}")

    return {}

@st.cache_data(ttl=10)
def parse_alpaca_log(n_lines: int = 200) -> pd.DataFrame:
    log_path = os.path.join(LOG_DIR, "app_execution_alpaca_paper.log")
    if not os.path.exists(log_path):
        return pd.DataFrame()

    rows = []

    try:
        with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = list(deque(f, maxlen=n_lines * 3))
    except Exception as e:
        st.error(f"Log read error: {e}")
        return pd.DataFrame()

    signal_re = re.compile(
        r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*MES signal: (BUY|SELL|HOLD) @ ([\d.]+).*current_pos=(\w+)"
    )

    for line in lines:
        m = signal_re.search(line)
        if m:
            rows.append({
                "time": pd.to_datetime(m.group(1)),
                "type": "signal",
                "side": m.group(2),
                "price": float(m.group(3)),
                "pos_after": m.group(4),
                "status": "generated",
            })

    return pd.DataFrame(rows).sort_values("time", ascending=False).head(n_lines)

def get_pipeline_metrics() -> pd.DataFrame:
    """Load recent pipeline metrics from DB. Not cached (engine not picklable)."""
    try:
        sys.path.insert(0, ROOT)
        from sqlalchemy import text
        from app.db.init import get_engine

        eng = get_engine()
        with eng.connect() as conn:
            rows = conn.execute(
                text("SELECT name, value, timestamp FROM metrics ORDER BY timestamp DESC LIMIT 20")
            ).fetchall()

        if rows:
            return pd.DataFrame(rows, columns=["metric", "value", "timestamp"])

    except Exception as e:
        st.warning(f"DB metrics unavailable: {e}")

    return pd.DataFrame()

# ══════════════════════════════════════════════════════════════════════════════
# UI HEADER
# ══════════════════════════════════════════════════════════════════════════════

now_str = datetime.now().strftime("%H:%M:%S")
st.title("Trading-AI Dashboard")
st.caption(f"Last loaded: {now_str}")

# ── AUTO REFRESH (FIXED) ─────────────────────────────────────────────────────
if refresh_sec:
    try:
        from streamlit_autorefresh import st_autorefresh
        st_autorefresh(interval=refresh_sec * 1000, key="refresh")
    except ImportError:
        # streamlit-autorefresh not installed — use meta-refresh fallback
        import time as _t
        st.caption(f"⏱ Auto-refresh every {refresh_sec}s (install streamlit-autorefresh for smoother refresh)")
        _t.sleep(0)

# ══════════════════════════════════════════════════════════════════════════════
# ROW 1 — Account + Live Market
# ══════════════════════════════════════════════════════════════════════════════

col_acct, col_mkt = st.columns([1, 1])

with col_acct:
    st.subheader("💰 Alpaca Account")
    acct = get_account()
    if acct:
        pv   = float(acct.get("portfolio_value", 0))
        cash = float(acct.get("cash", 0))
        bp   = float(acct.get("buying_power", 0))
        st.metric("Portfolio Value",  f"${pv:,.2f}")
        st.metric("Cash",             f"${cash:,.2f}")
        st.metric("Buying Power",     f"${bp:,.2f}")
        day_pl    = float(acct.get("equity", pv)) - float(acct.get("last_equity", pv))
        pct_pl    = day_pl / max(float(acct.get("last_equity", 1)), 1) * 100
        st.metric("Day P&L", f"${day_pl:+,.2f}", f"{pct_pl:+.2f}%")
    else:
        st.warning("Account data unavailable — check Alpaca API keys in .env")

with col_mkt:
    st.subheader("📊 SPY Live Bar (MES Proxy)")
    bar = get_latest_spy_bar()
    if bar:
        st.metric("SPY Close",  f"${bar.get('close', 0):.2f}")
        st.metric("Volume",     f"{int(bar.get('volume', 0)):,}")
        st.caption(f"Bar time: {bar.get('timestamp', 'N/A')}")
    else:
        st.info("No live bar yet — market may be closed or outside hours.")

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# ROW 2 — Open Positions + Recent Orders
# ══════════════════════════════════════════════════════════════════════════════

col_pos, col_ord = st.columns([1, 1])

with col_pos:
    st.subheader("📋 Open Positions")
    positions = get_positions()
    if positions:
        pos_df = pd.DataFrame(positions)[[
            "symbol", "qty", "side",
            "avg_entry_price", "current_price",
            "unrealized_pl", "unrealized_plpc"
        ]].copy()
        pos_df["unrealized_plpc"] = pos_df["unrealized_plpc"].apply(
            lambda x: f"{float(x)*100:.2f}%"
        )
        st.dataframe(pos_df, width='stretch')
    else:
        st.info("No open positions (flat).")

with col_ord:
    st.subheader("🗒 Recent Orders")
    orders = get_orders(20)
    if orders:
        ord_df = pd.DataFrame(orders)[[
            "submitted_at", "symbol", "side",
            "qty", "type", "status", "filled_avg_price"
        ]].copy()
        ord_df["submitted_at"] = pd.to_datetime(
            ord_df["submitted_at"]
        ).dt.strftime("%H:%M:%S")
        st.dataframe(ord_df, width='stretch')
    else:
        st.info("No recent orders found.")

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# ROW 3 — Signal history + Equity chart
# ══════════════════════════════════════════════════════════════════════════════

col_sig, col_eq = st.columns([1, 1])

with col_sig:
    st.subheader("🔔 Signal History (from log)")
    sig_df = parse_alpaca_log(log_lines)
    if not sig_df.empty:
        # Colour code by side — use map() (applymap removed in pandas 2.1+)
        def _colour_side(val):
            if val == "BUY":  return "background-color: #d4edda"
            if val == "SELL": return "background-color: #f8d7da"
            return ""
        styled = sig_df.style.map(_colour_side, subset=["side"])
        st.dataframe(styled, width='stretch')
    else:
        st.info("No signal log entries found yet.")

with col_eq:
    st.subheader("📈 Cumulative P&L (forward_MES.csv)")
    fwd_path = os.path.join(DATA_DIR, f"forward_{symbol}.csv")
    if os.path.exists(fwd_path):
        try:
            fwd_df = pd.read_csv(fwd_path, parse_dates=["timestamp"])
            if "pnl" in fwd_df.columns and not fwd_df.empty:
                fwd_df["cumulative_pnl"] = fwd_df["pnl"].cumsum()
                st.line_chart(fwd_df.set_index("timestamp")["cumulative_pnl"])
                total  = fwd_df["pnl"].sum()
                wins   = (fwd_df["pnl"] > 0).sum()
                total_t= len(fwd_df)
                wr     = wins / max(total_t, 1) * 100
                c1, c2, c3 = st.columns(3)
                c1.metric("Total P&L",    f"${total:+.2f}")
                c2.metric("Win Rate",     f"{wr:.1f}%")
                c3.metric("Trade Count",  str(total_t))
            else:
                st.info("forward_MES.csv has no pnl column yet.")
        except Exception as e:
            st.error(f"Could not load forward results: {e}")
    else:
        st.info("No forward results file yet — start trading first.")

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# ROW 4 — Pipeline Metrics + Log Tail
# ══════════════════════════════════════════════════════════════════════════════

col_pipe, col_log = st.columns([1, 1])

with col_pipe:
    st.subheader("🔧 Pipeline Metrics (DB)")
    pm_df = get_pipeline_metrics()
    if not pm_df.empty:
        st.dataframe(pm_df, width='stretch')
    else:
        st.info("No pipeline metrics yet — run the trading scheduler first.")

with col_log:
    st.subheader("📜 Trading Log (tail)")
    log_path = os.path.join(ROOT, "logs", "trading_ai.log")
    if os.path.exists(log_path):
        try:
            with open(log_path, "r", encoding="utf-8", errors="replace") as f:
                lines = list(deque(f, maxlen=log_lines))
            # Colour ERROR / WARNING lines
            coloured = []
            for ln in reversed(lines):
                ln = ln.rstrip()
                if "ERROR" in ln:
                    coloured.append(f":red[{ln}]")
                elif "WARNING" in ln or "WARN" in ln:
                    coloured.append(f":orange[{ln}]")
                else:
                    coloured.append(ln)
            st.code("\n".join(coloured), language="")
        except Exception as e:
            st.error(f"Log read error: {e}")
    else:
        st.info("Log file not found yet.")

st.divider()
st.caption(f"Trading-AI Dashboard | Last refreshed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")