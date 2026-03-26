"""
dashboard.py — Trading-AI Real-Time Dashboard

Tracks live trading activity, account status, signal history,
sentiment, model performance, and system health.

Run with:
    streamlit run app/monitor/dashboard.py

Auto-refreshes every 30 seconds during market hours.
"""

import os
import sys
import re
import json
import glob
import time
from datetime import datetime, timezone, timedelta

import joblib
import pandas as pd
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Trading-AI Live",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT     = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR = os.path.join(ROOT, "data")
LOG_DIR  = os.path.join(DATA_DIR, "logs")
os.makedirs(DATA_DIR, exist_ok=True)

# ── Sidebar controls ──────────────────────────────────────────────────────────
st.sidebar.title("Trading-AI")
st.sidebar.caption("Real-time paper trading monitor")

refresh_sec = st.sidebar.selectbox(
    "Auto-refresh", [0, 15, 30, 60], index=2,
    format_func=lambda x: "Off" if x == 0 else f"Every {x}s"
)
if refresh_sec:
    st.sidebar.caption(f"Next refresh in ~{refresh_sec}s")

symbol = st.sidebar.text_input("Symbol", value="MES")
log_lines = st.sidebar.slider("Log lines to show", 20, 200, 50, step=10)

st.sidebar.markdown("---")
if st.sidebar.button("Force refresh now"):
    st.experimental_rerun()


# ══════════════════════════════════════════════════════════════════════════════
# DATA FETCHERS
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=15)
def get_account() -> dict:
    """Live Alpaca account via REST — cached 15s."""
    import os, requests
    from dotenv import load_dotenv
    load_dotenv(os.path.join(ROOT, ".env"))
    key    = os.getenv("ALPACA_API_KEY", "")
    secret = os.getenv("ALPACA_SECRET_KEY", "")
    base   = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
    if not key or not secret:
        return {}
    try:
        r = requests.get(
            f"{base}/v2/account",
            headers={"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret},
            timeout=8,
        )
        return r.json() if r.status_code == 200 else {}
    except Exception:
        return {}


@st.cache_data(ttl=15)
def get_positions() -> list:
    """Live Alpaca positions — cached 15s."""
    import os, requests
    from dotenv import load_dotenv
    load_dotenv(os.path.join(ROOT, ".env"))
    key    = os.getenv("ALPACA_API_KEY", "")
    secret = os.getenv("ALPACA_SECRET_KEY", "")
    base   = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
    if not key or not secret:
        return []
    try:
        r = requests.get(
            f"{base}/v2/positions",
            headers={"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret},
            timeout=8,
        )
        return r.json() if r.status_code == 200 else []
    except Exception:
        return []


@st.cache_data(ttl=15)
def get_orders(limit: int = 20) -> list:
    """Recent Alpaca orders — cached 15s."""
    import os, requests
    from dotenv import load_dotenv
    load_dotenv(os.path.join(ROOT, ".env"))
    key    = os.getenv("ALPACA_API_KEY", "")
    secret = os.getenv("ALPACA_SECRET_KEY", "")
    base   = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
    if not key or not secret:
        return []
    try:
        r = requests.get(
            f"{base}/v2/orders",
            headers={"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret},
            params={"status": "all", "limit": limit},
            timeout=8,
        )
        return r.json() if r.status_code == 200 else []
    except Exception:
        return []


@st.cache_data(ttl=15)
def get_latest_spy_bar() -> dict:
    """Latest SPY 1-min bar from Alpaca data — cached 15s."""
    import os, requests
    from dotenv import load_dotenv
    load_dotenv(os.path.join(ROOT, ".env"))
    key    = os.getenv("ALPACA_API_KEY", "")
    secret = os.getenv("ALPACA_SECRET_KEY", "")
    data   = os.getenv("ALPACA_DATA_URL", "https://data.alpaca.markets")
    if not key or not secret:
        return {}
    try:
        r = requests.get(
            f"{data}/v2/stocks/SPY/bars/latest",
            headers={"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret},
            timeout=8,
        )
        if r.status_code == 200:
            b = r.json().get("bar", {})
            return {"close": b.get("c", 0), "volume": b.get("v", 0),
                    "timestamp": b.get("t", "")}
        return {}
    except Exception:
        return {}


@st.cache_data(ttl=10)
def parse_alpaca_log(n_lines: int = 200) -> pd.DataFrame:
    """
    Parse the Alpaca execution log into a DataFrame of signals/orders.
    Returns columns: time, type, side, price, status, order_id
    """
    log_path = os.path.join(LOG_DIR, "app_execution_alpaca_paper.log")
    if not os.path.exists(log_path):
        return pd.DataFrame()

    rows = []
    signal_re = re.compile(
        r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*"
        r"MES signal: (BUY|SELL|HOLD) @ ([\d.]+).*current_pos=(\w+)"
    )
    order_re = re.compile(
        r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*"
        r"Order placed: (BUY|SELL) SPY x(\d+) \| id=([\w-]+) \| local_state=(\w+)"
    )
    blocked_re = re.compile(
        r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*"
        r"Already (long|short) SPY"
    )

    try:
        with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()[-n_lines * 3:]
    except Exception:
        return pd.DataFrame()

    for line in lines:
        m = signal_re.search(line)
        if m:
            rows.append({
                "time":     m.group(1),
                "type":     "signal",
                "side":     m.group(2),
                "price":    float(m.group(3)),
                "pos_after": m.group(4),
                "status":   "generated",
                "order_id": "",
            })
            continue
        m = order_re.search(line)
        if m:
            rows.append({
                "time":     m.group(1),
                "type":     "order",
                "side":     m.group(2),
                "price":    0.0,
                "pos_after": m.group(5),
                "status":   "filled",
                "order_id": m.group(4)[:8],
            })
            continue

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time", ascending=False)
    return df.head(n_lines)


@st.cache_data(ttl=10)
def parse_main_log(n_lines: int = 100) -> list:
    """Return last N lines of the main trading log."""
    log_path = os.path.join(LOG_DIR, "__main__.log")
    if not os.path.exists(log_path):
        return []
    try:
        with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
        return [l.rstrip() for l in lines[-n_lines:]]
    except Exception:
        return []


@st.cache_data(ttl=30)
def parse_sentiment_history() -> pd.DataFrame:
    """Extract sentiment scores from main log."""
    log_path = os.path.join(LOG_DIR, "__main__.log")
    if not os.path.exists(log_path):
        return pd.DataFrame()

    pat = re.compile(
        r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*"
        r"Sentiment: (\w+) \(score=([-\d.]+)"
    )
    rows = []
    try:
        with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                m = pat.search(line)
                if m:
                    rows.append({
                        "time":  pd.to_datetime(m.group(1)),
                        "label": m.group(2),
                        "score": float(m.group(3)),
                    })
    except Exception:
        pass

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).tail(50)


@st.cache_data(ttl=60)
def get_pipeline_metrics() -> pd.DataFrame:
    """Pull pipeline training metrics from SQLite DB."""
    try:
        from sqlalchemy import text
        sys.path.insert(0, ROOT)
        from app.db.init import get_engine
        eng = get_engine()
        with eng.connect() as conn:
            rows = conn.execute(
                text("SELECT name, value, timestamp FROM metrics "
                     "WHERE name LIKE 'pipeline_%' "
                     "ORDER BY timestamp DESC LIMIT 20")
            ).fetchall()
        if rows:
            return pd.DataFrame(rows, columns=["metric", "value", "timestamp"])
    except Exception:
        pass
    return pd.DataFrame()


def is_market_hours() -> bool:
    now_et = datetime.now(timezone(timedelta(hours=-4)))  # ET (EDT)
    if now_et.weekday() >= 5:
        return False
    return time_to_minutes(now_et) >= 570 and time_to_minutes(now_et) <= 955


def time_to_minutes(dt) -> int:
    return dt.hour * 60 + dt.minute


# ══════════════════════════════════════════════════════════════════════════════
# LAYOUT
# ══════════════════════════════════════════════════════════════════════════════

now_str = datetime.now().strftime("%H:%M:%S")
market_open = is_market_hours()
status_color = "🟢" if market_open else "🔴"
st.title(f"Trading-AI  {status_color}  {'LIVE' if market_open else 'CLOSED'}")
st.caption(f"Last loaded: {now_str}  ·  Symbol: {symbol} via SPY proxy  ·  Paper account")

# ── ROW 1: Account metrics ────────────────────────────────────────────────────
acct    = get_account()
pos     = get_positions()
spy_bar = get_latest_spy_bar()

portfolio = float(acct.get("portfolio_value", 0))
cash      = float(acct.get("cash", 0))
bp        = float(acct.get("buying_value", acct.get("buying_power", 0)))
pnl_total = portfolio - 100_000.0

spy_pos   = next((p for p in pos if p.get("symbol") == "SPY"), None)
unr_pnl   = float(spy_pos.get("unrealized_pl", 0)) if spy_pos else 0.0
spy_qty   = int(float(spy_pos.get("qty", 0))) if spy_pos else 0
spy_side  = spy_pos.get("side", "flat") if spy_pos else "flat"
spy_price = float(spy_bar.get("close", 0))
spy_ts    = spy_bar.get("timestamp", "")[:16].replace("T", " ")

col1, col2, col3, col4, col5, col6 = st.columns(6)
col1.metric("Portfolio",     f"${portfolio:,.2f}")
col2.metric("Total P&L",     f"${pnl_total:+.2f}",
            delta=f"${pnl_total:+.2f}",
            delta_color="normal")
col3.metric("Unrealized P&L", f"${unr_pnl:+.2f}",
            delta=f"${unr_pnl:+.2f}",
            delta_color="normal")
col4.metric("SPY price",     f"${spy_price:.2f}" if spy_price else "—",
            help=f"Last bar: {spy_ts}")
col5.metric("Position",      f"{spy_side.upper()}  {spy_qty} share{'s' if abs(spy_qty)!=1 else ''}",
            help="Current SPY position")
col6.metric("Buying Power",  f"${float(acct.get('buying_power',0)):,.0f}")

st.markdown("---")

# ── ROW 2: Live signals + Recent orders ──────────────────────────────────────
left, right = st.columns([3, 2])

with left:
    st.subheader("Live signal feed")
    sig_df = parse_alpaca_log(n_lines=log_lines)

    if not sig_df.empty:
        # Only signals (not internal order records)
        signals = sig_df[sig_df["type"] == "signal"].copy()

        if not signals.empty:
            display = signals[["time", "side", "price", "pos_after", "status"]].copy()
            display["time"] = display["time"].dt.strftime("%H:%M:%S")

            # Color-code BEFORE renaming columns so row["side"] still works
            def row_style(row):
                if row["side"] == "BUY":
                    return ["background-color: #eaf3de"] * len(row)
                elif row["side"] == "SELL":
                    return ["background-color: #fcebeb"] * len(row)
                return [""] * len(row)

            styled = display.style.apply(row_style, axis=1)
            display.columns = ["Time", "Signal", "Price", "Position after", "Status"]
            styled.columns = display.columns
            st.dataframe(
                display,
                use_container_width=True,
                height=350,
            )

            # Mini price chart from signals
            price_data = signals[signals["price"] > 0].sort_values("time")
            if len(price_data) > 1:
                chart_df = price_data.set_index("time")[["price"]].rename(
                    columns={"price": "SPY price"}
                )
                st.line_chart(chart_df, height=150)
        else:
            st.info("No signals in log yet today.")
    else:
        st.info("Alpaca log not found.")

with right:
    st.subheader("Recent orders (Alpaca)")
    orders = get_orders(limit=15)
    if orders:
        order_rows = []
        for o in orders:
            filled_at = o.get("filled_at") or o.get("submitted_at") or ""
            filled_at = filled_at[:16].replace("T", " ") if filled_at else "—"
            filled_px  = o.get("filled_avg_price")
            order_rows.append({
                "Time":   filled_at,
                "Side":   o.get("side", "").upper(),
                "Qty":    o.get("filled_qty") or o.get("qty"),
                "Price":  f"${float(filled_px):.2f}" if filled_px else "—",
                "Status": o.get("status", "").upper(),
            })
        order_df = pd.DataFrame(order_rows)
        st.dataframe(order_df, use_container_width=True, height=280)
    else:
        st.info("No orders found or Alpaca unavailable.")

    # Position card
    st.subheader("Current position")
    if spy_pos:
        avg_entry = float(spy_pos.get("avg_entry_price", 0))
        mkt_val   = float(spy_pos.get("market_value", 0))
        pnl_pct   = float(spy_pos.get("unrealized_plpc", 0)) * 100
        st.markdown(f"""
| Field | Value |
|-------|-------|
| Symbol | SPY |
| Side | {spy_side.upper()} |
| Qty | {spy_qty} share |
| Avg entry | ${avg_entry:.2f} |
| Market value | ${mkt_val:.2f} |
| Unrealized P&L | ${unr_pnl:+.2f} ({pnl_pct:+.2f}%) |
""")
    else:
        st.info("Flat — no open position.")

st.markdown("---")

# ── ROW 3: Sentiment + Model ──────────────────────────────────────────────────
s_col, m_col = st.columns([1, 1])

with s_col:
    st.subheader("Sentiment history (today)")
    sent_df = parse_sentiment_history()

    if not sent_df.empty:
        today = pd.Timestamp.now().date()
        today_sent = sent_df[sent_df["time"].dt.date == today]

        if not today_sent.empty:
            latest = today_sent.iloc[-1]
            score  = latest["score"]
            label  = latest["label"]
            color  = "🟢" if label == "BULLISH" else "🔴" if label == "BEARISH" else "🟡"
            st.metric(
                "Current sentiment",
                f"{color} {label}",
                delta=f"score {score:+.3f}",
                delta_color="normal",
            )

            chart_df = today_sent.set_index("time")[["score"]]
            st.line_chart(chart_df, height=160)

            # Distribution
            dist = today_sent["label"].value_counts()
            c1, c2, c3 = st.columns(3)
            c1.metric("Bullish reads",  dist.get("BULLISH", 0))
            c2.metric("Neutral reads",  dist.get("NEUTRAL", 0))
            c3.metric("Bearish reads",  dist.get("BEARISH", 0))
        else:
            st.info("No sentiment readings today yet.")
            if not sent_df.empty:
                st.caption("Recent (prior days):")
                st.line_chart(sent_df.set_index("time")[["score"]], height=120)
    else:
        st.info("No sentiment data in log.")

with m_col:
    st.subheader("Model & pipeline")
    pipe_df = get_pipeline_metrics()

    if not pipe_df.empty:
        acc_rows = pipe_df[pipe_df["metric"] == "pipeline_rf_accuracy"]
        if not acc_rows.empty:
            latest_acc = float(acc_rows.iloc[0]["value"])
            st.metric("Latest RF accuracy", f"{latest_acc:.2%}",
                      help="Trained on real SPY bars")

        st.dataframe(
            pipe_df[["metric", "value", "timestamp"]].head(10),
            use_container_width=True,
            height=200,
        )
    else:
        st.info("No pipeline metrics yet.")

    # Model file ages
    st.caption("Model files:")
    model_dir = os.path.join(DATA_DIR, "models")
    if os.path.exists(model_dir):
        model_files = sorted(
            glob.glob(os.path.join(model_dir, "*.pkl")) +
            glob.glob(os.path.join(model_dir, "*.pt")),
            key=os.path.getmtime, reverse=True
        )[:5]
        for mf in model_files:
            age_min = (time.time() - os.path.getmtime(mf)) / 60
            age_str = f"{int(age_min)}m ago" if age_min < 120 else f"{age_min/60:.1f}h ago"
            st.caption(f"  {os.path.basename(mf)}  ·  {age_str}")

st.markdown("---")

# ── ROW 4: Live log tail ─────────────────────────────────────────────────────
st.subheader("Live trading log")
log_lines_data = parse_main_log(n_lines=log_lines)

if log_lines_data:
    # Highlight different log levels
    log_text = "\n".join(log_lines_data[-log_lines:])
    st.code(log_text, language=None)
else:
    st.info("Main log not found.")

st.markdown("---")

# ── ROW 5: Intraday P&L from orders ──────────────────────────────────────────
st.subheader("Intraday order history")
if orders:
    filled = [o for o in orders if o.get("status") == "filled"]
    if filled:
        rows = []
        for o in filled:
            filled_at = o.get("filled_at", "")[:16].replace("T", " ")
            rows.append({
                "Time":   filled_at,
                "Side":   o.get("side", "").upper(),
                "Symbol": o.get("symbol", ""),
                "Qty":    o.get("filled_qty"),
                "Price":  float(o.get("filled_avg_price") or 0),
                "Status": o.get("status", "").upper(),
            })
        filled_df = pd.DataFrame(rows)
        st.dataframe(filled_df, use_container_width=True)

        # Simple P&L calc from pairs of filled orders
        filled_df["Price"] = pd.to_numeric(filled_df["Price"], errors="coerce")
        buys  = filled_df[filled_df["Side"] == "BUY"]["Price"].tolist()
        sells = filled_df[filled_df["Side"] == "SELL"]["Price"].tolist()
        if buys and sells:
            pairs = min(len(buys), len(sells))
            pnl   = sum(sells[:pairs]) - sum(buys[:pairs])
            st.metric("Estimated intraday P&L (filled pairs)",
                      f"${pnl:+.2f}",
                      delta=f"${pnl:+.2f}",
                      delta_color="normal")
    else:
        st.info("No filled orders today.")

st.markdown("---")
st.caption(
    f"Trading-AI Dashboard  ·  "
    f"Loaded {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ·  "
    f"Auto-refresh: {'every ' + str(refresh_sec) + 's' if refresh_sec else 'off'}"
)

# ── Auto-refresh ──────────────────────────────────────────────────────────────
if refresh_sec:
    time.sleep(refresh_sec)
    st.experimental_rerun()
