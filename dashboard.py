# File: dashboard.py
import streamlit as st
import pandas as pd
from app.data.live_feed import get_latest_bar, get_last_n_bars
from app.ml.model_manager import load_metrics_history, get_best_model

st.set_page_config(page_title="Trading AI Dashboard", layout="wide")

st.title("📊 Trading AI Dashboard")

# --- Live Market Feed ---
st.header("Live MES Futures Feed")
latest = get_latest_bar()
if not latest.empty:
    st.write(f"**Latest Bar:** {latest['timestamp'].iloc[0]}")
    st.write(f"Close: {latest['close'].iloc[0]}, Volume: {latest['volume'].iloc[0]}")

    hist = get_last_n_bars(100)
    st.line_chart(hist.set_index("timestamp")["close"])
else:
    st.warning("No live data yet.")

# --- Model Info ---
st.header("Current Model")
model, label = get_best_model()
if model:
    st.success(f"Using model: {label}")
else:
    st.error("No model available yet.")

# --- Metrics History ---
st.header("Performance History")
metrics_df = load_metrics_history()
if metrics_df is not None and not metrics_df.empty:
    st.dataframe(metrics_df.tail(10))
    st.line_chart(metrics_df.set_index("date")[["accuracy", "sharpe"]])
else:
    st.warning("No metrics logged yet.")
