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
# File: app/adaptive/learner.py
import torch
import torch.optim as optim
from app.adaptive.model_hub import ModelHub
from app.adaptive.policy_net import PolicyNet
from app.monitor.logger import get_logger
from app.db.init import get_engine
from sqlalchemy.orm import sessionmaker
from app.db.schema import TradeMetric
from app.adaptive.reward import compute_batch_reward
import datetime
logger = get_logger(__name__)
class ReinforcementLearner:
    """
    Offline reinforcement-learning engine using policy-gradient (REINFORCE).
    """

    def __init__(self, feature_dim, lr=1e-4, gamma=0.99,
                 model_name="adaptive_policy"):
        self.policy = PolicyNet(feature_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma
        self.model_name = model_name
        self.hub = ModelHub()
        self.engine = get_engine()

        logger.info(f"Adaptive RL initialized: feature_dim={feature_dim}, lr={lr}")
    # ------------------------------------------------------------
    # Offline Training Loop
    # ------------------------------------------------------------
    def train(self, n_epochs=10, batch_size=64):
        """
        Train the RL policy using historical trade data.
        """
        Session = sessionmaker(bind=self.engine)
        session = Session()

        for epoch in range(n_epochs):
            logger.info(f"RL Training Epoch {epoch+1}/{n_epochs}")

            # Load trade metrics from DB
            trade_metrics = session.query(TradeMetric).order_by(TradeMetric.id).all()
            features = []
            actions = []
            rewards = []

            for tm in trade_metrics:
                features.append(torch.tensor(tm.features, dtype=torch.float32))
                actions.append(tm.action)
                rewards.append(tm.reward)

                if len(features) == batch_size:
                    self._update_policy(features, actions, rewards)
                    features, actions, rewards = [], [], []

            # Final batch
            if features:
                self._update_policy(features, actions, rewards)

        # Save updated policy
        self.hub.save_model(self.policy.state_dict(), model_name=self.model_name)
        session.close()