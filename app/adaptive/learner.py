"""
Module: learner.py
Author: Adaptive Framework Generator

Description:
    Offline reinforcement-learning system (REINFORCE) for optimizing
    trading strategies based on historical trade performance. This module
    integrates with the database, model hub, and reward engine.

Features:
    - Medium-capacity policy network (128→64 hidden layers)
    - REINFORCE policy gradient updates
    - Batch reward computation via reward.py
    - Offline training using stored trade metrics
    - Versioned RL model persistence using ModelHub
    - APScheduler + CLI-compatible training entrypoints
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from sqlalchemy.orm import Session
from app.db.init import get_engine
from app.db.schema import TradeMetric

from app.monitor.logger import get_logger
from .reward import compute_batch_reward
from .model_hub import ModelHub

logger = get_logger(__name__)


# ===================================================================
# POLICY NETWORK
# ===================================================================

class PolicyNet(nn.Module):
    """
    Medium-capacity policy network:
        Input → 128 → 64 → Output(3 actions)
    """
    def __init__(self, feature_dim, n_actions=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.net(x)


# ===================================================================
# REINFORCEMENT LEARNER (Offline Training)
# ===================================================================

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

    # ----------------------------------------------------------------------
    # SAVE AND LOAD POLICY
    # ----------------------------------------------------------------------

    def save_policy(self, reward=None):
        """Persist policy network via ModelHub."""
        metrics = {"reward": reward}
        self.hub.save_model(
            model=self.policy,
            model_name=self.model_name,
            model_type="RLPolicy",
            metrics=metrics
        )
        logger.info("ReinforcementLearner: policy saved successfully.")

    def load_latest_policy(self):
        """Load latest saved policy parameters."""
        state = self.hub.load_model(self.model_name, model_type="RLPolicy")
        if state:
            self.policy.load_state_dict(state)
            logger.info("ReinforcementLearner: loaded latest policy.")
        else:
            logger.warning("ReinforcementLearner: no saved policy found; starting fresh.")

    # ----------------------------------------------------------------------
    # POLICY UPDATE (REINFORCE)
    # ----------------------------------------------------------------------

    def update_policy(self, log_probs, reward):
        """
        Perform REINFORCE update:
            loss = -log(pi(a|s)) * reward
        """
        loss = torch.stack([-lp * reward for lp in log_probs]).sum()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return float(loss.item())

    # ----------------------------------------------------------------------
    # OFFLINE TRAIN FROM TRADE HISTORY
    # ----------------------------------------------------------------------

    def train_from_history(self, episodes=5):
        """
        Offline training loop that:
            - pulls trade results from DB
            - computes risk-adjusted reward
            - applies REINFORCE updates
            - saves versioned policy to DB/model hub
        """

        logger.info(f"ReinforcementLearner: offline training started ({episodes} episodes)")

        # -------------------------------
        # Load trade performance data
        # -------------------------------
        session = Session(self.engine)
        rows = session.query(TradeMetric).order_by(TradeMetric.id.asc()).all()
        session.close()

        if not rows or len(rows) < 20:
            logger.warning("RL: insufficient trade history for update.")
            return None

        trade_pnls = [float(r.pnl) for r in rows]
        wins = sum(1 for r in rows if r.pnl > 0)
        win_rate = wins / len(rows)

        reward = compute_batch_reward(trade_pnls, win_rate)
        logger.info(f"RL session reward: {reward:.4f}")

        # -------------------------------
        # Offline RL training episodes
        # -------------------------------
        all_losses = []

        for ep in range(episodes):
            log_probs = []

            # create synthetic "state" from PnL distribution features
            # these are low-dimensional state approximations
            pnl_tensor = torch.tensor([
                float(torch.mean(torch.tensor(trade_pnls))),
                float(torch.std(torch.tensor(trade_pnls))),
                win_rate,
                reward
            ], dtype=torch.float32).unsqueeze(0)

            # forward pass
            probs = self.policy(pnl_tensor)
            dist = Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            log_probs.append(log_prob)

            # update policy
            loss = self.update_policy(log_probs, reward)
            all_losses.append(loss)

            logger.info(f"RL Episode {ep+1}/{episodes} — Loss={loss:.6f}")

        # -------------------------------
        # Save model version
        # -------------------------------
        self.save_policy(reward=reward)
        logger.info("ReinforcementLearner: offline training complete.")

        return {
            "reward": reward,
            "losses": all_losses,
            "episodes": episodes
        }

