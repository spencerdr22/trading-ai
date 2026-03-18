"""
learner.py — Offline REINFORCE policy gradient learner.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime, timezone
from torch.distributions import Categorical

from app.db.init import get_engine, get_session
from app.models.schema import TradeMetric
from app.monitor.logger import get_logger
from .reward import compute_batch_reward
from .model_hub import ModelHub

logger = get_logger(__name__)


class PolicyNet(nn.Module):
    """128 -> 64 -> 3-action softmax policy."""

    def __init__(self, feature_dim: int, n_actions: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions),
            nn.Softmax(dim=-1),
        )

    def forward(self, x):
        return self.net(x)


class ReinforcementLearner:
    """Offline REINFORCE learner using trade history from the DB."""

    def __init__(
        self,
        feature_dim: int  = 4,
        lr:          float = 1e-4,
        gamma:       float = 0.99,
        model_name:  str   = "adaptive_policy",
    ):
        self.policy     = PolicyNet(feature_dim)
        self.optimizer  = optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma      = gamma
        self.model_name = model_name
        self.hub        = ModelHub()
        logger.info("ReinforcementLearner ready | feature_dim=%d lr=%s", feature_dim, lr)

    def save_policy(self, reward: float = None):
        self.hub.save_model(
            model      = self.policy,
            model_name = self.model_name,
            model_type = "RLPolicy",
            metrics    = {"reward": reward},
        )
        logger.info("Policy saved (reward=%.4f).", reward or 0)

    def load_latest_policy(self):
        state = self.hub.load_model(self.model_name, model_type="RLPolicy")
        if state is not None:
            try:
                self.policy.load_state_dict(state)
                logger.info("Loaded latest policy from ModelHub.")
            except Exception as e:
                logger.warning("Could not load policy state dict: %s", e)
        else:
            logger.info("No saved policy found — starting fresh.")

    def update_policy(self, log_probs: list, reward: float) -> float:
        loss = torch.stack([-lp * reward for lp in log_probs]).sum()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return float(loss.item())

    def train_from_history(self, episodes: int = 5):
        """
        Pull trade PnL from DB and run REINFORCE updates.
        All DB attribute access happens inside the session to avoid
        DetachedInstanceError.
        """
        logger.info("RL offline training: %d episodes", episodes)

        # Read all values INSIDE the session before it closes
        try:
            with get_session() as s:
                rows = (
                    s.query(TradeMetric)
                     .order_by(TradeMetric.id.asc())
                     .all()
                )
                # Extract primitive values while session is still open
                trade_pnls = [float(r.pnl) for r in rows]
                n_rows     = len(rows)
        except Exception as e:
            logger.error("Could not load trade history: %s", e)
            return None

        if n_rows < 20:
            logger.warning("RL: only %d trades in DB — need >=20.", n_rows)
            return None

        win_rate = sum(1 for p in trade_pnls if p > 0) / n_rows
        reward   = compute_batch_reward(trade_pnls, win_rate)
        logger.info("RL reward: %.4f  win_rate=%.2f  trades=%d",
                    reward, win_rate, n_rows)

        pnl_t = torch.tensor(trade_pnls, dtype=torch.float32)
        state = torch.tensor(
            [float(pnl_t.mean()), float(pnl_t.std()), win_rate, reward],
            dtype=torch.float32,
        ).unsqueeze(0)

        all_losses = []
        for ep in range(episodes):
            probs    = self.policy(state)
            dist     = Categorical(probs)
            action   = dist.sample()
            log_prob = dist.log_prob(action)
            loss     = self.update_policy([log_prob], reward)
            all_losses.append(loss)
            logger.info("RL episode %d/%d  loss=%.6f", ep + 1, episodes, loss)

        self.save_policy(reward=reward)
        logger.info("RL offline training complete.")
        return {"reward": reward, "losses": all_losses, "episodes": episodes}
