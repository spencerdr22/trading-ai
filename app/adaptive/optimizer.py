"""
Module: optimizer.py
Description:
    Hyperparameter optimization using Optuna. This optimizer runs in
    conjunction with the nightly offline reinforcement-learning update.
    It performs lightweight sampling (few trials) to adapt the RL system
    to changing market regimes.

Features:
    - Bayesian hyperparameter optimization
    - Dynamic learning rate tuning
    - Dynamic gamma (discount factor) tuning
    - Reward weighting optimization
    - Integration with ReinforcementLearner + ModelHub
"""

import optuna
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch.distributions import Categorical

from sqlalchemy.orm import Session
from ..db.init import get_engine
from ..db.schema import TradeMetric

from .reward import compute_batch_reward
from .model_hub import ModelHub
from ..monitor.logger import get_logger

logger = get_logger(__name__)


class RLHyperOptimizer:
    """
    Runs a short Optuna hyperparameter search to improve RL behavior.
    """

    def __init__(self, feature_dim, model_name="adaptive_policy"):
        self.feature_dim = feature_dim
        self.engine = get_engine()
        self.model_name = model_name
        self.hub = ModelHub()

    # ------------------------------------------------------------
    # Trial Objective Function
    # ------------------------------------------------------------
    def _objective(self, trial):
        """
        The Optuna trial objective.
        This defines what hyperparameters get tuned and how performance
        is measured.
        """

        # Hyperparameters to test
        lr = trial.suggest_float("lr", 1e-5, 5e-4, log=True)
        gamma = trial.suggest_float("gamma", 0.90, 0.999)

        pnl_weight = trial.suggest_float("pnl_weight", 0.3, 0.7)
        sharpe_weight = trial.suggest_float("sharpe_weight", 0.1, 0.4)
        sortino_weight = trial.suggest_float("sortino_weight", 0.1, 0.4)
        dd_penalty_weight = trial.suggest_float("dd_penalty_weight", 0.2, 0.5)

        # Simple 1-layer trial policy for speed
        policy = nn.Sequential(
            nn.Linear(self.feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
            nn.Softmax(dim=-1)
        )

        optimizer = optim.Adam(policy.parameters(), lr=lr)

        # -------------------------------------------------------
        # Load session trade history
        # -------------------------------------------------------
        session = Session(self.engine)
        rows = session.query(TradeMetric).all()
        session.close()

        if len(rows) < 20:
            logger.warning("Optuna: insufficient trade data for optimization.")
            return -9999

        trade_pnls = [float(r.pnl) for r in rows]
        wins = sum(1 for r in rows if r.pnl > 0)
        win_rate = wins / len(rows)

        reward = compute_batch_reward(
            pnl_series=trade_pnls,
            win_rate=win_rate,
            pnl_weight=pnl_weight,
            sharpe_weight=sharpe_weight,
            sortino_weight=sortino_weight,
            dd_penalty_weight=dd_penalty_weight,
            win_rate_weight=0.4,
        )

        # -------------------------------------------------------
        # Apply one REINFORCE update to measure effect
        # -------------------------------------------------------

        state_vec = torch.tensor([
            float(np.mean(trade_pnls)),
            float(np.std(trade_pnls)),
            win_rate,
            reward
        ], dtype=torch.float32).unsqueeze(0)

        probs = policy(state_vec)
        dist = Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        loss = -log_prob * reward

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Higher reward is better
        trial_score = reward - float(loss.item())
        return trial_score

    # ------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------
    def optimize(self, n_trials=5):
        """
        Runs a short optimization loop (5–10 trials).
        Stores the best hyperparameters in the model registry.
        """

        logger.info(f"Optuna: Starting {n_trials} hyperparameter trials...")
        study = optuna.create_study(direction="maximize")
        study.optimize(self._objective, n_trials=n_trials)

        best = study.best_params
        score = study.best_value

        logger.info(f"Optuna: Best Params → {best}")
        logger.info(f"Optuna: Best Score → {score:.4f}")

        # Persist best hyperparams into model registry
        self.hub.save_model(
            model={},
            model_name=f"{self.model_name}_hyperparams",
            model_type="Metadata",
            metrics={"params": best, "score": score}
        )

        return best, score

