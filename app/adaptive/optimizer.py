"""
Module: optimizer.py
Optuna hyperparameter search for the RL policy.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import optuna

from torch.distributions import Categorical
from sqlalchemy.orm import Session

from ..db.init import get_engine
from ..models.schema import TradeMetric        # canonical schema
from .reward import compute_batch_reward
from .model_hub import ModelHub
from ..monitor.logger import get_logger

logger = get_logger(__name__)

# Silence Optuna's verbose per-trial logging
optuna.logging.set_verbosity(optuna.logging.WARNING)


class RLHyperOptimizer:
    """
    Short Optuna study (5-10 trials) to tune RL hyperparameters.
    Uses trade history already stored in the DB.
    """

    def __init__(self, feature_dim: int = 4, model_name: str = "adaptive_policy"):
        self.feature_dim = feature_dim
        self.engine      = get_engine()
        self.model_name  = model_name
        self.hub         = ModelHub()

    # ------------------------------------------------------------------
    def _load_trade_history(self):
        """Pull PnL values from TradeMetric table."""
        session = Session(self.engine)
        try:
            rows = session.query(TradeMetric).all()
            return rows
        finally:
            session.close()

    # ------------------------------------------------------------------
    def _objective(self, trial: optuna.Trial) -> float:
        lr                = trial.suggest_float("lr",               1e-5, 5e-4, log=True)
        _gamma            = trial.suggest_float("gamma",            0.90, 0.999)   # noqa: F841
        pnl_weight        = trial.suggest_float("pnl_weight",       0.3,  0.7)
        sharpe_weight     = trial.suggest_float("sharpe_weight",    0.05, 0.15)
        sortino_weight    = trial.suggest_float("sortino_weight",   0.2,  0.4)
        dd_penalty_weight = trial.suggest_float("dd_penalty_weight",0.2,  0.5)

        rows = self._load_trade_history()
        if len(rows) < 20:
            logger.warning("Optuna: not enough trade data (%d rows).", len(rows))
            return -9999.0

        trade_pnls = [float(r.pnl) for r in rows]
        win_rate   = sum(1 for r in rows if r.pnl > 0) / len(rows)

        reward = compute_batch_reward(
            trade_pnls,
            win_rate,
            pnl_weight        = pnl_weight,
            sharpe_weight     = sharpe_weight,
            sortino_weight    = sortino_weight,
            dd_penalty_weight = dd_penalty_weight,
            win_rate_weight   = 0.4,
        )

        # Build a small trial policy and apply one REINFORCE step
        policy = nn.Sequential(
            nn.Linear(self.feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
            nn.Softmax(dim=-1),
        )
        optimizer = optim.Adam(policy.parameters(), lr=lr)

        state_vec = torch.tensor(
            [float(np.mean(trade_pnls)),
             float(np.std(trade_pnls)),
             win_rate,
             reward],
            dtype=torch.float32,
        ).unsqueeze(0)

        probs    = policy(state_vec)
        dist     = Categorical(probs)
        action   = dist.sample()
        log_prob = dist.log_prob(action)

        loss = -log_prob * reward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return float(reward - loss.item())

    # ------------------------------------------------------------------
    def optimize(self, n_trials: int = 5):
        """
        Run the Optuna study and persist the best hyperparameters.

        Returns:
            (best_params dict, best_score float)
        """
        logger.info("Optuna: starting %d trials...", n_trials)
        study = optuna.create_study(direction="maximize")
        study.optimize(self._objective, n_trials=n_trials, show_progress_bar=False)

        best   = study.best_params
        score  = study.best_value

        logger.info("Optuna: best score=%.4f  params=%s", score, best)

        # Persist as a metadata-only record in the hub
        self.hub.save_model(
            model      = best,          # plain dict — joblib handles this fine
            model_name = f"{self.model_name}_hyperparams",
            model_type = "Metadata",
            metrics    = {"params": best, "score": score},
        )

        return best, score
