"""
Module: engine.py
Author: Adaptive Framework Generator

Description:
    Adaptive strategy engine combining:
        (1) Supervised model prediction (RF, LSTM, Hybrid)
        (2) RL policy output (probabilistic BUY/HOLD/SELL)
    using dynamic blending based on RL confidence.

Blending Logic:
    score = α * supervised + β * rl

    where β = RL_confidence
          α = 1 - RL_confidence

    RL_confidence = max(policy_probs) - min(policy_probs)
"""

import numpy as np
import torch

from ..monitor.logger import get_logger
from ..adaptive.model_hub import ModelHub

logger = get_logger(__name__)


# ================================================================
# ACTION ENUMERATION
# ================================================================

ACTIONS = {
    0: "SELL",
    1: "HOLD",
    2: "BUY",
}

ACTION_TO_ID = {v: k for k, v in ACTIONS.items()}


# ================================================================
# STRATEGY ENGINE
# ================================================================

class StrategyEngine:

    def __init__(self, model, model_type="rf", rl_model_name="adaptive_policy"):
        """
        model         — supervised model (RF/LSTM/Hybrid)
        model_type    — "rf", "lstm", or "hybrid"
        rl_model_name — name used in ModelHub for RL policy
        """
        self.model = model
        self.model_type = model_type

        self.hub = ModelHub()

        # Load RL policy network (PyTorch)
        self.rl_policy = self.hub.load_model(
            model_name=rl_model_name,
            model_type="RLPolicy"
        )

        if self.rl_policy is None:
            logger.warning("StrategyEngine: No RL policy found, running supervised-only.")
        else:
            self.rl_policy.eval()

        logger.info(f"StrategyEngine initialized: model_type={model_type}")

    # --------------------------------------------------------------------
    # FEATURE PREPARATION
    # --------------------------------------------------------------------

    def _prep_supervised_features(self, feat_vector):
        """
        Prepare input vector for RF or hybrid model.
        feat_vector must be a 1D numpy array.
        """
        return feat_vector.reshape(1, -1)

    def _prep_lstm_features(self, window):
        """
        Prepare a window of past features for LSTM input.
        window shape: (seq_len, features)
        """
        return torch.tensor(window, dtype=torch.float32).unsqueeze(0)

    # --------------------------------------------------------------------
    # SUPERVISED PREDICTION
    # --------------------------------------------------------------------

    def supervised_predict(self, features, lstm_window=None):
        """
        Returns supervised model prediction (probability of BUY).
        """

        if self.model_type == "rf":
            X = self._prep_supervised_features(features)
            prob_up = float(self.model.predict_proba(X)[0][1])
            return {"buy": prob_up, "sell": 1 - prob_up}

        elif self.model_type == "lstm":
            if lstm_window is None:
                logger.error("LSTM model requires `lstm_window` input.")
                return {"buy": 0.5, "sell": 0.5}

            X = self._prep_lstm_features(lstm_window)
            with torch.no_grad():
                preds = self.model(X)
            buy_prob = float(preds[0][1])
            return {"buy": buy_prob, "sell": 1 - buy_prob}

        elif self.model_type == "hybrid":
            # Hybrid = LSTM encoding + RF classifier
            lstm_model, rf_model = self.model

            if lstm_window is None:
                logger.error("Hybrid model requires `lstm_window` input.")
                return {"buy": 0.5, "sell": 0.5}

            X = self._prep_lstm_features(lstm_window)
            with torch.no_grad():
                lstm_embed = lstm_model.lstm(X)[0][:, -1, :].numpy()

            prob = float(rf_model.predict_proba(lstm_embed)[0][1])
            return {"buy": prob, "sell": 1 - prob}

        else:
            logger.error(f"Unknown model_type: {self.model_type}")
            return {"buy": 0.5, "sell": 0.5}

    # --------------------------------------------------------------------
    # RL POLICY PREDICTION
    # --------------------------------------------------------------------

    def rl_predict(self, state_vector):
        """
        state_vector = [mean_pnl, std_pnl, win_rate, reward_estimate]
        """
        if self.rl_policy is None:
            return None

        X = torch.tensor(state_vector, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            probs = self.rl_policy(X).squeeze(0).numpy()

        buy = probs[2]
        hold = probs[1]
        sell = probs[0]

        return {
            "buy": float(buy),
            "hold": float(hold),
            "sell": float(sell),
            "probs_raw": probs
        }

    # --------------------------------------------------------------------
    # DYNAMIC BLENDING
    # --------------------------------------------------------------------

    def blend(self, sup, rl):
        """
        Dynamic blending:
            RL_conf = max(probs) - min(probs)
            weight_rl = RL_conf
            weight_sup = 1 - RL_conf
        """
        if rl is None:
            return sup  # supervised fallback

        probs = rl["probs_raw"]
        rl_conf = float(np.max(probs) - np.min(probs))
        rl_conf = max(0.0, min(1.0, rl_conf))

        weight_rl = rl_conf
        weight_sup = 1.0 - rl_conf

        blended_buy = sup["buy"] * weight_sup + rl["buy"] * weight_rl
        blended_sell = sup["sell"] * weight_sup + rl["sell"] * weight_rl

        return {
            "buy": blended_buy,
            "sell": blended_sell,
            "rl_confidence": rl_conf,
        }

    # --------------------------------------------------------------------
    # FINAL DECISION
    # --------------------------------------------------------------------

    def decide(
        self,
        features,
        lstm_window=None,
        rl_state_vector=None
    ):
        """
        Returns final action recommendation.
        """

        # 1. Supervised model prediction
        sup = self.supervised_predict(features, lstm_window)

        # 2. RL policy prediction
        rl = None
        if rl_state_vector is not None:
            rl = self.rl_predict(rl_state_vector)

        # 3. Blending
        blended = self.blend(sup, rl)

        # 4. Action decision
        action = "BUY" if blended["buy"] > 0.55 else "SELL" if blended["sell"] > 0.55 else "HOLD"

        logger.info(
            f"Decision → {action}   "
            f"(sup_buy={sup['buy']:.3f}, rl_conf={blended['rl_confidence']:.3f})"
        )

        return {
            "action": action,
            "supervised": sup,
            "rl": rl,
            "final": blended
        }
