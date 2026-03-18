"""
engine.py — Adaptive Strategy Engine

Combines supervised model predictions (RF/LSTM/Hybrid) with an optional
RL policy using dynamic confidence-weighted blending.
"""

import numpy as np
import pandas as pd

from ..monitor.logger import get_logger

# Torch is optional — only needed for LSTM / hybrid / RL
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

logger = get_logger(__name__)

ACTIONS      = {0: "SELL", 1: "HOLD", 2: "BUY"}
ACTION_TO_ID = {v: k for k, v in ACTIONS.items()}


class StrategyEngine:
    """
    Integrates a supervised predictor and optional RL policy into a
    unified bar-by-bar signal generator.

    Args:
        predictor : trained ML model (sklearn RF, LSTM wrapper, or Hybrid)
        adaptor   : Adaptor instance for threshold / signal filtering
        cfg       : strategy config dict (from config.yml)
        classes   : label classes — default [-1, 0, 1]
    """

    def __init__(self, predictor, adaptor, cfg: dict, classes=None):
        self.predictor  = predictor
        self.adaptor    = adaptor
        self.cfg        = cfg
        self.classes    = classes or [-1, 0, 1]

        # Pull model and type from predictor safely
        self.model      = getattr(predictor, "model", predictor)
        self.model_type = getattr(predictor, "model_type", "rf")
        self.rl_policy  = getattr(predictor, "rl_policy", None)

        logger.info(
            "StrategyEngine ready | model_type=%s | classes=%s",
            self.model_type, self.classes,
        )

    # ------------------------------------------------------------------
    # Feature helpers
    # ------------------------------------------------------------------

    def _prep_rf_features(self, feat_vector: np.ndarray) -> np.ndarray:
        return feat_vector.reshape(1, -1)

    def _prep_lstm_features(self, window: np.ndarray):
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch required for LSTM models.")
        return torch.tensor(window, dtype=torch.float32).unsqueeze(0)

    # ------------------------------------------------------------------
    # Supervised prediction
    # ------------------------------------------------------------------

    def supervised_predict(self, features: np.ndarray, lstm_window=None) -> dict:
        """Return {"buy": float, "sell": float} probabilities."""
        try:
            if self.model_type == "rf":
                X        = self._prep_rf_features(features)
                prob_up  = float(self.model.predict_proba(X)[0][1])
                return {"buy": prob_up, "sell": 1.0 - prob_up}

            elif self.model_type == "lstm":
                if lstm_window is None:
                    logger.warning("LSTM requires lstm_window — defaulting to neutral.")
                    return {"buy": 0.5, "sell": 0.5}
                if not TORCH_AVAILABLE:
                    return {"buy": 0.5, "sell": 0.5}
                X = self._prep_lstm_features(lstm_window)
                with torch.no_grad():
                    preds = self.model(X)
                return {"buy": float(preds[0][1]), "sell": float(preds[0][0])}

            elif self.model_type == "hybrid":
                if lstm_window is None or not TORCH_AVAILABLE:
                    return {"buy": 0.5, "sell": 0.5}
                lstm_model, rf_model = self.model
                X = self._prep_lstm_features(lstm_window)
                with torch.no_grad():
                    embed = lstm_model.lstm(X)[0][:, -1, :].numpy()
                prob = float(rf_model.predict_proba(embed)[0][1])
                return {"buy": prob, "sell": 1.0 - prob}

            else:
                logger.error("Unknown model_type: %s", self.model_type)
                return {"buy": 0.5, "sell": 0.5}

        except Exception as e:
            logger.error("supervised_predict failed: %s", e)
            return {"buy": 0.5, "sell": 0.5}

    # ------------------------------------------------------------------
    # RL prediction
    # ------------------------------------------------------------------

    def rl_predict(self, state_vector) -> dict | None:
        if self.rl_policy is None or not TORCH_AVAILABLE:
            return None
        try:
            X = torch.tensor(state_vector, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                probs = self.rl_policy(X).squeeze(0).numpy()
            return {
                "sell":      float(probs[0]),
                "hold":      float(probs[1]),
                "buy":       float(probs[2]),
                "probs_raw": probs,
            }
        except Exception as e:
            logger.error("rl_predict failed: %s", e)
            return None

    # ------------------------------------------------------------------
    # Blending
    # ------------------------------------------------------------------

    def blend(self, sup: dict, rl: dict | None) -> dict:
        """Confidence-weighted blend of supervised + RL signals."""
        if rl is None:
            return sup  # pure supervised — no rl_confidence key added here

        probs   = rl["probs_raw"]
        rl_conf = float(np.clip(np.max(probs) - np.min(probs), 0.0, 1.0))
        w_sup   = 1.0 - rl_conf
        w_rl    = rl_conf

        return {
            "buy":           sup["buy"]  * w_sup + rl["buy"]  * w_rl,
            "sell":          sup["sell"] * w_sup + rl["sell"] * w_rl,
            "rl_confidence": rl_conf,
        }

    # ------------------------------------------------------------------
    # Bar-level entry point
    # ------------------------------------------------------------------

    def on_bar(self, X_row) -> dict:
        """
        Called once per bar.  X_row is a single-row DataFrame or Series
        containing numeric feature columns.

        Returns {"side": "BUY"|"SELL"|"HOLD", "strength": float}
        """
        try:
            if isinstance(X_row, pd.Series):
                X_row = X_row.to_frame().T

            features   = X_row.select_dtypes(include=[np.number]).iloc[0].to_numpy()
            decision   = self.decide(features)
            action     = decision["action"].upper()
            signal_map = {"BUY": 1, "HOLD": 0, "SELL": -1}
            raw        = signal_map.get(action, 0)
            adjusted   = self.adaptor.adapt(raw)

            side = "BUY" if adjusted == 1 else "SELL" if adjusted == -1 else "HOLD"
            logger.debug("on_bar: %s (raw=%d adjusted=%d)", side, raw, adjusted)
            return {"side": side, "strength": abs(adjusted)}

        except Exception as e:
            logger.error("on_bar error: %s", e)
            return {"side": "HOLD", "strength": 0}

    # ------------------------------------------------------------------
    # Full decision pipeline
    # ------------------------------------------------------------------

    # Thresholds — only act when the model is genuinely confident
    BUY_THRESHOLD  = 0.60   # probability must exceed this to BUY
    SELL_THRESHOLD = 0.60   # probability of DOWN must exceed this to SELL
    MIN_MARGIN     = 0.08   # buy_prob must beat sell_prob by at least this

    def decide(
        self,
        features: np.ndarray,
        lstm_window=None,
        rl_state_vector=None,
    ) -> dict:
        """
        Run the full supervised -> RL -> blend -> action pipeline.
        Only fires BUY/SELL when confidence is meaningfully above threshold
        AND the model has a clear directional edge (margin check).

        Returns a dict with keys: action, supervised, rl, final.
        """
        sup     = self.supervised_predict(features, lstm_window)
        rl      = self.rl_predict(rl_state_vector) if rl_state_vector is not None else None
        blended = self.blend(sup, rl)

        buy_p  = blended["buy"]
        sell_p = blended["sell"]
        margin = abs(buy_p - sell_p)

        # Require both threshold AND margin to fire — avoids borderline trades
        if buy_p > self.BUY_THRESHOLD and margin >= self.MIN_MARGIN:
            action = "BUY"
        elif sell_p > self.SELL_THRESHOLD and margin >= self.MIN_MARGIN:
            action = "SELL"
        else:
            action = "HOLD"

        rl_conf = blended.get("rl_confidence", 0.0)
        logger.info(
            "Decision: %s  (buy=%.3f  sell=%.3f  margin=%.3f  rl_conf=%.3f)",
            action, buy_p, sell_p, margin, rl_conf,
        )
        return {"action": action, "supervised": sup, "rl": rl, "final": blended}
