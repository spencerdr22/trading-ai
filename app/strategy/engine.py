# File: app/strategy/engine.py
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict
from ..monitor.logger import get_logger
from ..ml.trainer import Trainer
from ..ml.features import make_features

logger = get_logger(__name__)

@dataclass
class StrategyParams:
    """Strategy configuration parameters."""
    threshold_buy: float = 0.55
    threshold_sell: float = 0.45
    position_size: float = 1.0
    retrain_interval: int = 100  # trades before triggering retraining

class StrategyEngine:
    """
    Generates trading signals using a trained ML model.
    Can adaptively update thresholds and integrate with paper or live executors.
    """

    def __init__(self, model_path: str = "data/models/model.pkl", params: Optional[StrategyParams] = None):
        self.params = params or StrategyParams()
        self.trainer = Trainer(model_path=model_path)
        self.model = self.trainer.load()
        self.trade_counter = 0

    def generate_signal(self, df: pd.DataFrame) -> Optional[Dict]:
        """
        Generate a trading signal (buy/sell/hold) based on most recent market features.
        """
        if df is None or df.empty:
            logger.error("StrategyEngine: empty DataFrame received.")
            return None

        if self.model is None:
            logger.warning("StrategyEngine: model not loaded. Attempting reload.")
            self.model = self.trainer.load()
            if self.model is None:
                logger.error("StrategyEngine: unable to load model.")
                return None

        # === Prepare latest feature row ===
        feat_df = make_features(df).dropna()
        if feat_df.empty:
            logger.warning("StrategyEngine: insufficient data for signal generation.")
            return None

        latest_row = feat_df.iloc[-1:]
        feature_cols = [
            c for c in latest_row.columns
            if c not in ("timestamp", "target", "future_return", "open", "high", "low", "close", "volume")
        ]

        # === Predict probability ===
        try:
            proba = self.model.predict_proba(latest_row[feature_cols])[0][1]
        except Exception as e:
            logger.error(f"StrategyEngine: prediction failed ({e})")
            return None

        signal = self._interpret_signal(proba)
        logger.info(f"StrategyEngine: prob={proba:.3f} → signal={signal}")

        self.trade_counter += 1
        return {"signal": signal, "prob": proba, "position_size": self.params.position_size}

    def _interpret_signal(self, prob: float) -> str:
        """Map model probability to discrete signal."""
        if prob >= self.params.threshold_buy:
            return "BUY"
        elif prob <= self.params.threshold_sell:
            return "SELL"
        return "HOLD"

    def maybe_retrain(self, df: pd.DataFrame):
        """Retrain periodically based on trade count."""
        if self.trade_counter >= self.params.retrain_interval:
            logger.info(f"StrategyEngine: retraining model after {self.trade_counter} trades.")
            self.trainer.train(df)
            self.model = self.trainer.load()
            self.trade_counter = 0