# File: app/ml/trainer.py
import os
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from ..monitor.logger import get_logger
from .features import make_features

logger = get_logger(__name__)

class Trainer:
    def __init__(self, model_path: str = "data/models/model.pkl"):
        self.model_path = model_path
        self.model = None

    def train(self, df: pd.DataFrame):
        """Train a RandomForest model on market data with feature engineering."""
        if df is None or df.empty:
            logger.error("Trainer: received empty DataFrame.")
            return None

        logger.info(f"Trainer: received {len(df)} raw rows.")

        # === Generate features ===
        feat_df = make_features(df)
        logger.info(f"Trainer: make_features() returned {len(feat_df)} rows, {len(feat_df.columns)} columns")

        if feat_df.empty:
            logger.error("No feature data available for training.")
            return None

        # === Ensure 'close' column exists ===
        if "close" not in feat_df.columns:
            logger.error("Trainer: missing 'close' column in feature data.")
            return None

        # === Create target ===
        feat_df["future_return"] = feat_df["close"].pct_change().shift(-1)
        feat_df["target"] = (feat_df["future_return"] > 0).astype(int)
        feat_df = feat_df.dropna()

        if len(feat_df) < 50:
            logger.warning(f"Trainer: only {len(feat_df)} rows after target generation — not enough for training.")
            return None

        # === Feature selection ===
        feature_cols = [c for c in feat_df.columns if c not in ("timestamp", "target", "future_return", "open", "high", "low", "close", "volume")]
        if not feature_cols:
            logger.error("Trainer: no usable feature columns found.")
            return None

        X = feat_df[feature_cols]
        y = feat_df["target"]

        # === Split train/test ===
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

        # === Model training ===
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        self.model = model

        # === Evaluate ===
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        logger.info(f"Trainer: model trained. Accuracy={acc:.3f}, Samples={len(X)}")

        # === Save ===
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump(model, self.model_path)
        logger.info(f"Trainer: model saved to {self.model_path}")

        return model

    def load(self):
        """Load a trained model from disk."""
        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)
            logger.info(f"Trainer: loaded model from {self.model_path}")
            return self.model
        logger.warning(f"Trainer: model not found at {self.model_path}")
        return None
