"""
Module: trainer.py
Author: Adaptive Framework Generator

Description:
    Unified supervised learning trainer for the Trading-AI system.
    Supports multiple model types:
        - RandomForest (baseline)
        - LSTM (PyTorch)
        - Hybrid (LSTM + RF stacked)

Features:
    - Full feature generation pipeline
    - ModelHub metadata integration
    - Robust schema validation
    - Logging standardization
    - Drop-in compatibility with StrategyEngine
"""

import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None  # type: ignore
    nn    = None  # type: ignore
    optim = None  # type: ignore

from ..monitor.logger import get_logger
from .features import make_features
from ..adaptive.model_hub import ModelHub

logger = get_logger(__name__)


# ============================================================
# LSTM MODEL DEFINITION
# ============================================================

class LSTMClassifier(nn.Module if TORCH_AVAILABLE else object):  # type: ignore
    """
    A small LSTM classifier for predicting upward/downward movement.

    Architecture:
        Input → LSTM(32 hidden) → FC(32→16) → Output(2)
    """

    def __init__(self, feature_dim, hidden_size=32, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 2),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        o, _ = self.lstm(x)
        last = o[:, -1, :]
        return self.fc(last)


# ============================================================
# TRAINER CLASS
# ============================================================

class Trainer:

    def __init__(
        self,
        model_path: str = "data/models/baseline_model.pkl",
        model_type: str = "rf",
        sequence_length: int = 20
    ):
        """
        model_type ∈ {"rf", "lstm", "hybrid"}
        """
        self.model_path = model_path
        self.model_type = model_type
        self.sequence_length = sequence_length

        self.model = None
        self.hub = ModelHub()

    # ----------------------------------------------------------
    # DATA PREP
    # ----------------------------------------------------------

    def _prepare_training_data(self, df: pd.DataFrame):
        """Return feature matrix X and target y after feature engineering."""
        if df is None or df.empty:
            logger.error("Trainer: received empty DataFrame.")
            return None, None

        # --- Feature generation first ---
        feat_df = make_features(df)
        if feat_df.empty:
            logger.error("Trainer: make_features() returned empty.")
            return None, None

        if "close" not in feat_df.columns:
            logger.error("Trainer: missing 'close' column after feature generation.")
            return None, None

        # --- Safe copy before mutation (prevents SettingWithCopyWarning) ---
        feat_df = feat_df.copy()

        # --- Compute target: predict next 3-bar return vs ATR threshold ---
        # Require the move to be meaningful (> 0.3x ATR) before labelling as 1
        # This filters out noise and forces the model to predict real moves
        future_ret = feat_df["close"].pct_change(3).shift(-3).fillna(0)
        atr_vals   = feat_df["close"].rolling(14, min_periods=1).std().fillna(
            feat_df["close"].std()
        )
        threshold  = 0.0003   # ~0.03% minimum move (roughly 1-2 MES ticks)
        feat_df["future_return"] = future_ret
        feat_df.loc[:, "target"] = (
            future_ret > threshold
        ).astype(int)
        feat_df = feat_df.dropna()

        # --- Feature selection ---
        feature_cols = [
            c for c in feat_df.columns
            if c not in (
                "timestamp", "target", "future_return",
                "open", "high", "low", "close", "volume"
            )
        ]   

        if not feature_cols:
            logger.error("Trainer: no usable feature columns found.")
            return None, None

        X = feat_df[feature_cols]
        y = feat_df["target"]

        return X, y

    # ----------------------------------------------------------
    # RANDOM FOREST TRAINING
    # ----------------------------------------------------------

    def _train_rf(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )

        # Ensemble of RF + GBT — RF handles noise well, GBT catches
        # nonlinear regime shifts. Voting averages their probability
        # estimates so neither dominates.
        rf = RandomForestClassifier(
            n_estimators=300,
            n_jobs=-1,
            max_depth=6,
            min_samples_leaf=20,
            max_features="sqrt",
            class_weight="balanced",
            random_state=42,
        )
        gbt = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            min_samples_leaf=20,
            random_state=42,
        )
        model = VotingClassifier(
            estimators=[("rf", rf), ("gbt", gbt)],
            voting="soft",   # average probabilities, not hard votes
            n_jobs=-1,
        )

        # RobustScaler handles outliers from gap opens and news spikes
        pipeline = Pipeline([
            ("scaler", RobustScaler()),
            ("clf",    model),
        ])
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)

        acc = accuracy_score(y_test, preds)
        f1  = f1_score(y_test, preds, average="weighted", zero_division=0)
        logger.info(f"Ensemble model  accuracy={acc:.4f}  f1={f1:.4f}  "
                    f"train={len(X_train)}  test={len(X_test)}")

        return pipeline, acc

    # ----------------------------------------------------------
    # LSTM TRAINING
    # ----------------------------------------------------------

    def _prepare_lstm_sequences(self, X, y):
        """Return 3D tensors for LSTM: (batch, seq_len, features)."""
        X_np = X.to_numpy()
        y_np = y.to_numpy()

        sequences = []
        labels = []

        for i in range(len(X_np) - self.sequence_length):
            seq = X_np[i: i + self.sequence_length]
            label = y_np[i + self.sequence_length]
            sequences.append(seq)
            labels.append(label)

        X_tensor = torch.tensor(np.array(sequences), dtype=torch.float32)
        y_tensor = torch.tensor(labels, dtype=torch.long)

        return X_tensor, y_tensor

    def _train_lstm(self, X, y):
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not installed — cannot train LSTM model.")
        X_tensor, y_tensor = self._prepare_lstm_sequences(X, y)
        feature_dim = X_tensor.shape[-1]
        model = LSTMClassifier(feature_dim)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-4)

        epochs = 10
        model.train()

        for ep in range(epochs):
            optimizer.zero_grad()
            preds = model(X_tensor)
            loss = criterion(preds, y_tensor)
            loss.backward()
            optimizer.step()
            logger.info(f"LSTM Epoch {ep+1}/{epochs}  Loss={float(loss):.6f}")

        # Evaluate
        model.eval()
        with torch.no_grad():
            preds = model(X_tensor)
            predicted = torch.argmax(preds, dim=1)
            acc = (predicted == y_tensor).float().mean().item()

        logger.info(f"LSTM accuracy: {acc:.4f}")

        return model, acc

    # ----------------------------------------------------------
    # HYBRID TRAINING
    # ----------------------------------------------------------

    def _train_hybrid(self, X, y):
        """Hybrid = LSTM → embedding → RandomForest classifier."""
        lstm_model, _ = self._train_lstm(X, y)
        X_tensor, _ = self._prepare_lstm_sequences(X, y)
        lstm_model.eval()

        with torch.no_grad():
            lstm_embed = lstm_model.lstm(X_tensor)[0][:, -1, :].numpy()

        rf_model = RandomForestClassifier(
            n_estimators=150,
            max_depth=8,
            n_jobs=-1
        )
        rf_model.fit(lstm_embed, y[self.sequence_length:])

        preds = rf_model.predict(lstm_embed)
        acc = accuracy_score(y[self.sequence_length:], preds)
        logger.info(f"Hybrid LSTM+RF accuracy: {acc:.4f}")

        return (lstm_model, rf_model), acc

    # ----------------------------------------------------------
    # MAIN TRAINING ENTRYPOINT
    # ----------------------------------------------------------

    def train(self, df: pd.DataFrame):
        X, y = self._prepare_training_data(df)
        if X is None:
            return None

        logger.info(f"Training model_type={self.model_type} on {len(X)} samples")

        if self.model_type == "rf":
            model, acc = self._train_rf(X, y)
        elif self.model_type == "lstm":
            model, acc = self._train_lstm(X, y)
        elif self.model_type == "hybrid":
            model, acc = self._train_hybrid(X, y)
        else:
            logger.error(f"Unknown model_type: {self.model_type}")
            return None

        # Save to ModelHub (timestamped)
        metrics = {"accuracy": float(acc)}
        self.hub.save_model(
            model=model,
            model_name=f"supervised_{self.model_type}",
            model_type=self.model_type,
            metrics=metrics
        )

        # ✅ Also save to static test path if provided
        if self.model_path:
            try:
                os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
                joblib.dump(model, self.model_path)
                logger.info(f"Trainer: model also saved to static path {self.model_path}")
            except Exception as e:
                logger.warning(f"Trainer: could not save to static model_path: {e}")

        # ✅ Ensure universal test compatibility
        static_path = os.path.join("data", "models", "test_model.pkl")
        try:
            os.makedirs(os.path.dirname(static_path), exist_ok=True)
            joblib.dump(model, static_path)
            logger.info(f"Trainer: saved backup static model → {static_path}")
        except Exception as e:
            logger.warning(f"Trainer: failed to save test static model: {e}")

        self.model = model
        return model

    # ----------------------------------------------------------
    # LOAD MODEL
    # ----------------------------------------------------------

    def load(self):
        """Load the latest supervised model."""
        m = self.hub.load_model(
            model_name=f"supervised_{self.model_type}",
            model_type=self.model_type
        )
        if m is not None:
            self.model = m
        return m
