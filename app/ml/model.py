"""
model.py — Predictor wrapper supporting sklearn (default) and PyTorch LSTM.

MODEL_DIR is set to data/models/ to be consistent with the rest of the project.
"""

import os
import joblib
import numpy as np
from typing import Optional
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# All models go under data/models/ — consistent with trainer.py and model_hub.py
MODEL_DIR = os.path.join(os.getcwd(), "data", "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# ── Optional PyTorch ──────────────────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None  # type: ignore
    nn    = None  # type: ignore


# ── LSTM definition ───────────────────────────────────────────────────────────

class _LSTMClassifier(nn.Module if TORCH_AVAILABLE else object):  # type: ignore
    def __init__(self, n_features: int, hidden: int = 32,
                 num_layers: int = 1, n_classes: int = 3):
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not installed.")
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features, hidden_size=hidden,
            num_layers=num_layers, batch_first=True,
        )
        self.fc = nn.Linear(hidden, n_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


# ── TorchPredictor ────────────────────────────────────────────────────────────

class TorchPredictor:
    """Thin sklearn-compatible wrapper around _LSTMClassifier."""

    def __init__(self, n_features: int, n_classes: int = 3,
                 device: Optional[str] = None):
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is not installed.")
        self.device   = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model    = _LSTMClassifier(n_features, n_classes=n_classes).to(self.device)
        self.loss_fn  = nn.CrossEntropyLoss()
        self.optim    = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.classes_ = np.array([-1, 0, 1])

    def fit(self, X: np.ndarray, y: np.ndarray,
            epochs: int = 5, batch_size: int = 256):
        self.model.train()
        Xt  = torch.tensor(X, dtype=torch.float32).unsqueeze(1).to(self.device)
        yt  = torch.tensor(y, dtype=torch.long).to(self.device)
        ds  = torch.utils.data.TensorDataset(Xt, yt)
        dl  = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)
        for _ in range(epochs):
            for xb, yb in dl:
                self.optim.zero_grad()
                # shift labels from [-1,0,1] -> [0,1,2]
                loss = self.loss_fn(self.model(xb), yb + 1)
                loss.backward()
                self.optim.step()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            Xt    = torch.tensor(X, dtype=torch.float32).unsqueeze(1).to(self.device)
            logits= self.model(Xt)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        return probs

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes_[self.predict_proba(X).argmax(axis=1)]

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)

    def load(self, path: str, n_features: int):
        self.model.load_state_dict(
            torch.load(path, map_location=self.device)
        )


# ── High-level Predictor ──────────────────────────────────────────────────────

class Predictor:
    """
    Unified predictor that delegates to sklearn GBM (default) or TorchPredictor.

    Usage:
        p = Predictor()
        p.fit(X_train, y_train)
        proba = p.predict_proba(X_test)
    """

    def __init__(
        self,
        use_pytorch:  bool           = False,
        model_path:   Optional[str]  = None,
        random_state: int            = 42,
        n_features:   Optional[int]  = None,
    ):
        self.use_pytorch   = use_pytorch
        self.random_state  = random_state
        self.model_path    = model_path or os.path.join(MODEL_DIR, "predictor.pkl")
        self._n_features   = n_features
        self.torch_model:  Optional[TorchPredictor] = None

        if use_pytorch and not TORCH_AVAILABLE:
            raise RuntimeError("use_pytorch=True but PyTorch is not installed.")

        if use_pytorch:
            self.pipeline = None
        else:
            self.pipeline = Pipeline([
                ("scaler", StandardScaler()),
                ("clf",    GradientBoostingClassifier(
                    n_estimators=80, random_state=random_state)),
            ])

    # ------------------------------------------------------------------
    def fit(self, X, y, epochs: int = 5, batch_size: int = 256):
        if self.use_pytorch:
            self._n_features = X.shape[1]
            self.torch_model = TorchPredictor(n_features=self._n_features)
            self.torch_model.fit(X, y, epochs=epochs, batch_size=batch_size)
        else:
            self.pipeline.fit(X, y)

    def predict_proba(self, X):
        if self.use_pytorch:
            return self.torch_model.predict_proba(X)
        return self.pipeline.predict_proba(X)

    def predict(self, X):
        if self.use_pytorch:
            return self.torch_model.predict(X)
        return self.pipeline.predict(X)

    # ------------------------------------------------------------------
    def save(self, path: Optional[str] = None):
        p = path or self.model_path
        os.makedirs(os.path.dirname(p), exist_ok=True)
        if self.use_pytorch:
            self.torch_model.save(p.replace(".pkl", "_torch.pt"))
        else:
            joblib.dump(self.pipeline, p)

    def load(self, path: Optional[str] = None):
        p = path or self.model_path
        if self.use_pytorch:
            n = self._n_features or 8
            self.torch_model = TorchPredictor(n_features=n)
            self.torch_model.load(p.replace(".pkl", "_torch.pt"), n_features=n)
        else:
            if not os.path.exists(p):
                raise FileNotFoundError(f"Model file not found: {p}")
            self.pipeline = joblib.load(p)
