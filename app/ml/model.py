import joblib
import os
from typing import Any, Optional
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "models_artifacts")
os.makedirs(MODEL_DIR, exist_ok=True)

# Optional torch model
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False
    torch = None
    nn = None

class _LSTMClassifier(nn.Module):
    def __init__(self, n_features: int, hidden: int = 32, num_layers: int = 1, n_classes: int = 3):
        super().__init__()
        self.lstm = nn.LSTM(input_size=n_features, hidden_size=hidden, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden, n_classes)

    def forward(self, x):
        # x: (B, T, F); we use T=1 (last obs) for fast prototype
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        logits = self.fc(out)
        return logits

class TorchPredictor:
    def __init__(self, n_features: int, n_classes: int = 3, device: Optional[str] = None):
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not available. Install torch to use TorchPredictor.")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = _LSTMClassifier(n_features=n_features, n_classes=n_classes).to(self.device)
        self.loss_fn = nn.CrossEntropyLoss()
        self.optim = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.classes_ = np.array([-1, 0, 1])

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 5, batch_size: int = 256):
        self.model.train()
        X = torch.tensor(X, dtype=torch.float32).unsqueeze(1).to(self.device)
        y = torch.tensor(y, dtype=torch.long).to(self.device)
        ds = torch.utils.data.TensorDataset(X, y)
        dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)
        for _ in range(epochs):
            for xb, yb in dl:
                self.optim.zero_grad()
                logits = self.model(xb)
                loss = self.loss_fn(logits, yb + 1)  # map classes [-1,0,1] -> [0,1,2]
                loss.backward()
                self.optim.step()

    def predict_proba(self, X: np.ndarray):
        self.model.eval()
        with torch.no_grad():
            X = torch.tensor(X, dtype=torch.float32).unsqueeze(1).to(self.device)
            logits = self.model(X)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        return probs

    def predict(self, X: np.ndarray):
        probs = self.predict_proba(X)
        idx = probs.argmax(axis=1)
        return self.classes_[idx]

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)

    def load(self, path: str, n_features: int):
        self.model = _LSTMClassifier(n_features=n_features).to(self.device)
        self.model.load_state_dict(torch.load(path, map_location=self.device))

class Predictor:
    """
    Wrapper that uses sklearn by default, or TorchPredictor if use_pytorch=True.
    """
    def __init__(self, use_pytorch: bool = False, model_path: Optional[str] = None, random_state: int = 42, n_features: Optional[int] = None):
        self.use_pytorch = use_pytorch
        self.random_state = random_state
        self.model_path = model_path or os.path.join(MODEL_DIR, "model.pkl")
        self._n_features = n_features

        if self.use_pytorch:
            if not TORCH_AVAILABLE:
                raise RuntimeError("use_pytorch=True but torch is not installed.")
            if n_features is None:
                # will be set on first fit
                pass
            self.pipeline = None
            self.torch_model: Optional[TorchPredictor] = None
        else:
            self.pipeline = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", GradientBoostingClassifier(n_estimators=80, random_state=random_state))
            ])
            self.torch_model = None

    def fit(self, X, y, epochs: int = 5, batch_size: int = 256):
        if self.use_pytorch:
            n_features = X.shape[1]
            self._n_features = n_features
            self.torch_model = TorchPredictor(n_features=n_features)
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

    def save(self, path: Optional[str] = None):
        p = path or self.model_path
        if self.use_pytorch:
            self.torch_model.save(p.replace(".pkl", "_torch.pt"))
        else:
            joblib.dump(self.pipeline, p)

    def load(self, path: Optional[str] = None):
        p = path or self.model_path
        if self.use_pytorch:
            self.torch_model = TorchPredictor(n_features=self._n_features or 8)
            self.torch_model.load(p.replace(".pkl", "_torch.pt"), n_features=self._n_features or 8)
        else:
            if os.path.exists(p):
                self.pipeline = joblib.load(p)
            else:
                raise FileNotFoundError(p)
