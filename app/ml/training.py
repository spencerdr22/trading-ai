# File: app/ml/training.py

import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from app.data.loader import load_sample

MODEL_DIR = os.path.join(os.getcwd(), "data", "models")
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_FILE = os.path.join(MODEL_DIR, "trade_model.pkl")

def load_model():
    if os.path.exists(MODEL_FILE):
        return joblib.load(MODEL_FILE)
    return RandomForestClassifier(n_estimators=100, random_state=42)

def save_model(model):
    joblib.dump(model, MODEL_FILE)

# ---------------------------
# Option A: Incremental Retraining
# ---------------------------
def retrain_on_recent(n_bars=1440):
    df = load_sample()
    if df is None or len(df) < 20:
        print("[WARN] Not enough data to retrain.")
        return
    df = df.tail(n_bars).copy()

    df["target"] = (df["close"].shift(-1) > df["close"]).astype(int)
    X = df[["open", "high", "low", "close", "volume"]][:-1]
    y = df["target"][:-1]

    model = load_model()
    model.fit(X, y)
    save_model(model)

    print(f"[INFO] Model retrained on last {n_bars} bars.")

# ---------------------------
# Option B: Full Rebuild (commented out for now)
# ---------------------------
def rebuild_model():
    """
    Full retraining from all available history.
    """
    df = load_sample()  # fetch all available history
    if len(df) < 100:
        print("[WARN] Not enough data to rebuild model.")
        return

    df["target"] = (df["close"].shift(-1) > df["close"]).astype(int)
    X = df[["open", "high", "low", "close", "volume"]][:-1]
    y = df["target"][:-1]

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X, y)
    save_model(model)

    print("[INFO] Model rebuilt from scratch.")
