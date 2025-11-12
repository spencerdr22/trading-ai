# File: app/ml/model_manager.py
import os
import pickle
from datetime import datetime
import numpy as np

MODEL_DIR = os.path.join(os.getcwd(), "models")
os.makedirs(MODEL_DIR, exist_ok=True)

METRICS_LOG = os.path.join(MODEL_DIR, "metrics_log.csv")

def save_model_if_better(model, metrics, lookback):
    """
    Save model only if performance improves.
    Metrics = dict with keys like {"accuracy": 0.67, "sharpe": 1.4}
    """
    accuracy = metrics.get("accuracy", 0)
    sharpe = metrics.get("sharpe", 0)

    # Create model label
    date_str = datetime.utcnow().strftime("%Y-%m-%d")
    label = f"model_{date_str}_acc{accuracy:.2f}_lookback{lookback}"
    filepath = os.path.join(MODEL_DIR, f"{label}.pkl")

    # Check best existing model
    best_acc = 0
    if os.path.exists(METRICS_LOG):
        import pandas as pd
        df = pd.read_csv(METRICS_LOG)
        if not df.empty:
            best_acc = df["accuracy"].max()

    # Save only if improved
    if accuracy > best_acc:
        with open(filepath, "wb") as f:
            pickle.dump(model, f)

        # Append to log
        with open(METRICS_LOG, "a") as f:
            f.write(f"{date_str},{accuracy:.4f},{sharpe:.4f},{label}\n")

        print(f"[MODEL] New model saved: {label}")
        return label
    else:
        print(f"[MODEL] Discarded (acc={accuracy:.2f} < best={best_acc:.2f})")
        return None

def get_best_model():
    """Load latest best model from storage."""
    import pandas as pd
    if not os.path.exists(METRICS_LOG):
        return None, None
    df = pd.read_csv(METRICS_LOG)
    if df.empty:
        return None, None
    best_row = df.loc[df["accuracy"].idxmax()]
    path = os.path.join(MODEL_DIR, f"{best_row['label']}.pkl")
    with open(path, "rb") as f:
        model = pickle.load(f)
    return model, best_row["label"]

def load_metrics_history():
    """Return full metrics log as DataFrame."""
    import pandas as pd
    if os.path.exists(METRICS_LOG):
        return pd.read_csv(METRICS_LOG)
    return None
