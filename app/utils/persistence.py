"""
persistence.py — Artifact save/load and DB strategy param helpers.
"""

import os
import joblib
from typing import Any

from ..db import get_session
from ..models.schema import StrategyParam


def save_artifact(obj: Any, path: str) -> None:
    """Persist any Python object to disk with joblib."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    joblib.dump(obj, path)


def load_artifact(path: str) -> Any:
    """Load a joblib artifact from disk."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Artifact not found: {path}")
    return joblib.load(path)


def save_strategy_params(params: dict, reason: str = "adapt") -> None:
    """
    Persist strategy parameter values to the DB.
    Skips the 'win_rate' key (it's a metric, not a tunable param).
    """
    with get_session() as s:
        for key, value in params.items():
            if key == "win_rate":
                continue
            try:
                s.add(StrategyParam(
                    name   = key,
                    value  = float(value),
                    reason = reason,
                ))
            except (TypeError, ValueError):
                pass   # skip non-numeric values silently
        s.commit()
