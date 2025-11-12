import joblib
import os
from typing import List, Dict
from ..db import get_session
from ..models.schema import StrategyParam

def save_artifact(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True
    )
    joblib.dump(obj, path)

def load_artifact(path):
    return joblib.load(path)

def save_strategy_params(params: dict, reason: str = "adapt"):
    with get_session() as s:
        for k, v in params.items():
            if k == "win_rate":
                continue
            s.add(StrategyParam(name=k, value=float(v), reason=reason))
        s.commit()
