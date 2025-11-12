from typing import Dict, Any
import numpy as np

def decision_from_probs(probs: np.ndarray, classes: list, threshold_up: float = 0.6, threshold_down: float = 0.6) -> Dict[str, Any]:
    """
    Convert predict_proba output into a signal dict: side, confidence
    classes: typically [-1, 0, 1] or [0,1] depending on model
    probs: 2D array single row
    """
    # map classes to probs
    row = probs[0]
    mapping = dict(zip(classes, row))
    p_up = mapping.get(1, 0.0)
    p_down = mapping.get(-1, 0.0)
    if p_up >= threshold_up:
        return {"side": "LONG", "confidence": float(p_up)}
    if p_down >= threshold_down:
        return {"side": "SHORT", "confidence": float(p_down)}
    return {"side": "HOLD", "confidence": float(max(p_up, p_down))}
