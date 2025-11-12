from typing import Dict, Any
from ..ml.model import Predictor
from .signals import decision_from_probs
from .adaption import Adaptor
from ..utils.persistence import save_strategy_params

class StrategyEngine:
    def __init__(self, predictor: Predictor, adaptor: Adaptor, classes: list = None, cfg: dict = None):
        self.predictor = predictor
        self.adaptor = adaptor
        self.classes = classes or [-1, 0, 1]
        self.cfg = cfg or {}
        self.params_history = []

    def on_bar(self, X_row):
        probs = self.predictor.predict_proba(X_row)
        signal = decision_from_probs(probs, self.classes, self.adaptor.threshold_up, self.adaptor.threshold_down)
        return signal

    def adapt(self, recent_trades):
        updated = self.adaptor.update(recent_trades)
        self.params_history.append(updated)
        try:
            save_strategy_params(updated, reason="rolling_adaptation")
        except Exception:
            pass
        return updated
