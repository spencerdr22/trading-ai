# File: app/tests/test_model.py
import pandas as pd
from app.ml.trainer import Trainer
from app.data.loader import load_sample


def test_trainer_train_predict():
    df = load_sample()
    trainer = Trainer()
    # ✅ pd.concat replaces deprecated .append()
    predictor = trainer.train(pd.concat([df] + [df] * 30, ignore_index=True))
    assert predictor is not None
