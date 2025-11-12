from ..ml.training import Trainer
from ..data.loader import load_sample
from ..ml.features import make_features

def test_trainer_train_predict():
    df = load_sample()
    trainer = Trainer()
    predictor = trainer.train(df.append([df]*30, ignore_index=True))
    feat = make_features(df)
    X = feat.drop(columns=["timestamp","open","high","low","close","volume"], errors="ignore")
    probs = predictor.predict_proba(X.iloc[[0]])
    assert probs.shape[0] == 1 and probs.shape[1] >= 2
