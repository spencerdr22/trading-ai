"""
test_system_integrity.py — Automated system-wide sanity checks.

Run with:
    pytest -v app/tests/test_system_integrity.py
    python app/tests/test_system_integrity.py
"""

import os
import numpy as np
import pandas as pd
import traceback
from pathlib import Path

from app.db.init import get_engine, get_session
from app.models.schema import Base, TradeMetric   # canonical schema
from app.ml.trainer import Trainer
from app.adaptive.model_hub import ModelHub
from app.monitor.logger import get_logger

logger = get_logger(__name__)


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_df(rows: int = 200) -> pd.DataFrame:
    np.random.seed(42)
    return pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01", periods=rows, freq="min"),
        "open":      np.random.rand(rows) * 100 + 5700,
        "high":      np.random.rand(rows) * 100 + 5750,
        "low":       np.random.rand(rows) * 100 + 5650,
        "close":     np.random.rand(rows) * 100 + 5700,
        "volume":    np.random.randint(100, 1000, rows),
    })


# ── TEST 1 — DB connectivity ──────────────────────────────────────────────────

def test_database_connection():
    """Engine connects and ORM tables are creatable."""
    engine = get_engine()
    Base.metadata.create_all(engine)
    logger.info("DB connection OK.")


# ── TEST 2 — DB write / read via context-manager session ─────────────────────

def test_mock_data_insertion():
    """Insert a TradeMetric row and read it back."""
    from datetime import datetime, timezone
    with get_session() as s:
        row = TradeMetric(
            symbol    = "MES",
            timestamp = datetime.now(timezone.utc),
            side      = {"side": "BUY", "confidence": 0.7},
            pnl       = 12.50,
            status    = "FILLED",
        )
        s.add(row)
        s.commit()

    # Read back
    with get_session() as s:
        result = s.query(TradeMetric).filter_by(symbol="MES").first()
        assert result is not None, "TradeMetric row not found after insert"
    logger.info("DB write/read OK.")


# ── TEST 3 — Model training ───────────────────────────────────────────────────

def test_model_training():
    """Train a RandomForest on synthetic data and check the file exists."""
    model_path = Path("data/models/test_model.pkl")
    df      = _make_df(200)
    trainer = Trainer(model_path=str(model_path))
    model   = trainer.train(df)

    assert model is not None, "Trainer returned None"
    assert model_path.exists(), f"Model file not found: {model_path}"
    logger.info("Model training OK.")


# ── TEST 4 — ModelHub save / load ─────────────────────────────────────────────

def test_modelhub_operations():
    """Round-trip a dummy model through ModelHub."""
    hub          = ModelHub()
    dummy_model  = {"param": 42, "description": "test"}
    saved_path   = hub.save_model(
        dummy_model, "integrity_test", "RandomForest",
        metrics={"accuracy": 0.99},
    )
    assert saved_path and os.path.exists(saved_path), \
        f"Saved path missing: {saved_path}"

    loaded = hub.load_model("integrity_test")
    assert loaded is not None, "ModelHub.load_model returned None"
    logger.info("ModelHub save/load OK.")


# ── TEST 5 — Core module imports ──────────────────────────────────────────────

def test_module_imports():
    """Critical modules must import without errors."""
    import app.strategy.engine   # noqa: F401
    import app.monitor.logger    # noqa: F401
    import app.ml.trainer        # noqa: F401
    import app.llm.news_analyzer # noqa: F401
    logger.info("Core module imports OK.")


# ── TEST 6 — StrategyEngine on_bar ───────────────────────────────────────────

def test_strategy_engine_on_bar():
    """StrategyEngine.on_bar must return a valid signal dict."""
    from app.ml.trainer import Trainer
    from app.strategy.engine import StrategyEngine
    from app.strategy.adaption import Adaptor
    from app.ml.features import make_features
    from app.config import load_config

    cfg     = load_config()
    df      = _make_df(200)
    trainer = Trainer()
    model   = trainer.train(df)
    assert model is not None

    adaptor = Adaptor()
    engine  = StrategyEngine(model, adaptor, cfg)
    feat    = make_features(df)
    X       = feat.drop(
        columns=["timestamp", "open", "high", "low", "close", "volume"],
        errors="ignore",
    )

    signal = engine.on_bar(X.iloc[[0]])
    assert isinstance(signal, dict), "on_bar did not return a dict"
    assert signal["side"] in ("BUY", "SELL", "HOLD"), \
        f"Unexpected side: {signal['side']}"
    logger.info("StrategyEngine on_bar OK: %s", signal)


# ── Entrypoint ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        test_database_connection,
        test_mock_data_insertion,
        test_model_training,
        test_modelhub_operations,
        test_module_imports,
        test_strategy_engine_on_bar,
    ]
    print("\nRunning system integrity checks...\n")
    passed = failed = 0
    for fn in tests:
        print(f"  {fn.__name__} ... ", end="", flush=True)
        try:
            fn()
            print("OK")
            passed += 1
        except Exception as e:
            print(f"FAIL\n    {e}")
            traceback.print_exc()
            failed += 1
    print(f"\n{passed} passed, {failed} failed.")
