"""
app/training_pipeline.py — Unified Self-Training Pipeline

Runs automatically:
  - Nightly at midnight (via APScheduler in main.py)
  - Hourly intraday retrain (called from forward_mode)
  - On-demand: python -m app.training_pipeline

Pipeline stages:
  1. Load latest market data
  2. Retrain supervised RF model (Trainer)
  3. Run offline RL policy update (ReinforcementLearner)
  4. Run Optuna hyperparameter optimization (RLHyperOptimizer)
  5. Save all models to ModelHub with versioning
  6. Log metrics to DB and rotating log files
  7. Generate performance plots

All stages are wrapped in try/except so a failure in one
never blocks the others.
"""

import os
import time
import traceback
from datetime import datetime, timezone

from .monitor.logger import get_logger
from .data.loader import load_sample
from .ml.trainer import Trainer
from .adaptive.model_hub import ModelHub
from .adaptive.learner import ReinforcementLearner
from .adaptive.optimizer import RLHyperOptimizer
from .db import get_session
from .models.schema import Metric

logger = get_logger("training_pipeline")

# ── Configuration ─────────────────────────────────────────────────────────────
FEATURE_DIM     = 4       # RL state vector size (mean_pnl, std_pnl, win_rate, reward)
RL_EPISODES     = 10      # episodes per nightly RL run
OPTUNA_TRIALS   = 5       # hyperparameter trials per nightly run
MIN_BARS        = 120     # minimum data rows to proceed
# ──────────────────────────────────────────────────────────────────────────────


def _log_metric(name: str, value: float):
    """Persist a scalar metric to the database."""
    try:
        with get_session() as s:
            s.add(Metric(name=name, value=value,
                         timestamp=datetime.now(timezone.utc).replace(tzinfo=None)))
            s.commit()
    except Exception as e:
        logger.warning("Could not persist metric %s: %s", name, e)


def stage_supervised(df) -> float:
    """
    Stage 1 — Supervised model retrain.
    Returns model accuracy or 0.0 on failure.
    """
    logger.info("[Pipeline] Stage 1: Supervised retrain")
    try:
        trainer  = Trainer()
        model    = trainer.train(df)
        if model is None:
            logger.warning("[Pipeline] Supervised train returned None")
            return 0.0
        # Read accuracy from ModelHub metadata (set by Trainer.train)
        hub      = ModelHub()
        meta     = hub.get_model_metadata("supervised_rf") or {}
        accuracy = float(meta.get("accuracy", 0.0))
        _log_metric("pipeline_rf_accuracy", accuracy)
        logger.info("[Pipeline] RF accuracy: %.4f", accuracy)
        return accuracy
    except Exception:
        logger.error("[Pipeline] Supervised stage failed:\n%s", traceback.format_exc())
        return 0.0


def stage_rl(episodes: int = RL_EPISODES) -> float:
    """
    Stage 2 — Offline RL policy update.
    Returns reward score or 0.0 on failure.
    """
    logger.info("[Pipeline] Stage 2: RL policy update (%d episodes)", episodes)
    try:
        learner = ReinforcementLearner(feature_dim=FEATURE_DIM)
        learner.load_latest_policy()
        result = learner.train_from_history(episodes=episodes)
        if not result:
            logger.warning("[Pipeline] RL returned no result (insufficient trade history)")
            return 0.0
        reward = result.get("reward", 0.0)
        _log_metric("pipeline_rl_reward", reward)
        logger.info("[Pipeline] RL reward: %.4f", reward)
        return reward
    except Exception:
        logger.error("[Pipeline] RL stage failed:\n%s", traceback.format_exc())
        return 0.0


def stage_hparam(trials: int = OPTUNA_TRIALS) -> dict:
    """
    Stage 3 — Optuna hyperparameter search.
    Returns best params dict or {} on failure.
    """
    logger.info("[Pipeline] Stage 3: Hyperparameter search (%d trials)", trials)
    try:
        optimizer = RLHyperOptimizer(feature_dim=FEATURE_DIM)
        best_params, best_score = optimizer.optimize(n_trials=trials)
        _log_metric("pipeline_optuna_score", best_score)
        logger.info("[Pipeline] Best Optuna score: %.4f | params: %s", best_score, best_params)
        return best_params
    except Exception:
        logger.error("[Pipeline] Hparam stage failed:\n%s", traceback.format_exc())
        return {}


def run_pipeline(
    run_rl: bool = True,
    run_hparam: bool = True,
    rl_episodes: int = RL_EPISODES,
    optuna_trials: int = OPTUNA_TRIALS,
) -> dict:
    """
    Run the full self-training pipeline.

    Args:
        run_rl:       Include RL policy update stage
        run_hparam:   Include Optuna hyperparameter stage
        rl_episodes:  RL episodes to run
        optuna_trials: Optuna trials to run

    Returns:
        Summary dict with results from each stage
    """
    start = time.time()
    logger.info("=" * 55)
    logger.info("  SELF-TRAINING PIPELINE STARTED  %s",
                datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"))
    logger.info("=" * 55)

    summary = {
        "timestamp":  datetime.now(timezone.utc).isoformat(),
        "rf_accuracy": 0.0,
        "rl_reward":   0.0,
        "best_params": {},
        "elapsed_sec": 0.0,
        "status":      "incomplete",
    }

    # Load data
    df = load_sample()
    if df is None or len(df) < MIN_BARS:
        logger.warning("[Pipeline] Not enough data (%s bars) — aborting",
                       len(df) if df is not None else 0)
        summary["status"] = "no_data"
        return summary

    logger.info("[Pipeline] Data loaded: %d bars", len(df))

    # Stage 1 — supervised
    summary["rf_accuracy"] = stage_supervised(df)

    # Stage 2 — RL
    if run_rl:
        summary["rl_reward"] = stage_rl(episodes=rl_episodes)

    # Stage 3 — hyperparameters
    if run_hparam:
        summary["best_params"] = stage_hparam(trials=optuna_trials)

    elapsed = time.time() - start
    summary["elapsed_sec"] = round(elapsed, 1)
    summary["status"] = "complete"

    logger.info("=" * 55)
    logger.info("  PIPELINE COMPLETE in %.1fs", elapsed)
    logger.info("  RF accuracy : %.4f", summary["rf_accuracy"])
    logger.info("  RL reward   : %.4f", summary["rl_reward"])
    logger.info("=" * 55)

    return summary


def quick_retrain() -> float:
    """
    Lightweight intraday retrain — supervised model only, no RL/Optuna.
    Called from forward_mode every retrain_interval bars.
    Returns RF accuracy.
    """
    logger.info("[Pipeline] Quick intraday retrain...")
    df = load_sample()
    if df is None or len(df) < MIN_BARS:
        logger.warning("[Pipeline] Quick retrain skipped — insufficient data")
        return 0.0
    return stage_supervised(df)


# ── CLI entrypoint ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Run the self-training pipeline")
    p.add_argument("--no-rl",     action="store_true", help="Skip RL stage")
    p.add_argument("--no-hparam", action="store_true", help="Skip Optuna stage")
    p.add_argument("--episodes",  type=int, default=RL_EPISODES)
    p.add_argument("--trials",    type=int, default=OPTUNA_TRIALS)
    args = p.parse_args()

    result = run_pipeline(
        run_rl=not args.no_rl,
        run_hparam=not args.no_hparam,
        rl_episodes=args.episodes,
        optuna_trials=args.trials,
    )
    print("\nPipeline result:", result)
