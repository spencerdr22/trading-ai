"""
mlflow_tracking.py — Optional MLflow integration.

All MLflow calls are wrapped in try/except so the trading system
continues to run even when the MLflow server is offline.
"""

import os
import contextlib
from typing import Any, Dict, Optional

import numpy as np
from ..monitor.logger import get_logger

logger = get_logger(__name__)

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")

# Lazy-import mlflow so a missing/offline server doesn't crash on import
try:
    import mlflow
    from mlflow.tracking import MlflowClient
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    MLFLOW_AVAILABLE = True
except Exception as _e:
    mlflow = None           # type: ignore
    MlflowClient = None     # type: ignore
    MLFLOW_AVAILABLE = False
    logger.warning("MLflow not available (%s) — tracking disabled.", _e)


def _clean_metric_value(v: Any) -> Optional[float]:
    """Convert value to a float MLflow accepts, or return None to skip."""
    if isinstance(v, bool):
        return float(v)
    if isinstance(v, (int, float)):
        vf = float(v)
        if np.isnan(vf) or np.isinf(vf):
            return None
        return vf
    return None


@contextlib.contextmanager
def start_run(
    experiment_name: str,
    run_name:        str,
    tags:            Optional[Dict[str, Any]] = None,
):
    """
    Context manager for an MLflow run.
    Yields the run object, or a dummy object if MLflow is unavailable.
    """
    if not MLFLOW_AVAILABLE:
        logger.debug("MLflow unavailable — skipping run '%s'.", run_name)
        yield None
        return

    try:
        mlflow.set_experiment(experiment_name)
        logger.info("MLflow: starting run '%s'", run_name)
        with mlflow.start_run(run_name=run_name) as run:
            if tags:
                mlflow.set_tags(tags)
            yield run
        logger.info("MLflow: run complete — %s", run.info.run_id)
    except Exception as e:
        logger.warning("MLflow run failed: %s", e)
        yield None


def log_metrics(metrics: Dict[str, Any]) -> None:
    """Log numeric metrics to the active MLflow run."""
    if not MLFLOW_AVAILABLE:
        return
    try:
        clean = {k: v for k, v in
                 ((k, _clean_metric_value(v)) for k, v in metrics.items())
                 if v is not None}
        if clean:
            mlflow.log_metrics(clean)
            logger.debug("MLflow: logged %d metrics", len(clean))
    except Exception as e:
        logger.warning("MLflow log_metrics failed: %s", e)


def log_params(params: Dict[str, Any]) -> None:
    """Log parameters to the active MLflow run."""
    if not MLFLOW_AVAILABLE:
        return
    try:
        flat = {k: v if isinstance(v, (str, int, float, bool)) else str(v)
                for k, v in params.items()}
        if flat:
            mlflow.log_params(flat)
            logger.debug("MLflow: logged %d params", len(flat))
    except Exception as e:
        logger.warning("MLflow log_params failed: %s", e)


def log_artifact(local_path: str, artifact_path: Optional[str] = None) -> None:
    """Log a file artifact to MLflow."""
    if not MLFLOW_AVAILABLE:
        return
    try:
        if artifact_path:
            mlflow.log_artifact(local_path, artifact_path=artifact_path)
        else:
            mlflow.log_artifact(local_path)
        logger.info("MLflow: logged artifact %s", local_path)
    except Exception as e:
        logger.warning("MLflow log_artifact failed: %s", e)


def get_mlflow_client():
    """Return an MlflowClient, or None if MLflow is unavailable."""
    if not MLFLOW_AVAILABLE:
        return None
    return MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
