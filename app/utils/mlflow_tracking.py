"""
MLflow tracking integration for Trading-AI system.
Adapted from ml_project_scaffold.
"""

import os
import contextlib
from typing import Any, Dict, Optional

import mlflow
import numpy as np
from mlflow.tracking import MlflowClient
from ..monitor.logger import get_logger

logger = get_logger(__name__)

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


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
    run_name: str,
    tags: Optional[Dict[str, Any]] = None,
):
    """Context manager for MLflow runs."""
    mlflow.set_experiment(experiment_name)
    logger.info(f"MLflow: Starting run '{run_name}'")
    
    with mlflow.start_run(run_name=run_name) as run:
        if tags:
            mlflow.set_tags(tags)
        yield run
    
    logger.info(f"MLflow: Run completed - {run.info.run_id}")


def log_metrics(metrics: Dict[str, Any]) -> None:
    """Log numeric metrics to MLflow."""
    clean: Dict[str, float] = {}
    for k, v in metrics.items():
        mv = _clean_metric_value(v)
        if mv is not None:
            clean[k] = mv
    
    if clean:
        mlflow.log_metrics(clean)
        logger.debug(f"MLflow: Logged {len(clean)} metrics")


def log_params(params: Dict[str, Any]) -> None:
    """Log parameters to MLflow."""
    flat: Dict[str, Any] = {}
    for k, v in params.items():
        if isinstance(v, (str, int, float, bool)):
            flat[k] = v
        else:
            flat[k] = str(v)
    
    if flat:
        mlflow.log_params(flat)
        logger.debug(f"MLflow: Logged {len(flat)} parameters")


def log_artifact(local_path: str, artifact_path: Optional[str] = None) -> None:
    """Log artifact file to MLflow."""
    try:
        if artifact_path:
            mlflow.log_artifact(local_path, artifact_path=artifact_path)
        else:
            mlflow.log_artifact(local_path)
        logger.info(f"MLflow: Logged artifact {local_path}")
    except Exception as e:
        logger.error(f"MLflow: Failed to log artifact: {e}")


def get_mlflow_client() -> MlflowClient:
    """Return MlflowClient for advanced operations."""
    return MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
