#!/usr/bin/env python3
"""
integrate_ml_scaffold.py

Automated integration script for merging ml_project_scaffold 
strengths into trading-ai.

Usage:
    python integrate_ml_scaffold.py --mode [setup|copy|verify|all]
"""

import os
import shutil
import argparse
from pathlib import Path
from typing import List, Tuple


# Paths
TRADING_AI_ROOT = Path(r"C:\Users\spenc\Documents\trading-ai")
ML_SCAFFOLD_ROOT = Path(r"C:\Users\spenc\Documents\ml_project_scaffold")


def log(message: str, level: str = "INFO"):
    """Simple logging."""
    print(f"[{level}] {message}")


def create_directories():
    """Create necessary directories for integration."""
    log("Creating directory structure...")
    
    dirs = [
        TRADING_AI_ROOT / "app" / "validation",
        TRADING_AI_ROOT / "governance",
        TRADING_AI_ROOT / "scripts",
        TRADING_AI_ROOT / "data" / "validation",
        TRADING_AI_ROOT / "data" / "monte_carlo",
    ]
    
    for dir_path in dirs:
        dir_path.mkdir(parents=True, exist_ok=True)
        log(f"  ✓ Created: {dir_path.relative_to(TRADING_AI_ROOT)}")
    
    log("✅ Directory structure created")


def copy_files():
    """Copy files from ml_project_scaffold to trading-ai."""
    log("Copying files from ml_project_scaffold...")
    
    # Files to copy: (source, destination)
    copies: List[Tuple[Path, Path]] = [
        # Walk-forward validation
        (
            ML_SCAFFOLD_ROOT / "utils" / "walk_forward.py",
            TRADING_AI_ROOT / "app" / "validation" / "walk_forward.py"
        ),
        # Monte Carlo
        (
            ML_SCAFFOLD_ROOT / "utils" / "monte_carlo.py",
            TRADING_AI_ROOT / "app" / "validation" / "monte_carlo.py"
        ),
        # VectorBT engine
        (
            ML_SCAFFOLD_ROOT / "backtests" / "vbt_engine.py",
            TRADING_AI_ROOT / "app" / "backtest" / "vbt_engine.py"
        ),
        # Governance documents
        (
            ML_SCAFFOLD_ROOT / "Governance_v3.0.md",
            TRADING_AI_ROOT / "governance" / "Governance_v3.0.md"
        ),
        (
            ML_SCAFFOLD_ROOT / "Audit_Compliance_Report_v3.0.md",
            TRADING_AI_ROOT / "governance" / "Audit_Compliance_Report_v3.0.md"
        ),
    ]
    
    for source, dest in copies:
        if source.exists():
            shutil.copy2(source, dest)
            log(f"  ✓ Copied: {source.name} → {dest.relative_to(TRADING_AI_ROOT)}")
        else:
            log(f"  ⚠ Source not found: {source}", level="WARN")
    
    log("✅ Files copied")


def create_init_files():
    """Create __init__.py files for new packages."""
    log("Creating __init__.py files...")
    
    init_files = {
        TRADING_AI_ROOT / "app" / "validation" / "__init__.py": """\"\"\"
Package: app.validation
Advanced validation utilities for trading-ai.
\"\"\"

from .walk_forward import walk_forward_validation, walk_forward_split
from .monte_carlo import monte_carlo_analysis, bootstrap_trades

__version__ = "1.0.0"
__all__ = [
    "walk_forward_validation",
    "walk_forward_split",
    "monte_carlo_analysis",
    "bootstrap_trades"
]
""",
        TRADING_AI_ROOT / "scripts" / "__init__.py": '"""Scripts package."""\n',
    }
    
    for file_path, content in init_files.items():
        file_path.write_text(content, encoding='utf-8')
        log(f"  ✓ Created: {file_path.relative_to(TRADING_AI_ROOT)}")
    
    log("✅ Init files created")


def create_mlflow_utils():
    """Create adapted MLflow utilities."""
    log("Creating MLflow utilities...")
    
    mlflow_content = '''"""
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
'''
    
    file_path = TRADING_AI_ROOT / "app" / "utils" / "mlflow_tracking.py"
    file_path.write_text(mlflow_content, encoding='utf-8')
    log(f"  ✓ Created: {file_path.relative_to(TRADING_AI_ROOT)}")
    
    log("✅ MLflow utilities created")


def update_requirements():
    """Update requirements.txt with new dependencies."""
    log("Updating requirements.txt...")
    
    req_file = TRADING_AI_ROOT / "requirements.txt"
    
    if req_file.exists():
        content = req_file.read_text()
        
        new_deps = [
            "\n# Advanced backtesting and validation (ml_project_scaffold integration)",
            "vectorbt==0.26.1",
            "mlflow==2.9.2",
        ]
        
        # Check if already added
        if "vectorbt" not in content:
            with open(req_file, 'a') as f:
                f.write('\n'.join(new_deps) + '\n')
            log("  ✓ Added new dependencies")
        else:
            log("  ℹ Dependencies already present")
    else:
        log("  ⚠ requirements.txt not found", level="WARN")
    
    log("✅ Requirements updated")


def update_env_file():
    """Update .env file with MLflow configuration."""
    log("Updating .env file...")
    
    env_file = TRADING_AI_ROOT / ".env"
    
    mlflow_config = """
# MLflow Configuration (ml_project_scaffold integration)
MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_EXPERIMENT_NAME=trading-ai-experiments
"""
    
    if env_file.exists():
        content = env_file.read_text()
        
        if "MLFLOW_TRACKING_URI" not in content:
            with open(env_file, 'a') as f:
                f.write(mlflow_config)
            log("  ✓ Added MLflow configuration")
        else:
            log("  ℹ MLflow config already present")
    else:
        env_file.write_text(mlflow_config)
        log("  ✓ Created .env with MLflow config")
    
    log("✅ .env file updated")


def create_test_file():
    """Create integration test file."""
    log("Creating integration test file...")
    
    test_content = '''"""
Integration tests for ml_project_scaffold features.
"""

import pytest
import pandas as pd
import numpy as np
from app.validation.walk_forward import walk_forward_split
from app.validation.monte_carlo import bootstrap_trades


def test_walk_forward_split():
    """Test walk-forward split generation."""
    splits = walk_forward_split(n_samples=1000, n_splits=5, mode='anchored')
    
    assert len(splits) == 5
    for train, test in splits:
        assert len(set(train) & set(test)) == 0


def test_monte_carlo_bootstrap():
    """Test Monte Carlo bootstrap sampling."""
    np.random.seed(42)
    trades_df = pd.DataFrame({'R': np.random.randn(100)})
    
    sequences = bootstrap_trades(trades_df, n_sequences=10, seed=42)
    
    assert len(sequences) == 10
    assert all(len(seq) == 100 for seq in sequences)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
'''
    
    file_path = TRADING_AI_ROOT / "app" / "tests" / "test_integration.py"
    file_path.write_text(test_content, encoding='utf-8')
    log(f"  ✓ Created: {file_path.relative_to(TRADING_AI_ROOT)}")
    
    log("✅ Integration test created")


def verify_integration():
    """Verify that all files are in place."""
    log("Verifying integration...")
    
    expected_files = [
        "app/validation/__init__.py",
        "app/validation/walk_forward.py",
        "app/validation/monte_carlo.py",
        "app/backtest/vbt_engine.py",
        "app/utils/mlflow_tracking.py",
        "app/tests/test_integration.py",
        "governance/Governance_v3.0.md",
        "governance/Audit_Compliance_Report_v3.0.md",
    ]
    
    missing = []
    for file_rel in expected_files:
        file_path = TRADING_AI_ROOT / file_rel
        if file_path.exists():
            log(f"  ✓ {file_rel}")
        else:
            log(f"  ✗ {file_rel}", level="ERROR")
            missing.append(file_rel)
    
    if missing:
        log(f"❌ {len(missing)} files missing", level="ERROR")
        return False
    else:
        log("✅ All files verified")
        return True


def main():
    parser = argparse.ArgumentParser(
        description="Integrate ml_project_scaffold features into trading-ai"
    )
    parser.add_argument(
        "--mode",
        choices=["setup", "copy", "verify", "all"],
        default="all",
        help="Integration mode"
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("🔧 TRADING-AI INTEGRATION AUTOMATION")
    print("="*60 + "\n")
    
    if args.mode in ["setup", "all"]:
        create_directories()
        create_init_files()
        print()
    
    if args.mode in ["copy", "all"]:
        copy_files()
        create_mlflow_utils()
        create_test_file()
        update_requirements()
        update_env_file()
        print()
    
    if args.mode in ["verify", "all"]:
        success = verify_integration()
        print()
        
        if success:
            print("✅ Integration complete!")
            print("\nNext steps:")
            print("  1. Install new dependencies:")
            print("     pip install vectorbt==0.26.1 mlflow==2.9.2")
            print("\n  2. Run tests:")
            print("     pytest app/tests/test_integration.py -v")
            print("\n  3. Start MLflow UI (optional):")
            print("     mlflow ui --port 5000")
        else:
            print("❌ Integration incomplete. Please review errors above.")
    
    print("\n" + "="*60 + "\n")


if __name__ == "__main__":
    main()