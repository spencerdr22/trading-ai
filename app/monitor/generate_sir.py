"""
Module: generate_sir.py
Author: Adaptive Framework Generator
Version: 1.3

Description:
    Extended Operational Integrity & Performance Report (OIPR)
    with Historical Drift Detection, Regression Tracking,
    and Automated Runtime Anomaly Detection.

Features:
    - Historical comparison to previous OIPR report
    - Detects performance degradation, test regressions, and accuracy drift
    - Scans logs for runtime anomalies (e.g., AttributeError, Traceback)
    - Produces both JSON and Markdown artifacts
"""

import re
import os
import json
import psutil
import time
import datetime
import platform
import subprocess
import logging
from pathlib import Path
from statistics import mean
from ..monitor.logger import get_logger

logger = get_logger(__name__)

# ======================================================================
# UTILITIES
# ======================================================================

def run_pytest_summary():
    """Run pytest quietly and extract basic pass/fail/warning counts."""
    start_time = time.time()
    try:
        result = subprocess.run(
            ["pytest", "-q", "--disable-warnings"],
            capture_output=True, text=True, check=False
        )
        duration = round(time.time() - start_time, 2)
        passed = result.stdout.count("PASSED")
        failed = result.stdout.count("FAILED")
        warnings = result.stdout.lower().count("warning")
        return {"passed": passed, "failed": failed, "warnings": warnings, "duration": duration}
    except Exception as e:
        logger.warning(f"OIPR: pytest summary failed: {e}")
        return {"passed": 0, "failed": 0, "warnings": 0, "duration": 0}


def collect_metadata():
    """Collect system-level environment information."""
    return {
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "system": platform.system(),
        "release": platform.release(),
        "python_version": platform.python_version(),
        "cpu_cores": psutil.cpu_count(logical=True),
        "memory_gb": round(psutil.virtual_memory().total / (1024 ** 3), 2),
        "framework_version": "Trading-AI v1.3"
    }

# ======================================================================
# PERFORMANCE PARSER
# ======================================================================

def extract_performance_from_logs(log_file="data/logs/training.log"):
    metrics = {"model_type": None, "accuracy": None, "training_runtime": None}
    log_path = Path(log_file)

    if not log_path.exists():
        logger.warning(f"OIPR: No log file found at {log_path}")
        return metrics

    try:
        with open(log_path, "r", encoding="utf-8") as f:
            lines = f.readlines()[-500:]
        for line in lines:
            if "Training model_type=" in line:
                match = re.search(r"model_type=(\w+)", line)
                if match:
                    metrics["model_type"] = match.group(1)
            if "accuracy" in line.lower():
                match = re.search(r"accuracy:\s*([0-9.]+)", line)
                if match:
                    metrics["accuracy"] = float(match.group(1))
    except Exception as e:
        logger.warning(f"OIPR: Failed to parse training logs: {e}")

    return metrics


def gather_resource_usage():
    process = psutil.Process(os.getpid())
    cpu_usage = psutil.cpu_percent(interval=1)
    mem_usage = process.memory_info().rss / (1024 ** 2)
    return {"cpu_usage_pct": cpu_usage, "memory_usage_mb": round(mem_usage, 2)}

# ======================================================================
# ANOMALY DETECTION
# ======================================================================

def collect_runtime_anomalies(log_dir="logs", pattern=r"(ERROR|Traceback|Exception)"):
    """
    Scan recent log files for runtime anomalies and errors.
    Returns a list of detected lines.
    """
    anomalies = []
    path = Path(log_dir)
    if not path.exists():
        logger.info(f"OIPR: No log directory found at {log_dir}")
        return anomalies

    for log_file in path.glob("*.log"):
        try:
            with open(log_file, "r", encoding="utf-8", errors="ignore") as fh:
                for line in fh:
                    if re.search(pattern, line):
                        anomalies.append(line.strip())
        except Exception as e:
            logger.warning(f"OIPR: Failed to scan {log_file}: {e}")

    return anomalies

# ======================================================================
# HISTORICAL COMPARISON
# ======================================================================

def load_previous_oipr(report_dir="reports/oipr"):
    path = Path(report_dir)
    if not path.exists():
        return None
    reports = sorted(path.glob("oipr_*.json"), key=os.path.getmtime)
    if not reports:
        return None
    last_report = reports[-1]
    try:
        with open(last_report, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"OIPR: Failed to load previous report: {e}")
        return None


def compute_historical_deltas(current, previous):
    """Compare current OIPR with the previous report."""
    deltas = {}
    try:
        if previous:
            prev_acc = previous.get("performance", {}).get("accuracy") or 0
            prev_score = previous.get("integrity_score", 100)
            curr_acc = current["performance"].get("accuracy") or 0
            curr_score = current["integrity_score"]

            deltas["accuracy_delta"] = round(curr_acc - prev_acc, 4)
            deltas["integrity_delta"] = curr_score - prev_score
        else:
            deltas = {"accuracy_delta": 0, "integrity_delta": 0}
    except Exception as e:
        logger.warning(f"OIPR: Delta computation failed: {e}")
        deltas = {"accuracy_delta": 0, "integrity_delta": 0}
    return deltas


def detect_regressions(oipr, deltas):
    """Flag possible regressions or drift issues."""
    findings = []

    if deltas["accuracy_delta"] < -0.02:
        findings.append("⚠️ Model accuracy has decreased significantly.")
    elif deltas["accuracy_delta"] > 0.02:
        findings.append("✅ Model accuracy has improved.")

    if oipr["test_summary"]["failed"] > 0:
        findings.append("❌ One or more tests failed.")
    elif deltas["integrity_delta"] < 0:
        findings.append("⚠️ Integrity score declined slightly.")
    else:
        findings.append("✅ All systems performing as expected.")

    if oipr.get("anomaly_count", 0) > 0:
        findings.append(f"⚠️ {oipr['anomaly_count']} runtime anomaly(s) detected in logs.")

    return findings

# ======================================================================
# BUILD OIPR
# ======================================================================

def build_oipr_dict(test_summary, perf_metrics, system_metrics, previous=None):
    integrity_score = 100
    if test_summary["failed"] > 0:
        integrity_score -= 15
    if test_summary["warnings"] > 10:
        integrity_score -= 5

    # Collect anomalies from logs
    anomalies = collect_runtime_anomalies()
    anomaly_count = len(anomalies)

    oipr = {
        "metadata": system_metrics,
        "test_summary": test_summary,
        "performance": perf_metrics,
        "runtime_resources": gather_resource_usage(),
        "integrity_score": integrity_score,
        "status": "STABLE" if integrity_score == 100 else "DEGRADED",
        "runtime_anomalies": anomalies,
        "anomaly_count": anomaly_count
    }

    # Historical delta calculation
    deltas = compute_historical_deltas(oipr, previous)
    findings = detect_regressions(oipr, deltas)
    oipr["deltas"] = deltas
    oipr["findings"] = findings

    return oipr

# ======================================================================
# SAVE REPORTS
# ======================================================================

def save_oipr_reports(oipr):
    out_dir = Path("reports/oipr")
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / f"oipr_{oipr['metadata']['timestamp'].replace(':','-')}.json"
    with open(json_path, "w") as f:
        json.dump(oipr, f, indent=4)

    # Markdown generation
    template_path = Path("app/monitor/templates/oipr_template.md")
    with open(template_path, "r", encoding="utf-8") as f:
        template = f.read()

    delta_acc = f"{oipr['deltas']['accuracy_delta']:+.4f}"
    delta_int = f"{oipr['deltas']['integrity_delta']:+.1f}"

    md_filled = template.format(
        DATE=oipr["metadata"]["timestamp"],
        STATUS=oipr["status"],
        SCORE=oipr["integrity_score"],
        MODEL=oipr["performance"]["model_type"] or "Unknown",
        ACC=oipr["performance"]["accuracy"] or "N/A",
        ACC_DELTA=delta_acc,
        SCORE_DELTA=delta_int,
        CPU=oipr["runtime_resources"]["cpu_usage_pct"],
        MEM=oipr["runtime_resources"]["memory_usage_mb"],
        PASSED=oipr["test_summary"]["passed"],
        FAILED=oipr["test_summary"]["failed"],
        WARNINGS=oipr["test_summary"]["warnings"],
        FINDINGS="\n".join(oipr["findings"]),
        ANOMALY_COUNT=oipr.get("anomaly_count", 0)
    )

    md_output = out_dir / "latest_OIPR.md"
    with open(md_output, "w", encoding="utf-8") as f:
        f.write(md_filled)

    logger.info(f"OIPR: Reports generated → {json_path}, {md_output}")

# ======================================================================
# MAIN
# ======================================================================

def main():
    logger.info("Generating Enhanced Operational Integrity and Performance Report (OIPR+)...")
    tests = run_pytest_summary()
    perf = extract_performance_from_logs()
    meta = collect_metadata()
    previous = load_previous_oipr()
    oipr = build_oipr_dict(tests, perf, meta, previous)
    save_oipr_reports(oipr)
    logger.info(f"OIPR+ complete. Status: {oipr['status']} | Δ Accuracy={oipr['deltas']['accuracy_delta']:+.4f} | Δ Integrity={oipr['deltas']['integrity_delta']:+.1f} | Anomalies={oipr['anomaly_count']}")

if __name__ == "__main__":
    main()
