"""
autostart.py — Full system auto-launcher for Trading-AI
────────────────────────────────────────────────────────
This is the single script that Task Scheduler runs every weekday at 9:20 AM.
It starts every component in the correct order with health checks, then
hands off to start_trading.py which waits for 9:30 AM ET market open.

Components started (in order):
  1. AI Orchestrator via Docker  (port 8000)
  2. Ollama server with GPU settings  (port 11434)
  3. Ollama priority arbiter  (port 11435)
  4. Streamlit dashboard  (port 8501)
  5. Trading scheduler (start_trading.py) — waits for 9:30 AM ET

On shutdown (CTRL+C or end of session):
  - Alpaca positions are flattened automatically by start_trading.py
  - Subprocesses are terminated cleanly

Usage (Task Scheduler calls this directly):
    C:\Users\spenc\Documents\trading-ai\.venv\Scripts\python.exe autostart.py
"""

import os
import sys
import subprocess
import time
import signal
import logging
import requests
from datetime import datetime

# ── Force UTF-8 on Windows console ───────────────────────────────────────────
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT       = os.path.dirname(os.path.abspath(__file__))
VENV_PY    = os.path.join(ROOT, ".venv", "Scripts", "python.exe")
VENV_ST    = os.path.join(ROOT, ".venv", "Scripts", "streamlit.exe")
ARBITER_PY = r"C:\Users\spenc\Documents\ollama_arbiter.py"
ORCH_DIR   = r"C:\Users\spenc\Documents\ai-orchestrator"

# ── Logging ───────────────────────────────────────────────────────────────────
os.makedirs(os.path.join(ROOT, "data"), exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(
            os.path.join(ROOT, "data", "autostart.log"),
            encoding="utf-8",
        ),
    ],
)
log = logging.getLogger("autostart")

# Track subprocesses so we can clean up on exit
_procs: list[subprocess.Popen] = []


def _kill_all():
    for p in _procs:
        try:
            if p.poll() is None:
                p.terminate()
        except Exception:
            pass


def _handle_signal(sig, frame):
    log.info("Shutdown signal received — cleaning up...")
    _kill_all()
    sys.exit(0)


signal.signal(signal.SIGINT,  _handle_signal)
signal.signal(signal.SIGTERM, _handle_signal)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _ping(url: str, timeout: int = 3) -> bool:
    try:
        r = requests.get(url, timeout=timeout)
        return r.status_code == 200
    except Exception:
        return False


def _wait_for(url: str, label: str, max_wait: int = 30) -> bool:
    """Poll url until it responds or max_wait seconds pass."""
    for i in range(max_wait):
        if _ping(url):
            log.info("%s is up.", label)
            return True
        time.sleep(1)
    log.warning("%s did not respond within %ds.", label, max_wait)
    return False


def _kill_port(port: int):
    """Kill whatever is listening on a TCP port (Windows)."""
    try:
        result = subprocess.run(
            f'netstat -aon | findstr ":{port} "',
            shell=True, capture_output=True, text=True,
        )
        for line in result.stdout.strip().splitlines():
            parts = line.split()
            if len(parts) >= 5:
                pid = parts[-1]
                try:
                    subprocess.run(f"taskkill /F /PID {pid}", shell=True,
                                   capture_output=True)
                except Exception:
                    pass
    except Exception:
        pass


# ── Step 1: AI Orchestrator ───────────────────────────────────────────────────

def start_orchestrator() -> bool:
    log.info("[1/5] AI Orchestrator (port 8000)...")

    if _ping("http://localhost:8000/health"):
        log.info("      Orchestrator already running.")
        return True

    # Check Docker is available
    docker_check = subprocess.run("docker info", shell=True, capture_output=True)
    if docker_check.returncode != 0:
        log.warning("      Docker not running — skipping orchestrator. Trading will use direct Ollama.")
        return False

    log.info("      Starting via Docker Compose...")
    p = subprocess.Popen(
        "docker compose -f docker\\docker-compose.yml up --remove-orphans",
        shell=True,
        cwd=ORCH_DIR,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    _procs.append(p)

    up = _wait_for("http://localhost:8000/health", "Orchestrator", max_wait=30)
    if up:
        log.info("      Orchestrator ready on port 8000.")
    else:
        log.warning("      Orchestrator not ready — trading will use direct Ollama.")
    return up


# ── Step 2: Ollama ─────────────────────────────────────────────────────────────

def start_ollama() -> bool:
    log.info("[2/5] Ollama server (port 11434)...")

    # Kill stale Ollama instances
    subprocess.run("taskkill /F /IM ollama.exe", shell=True, capture_output=True)
    subprocess.run("taskkill /F /IM ollama_llama_server.exe", shell=True,
                   capture_output=True)
    time.sleep(2)

    # Set GPU environment before launching
    env = os.environ.copy()
    env.update({
        "OLLAMA_FLASH_ATTENTION":  "1",
        "OLLAMA_GPU_OVERHEAD":     "0",
        "OLLAMA_NUM_PARALLEL":     "1",
        "OLLAMA_KEEP_ALIVE":       "10m",
        "OLLAMA_MAX_LOADED_MODELS": "1",
    })

    p = subprocess.Popen(
        "ollama serve",
        shell=True,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    _procs.append(p)

    up = _wait_for("http://localhost:11434/api/tags", "Ollama", max_wait=20)
    if not up:
        log.error("      Ollama failed to start — trading cannot proceed without it.")
        return False

    log.info("      Ollama ready. FLASH_ATTENTION=1 NUM_PARALLEL=1 KEEP_ALIVE=10m")
    return True


# ── Step 3: Ollama arbiter ────────────────────────────────────────────────────

def start_arbiter():
    log.info("[3/5] Ollama priority arbiter (port 11435)...")
    _kill_port(11435)
    time.sleep(1)

    if not os.path.exists(ARBITER_PY):
        log.warning("      Arbiter script not found at %s — skipping.", ARBITER_PY)
        return

    p = subprocess.Popen(
        [VENV_PY, ARBITER_PY],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    _procs.append(p)
    time.sleep(3)

    if _ping("http://localhost:11435/api/tags"):
        log.info("      Arbiter running on port 11435.")
    else:
        log.info("      Arbiter starting in background.")


# ── Step 4: Streamlit dashboard ───────────────────────────────────────────────

def start_dashboard():
    log.info("[4/5] Streamlit dashboard (port 8501)...")

    if not os.path.exists(VENV_ST):
        log.warning("      streamlit.exe not found — skipping dashboard.")
        return

    p = subprocess.Popen(
        [
            VENV_ST, "run",
            os.path.join(ROOT, "app", "monitor", "dashboard.py"),
            "--server.port", "8501",
            "--server.headless", "true",
            "--browser.gatherUsageStats", "false",
            "--server.fileWatcherType", "none",
        ],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    _procs.append(p)
    time.sleep(4)
    log.info("      Dashboard starting at http://localhost:8501")


# ── Step 5: Trading scheduler ─────────────────────────────────────────────────

def start_trading():
    log.info("[5/5] Trading scheduler (start_trading.py)...")
    log.info("      Will wait for 9:30 AM ET then begin paper trading.")

    # Run in foreground so autostart.py stays alive and monitors it
    p = subprocess.Popen(
        [VENV_PY, os.path.join(ROOT, "start_trading.py")],
        cwd=ROOT,
        stdout=sys.stdout,
        stderr=sys.stderr,
    )
    _procs.append(p)
    return p


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    log.info("=" * 60)
    log.info("  Trading-AI Auto-Launcher  —  %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    log.info("=" * 60)

    # Step 1 — Orchestrator (optional)
    start_orchestrator()

    # Step 2 — Ollama (required)
    if not start_ollama():
        log.error("Cannot start without Ollama. Exiting.")
        sys.exit(1)

    # Step 3 — Dashboard (was arbiter, now handled by AI Orchestrator)
    start_dashboard()

    # Step 4 — Trading (blocks until end of session)
    trading_proc = start_trading()

    log.info("=" * 60)
    log.info("  All components started. Monitoring trading process.")
    log.info("  Dashboard: http://localhost:8501")
    log.info("  Orchestrator: http://localhost:8000")
    log.info("=" * 60)

    # Wait for trading process to finish (3:55 PM ET EOD close)
    trading_proc.wait()
    log.info("Trading session ended (returncode=%d).", trading_proc.returncode)

    # Give dashboard a moment to show final state
    time.sleep(5)

    log.info("Auto-launcher complete. Goodbye.")
    _kill_all()


if __name__ == "__main__":
    main()
