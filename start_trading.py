"""
start_trading.py — Auto-scheduler for MES paper trading via Alpaca.

Waits until 9:30 AM Eastern Time (market open), then launches forward_mode
with the --alpaca flag.  Runs until 4:00 PM ET, then closes all positions
and exits cleanly.

Usage (run this tonight before bed, or add to Windows Task Scheduler):
    python start_trading.py

Options (edit constants below):
    SYMBOL          — futures symbol to trade (MES)
    START_TIME_ET   — when to start (default 09:30)
    STOP_TIME_ET    — when to stop and flatten (default 15:55, 5 min before close)
    BAR_SLEEP_SEC   — seconds between bars in forward loop
"""

import subprocess
import sys
import time
import logging
import os
import signal
from datetime import datetime, time as dt_time
import pytz

# Force UTF-8 on Windows console so log messages don't crash
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# ── Configuration ────────────────────────────────────────────────────────────
SYMBOL          = "MES"
START_TIME_ET   = dt_time(9, 30, 0)    # 9:30 AM ET — regular market open
STOP_TIME_ET    = dt_time(15, 55, 0)   # 3:55 PM ET — flatten 5 min before close
CHECK_INTERVAL  = 10                    # seconds between clock checks while waiting
ET_ZONE         = pytz.timezone("America/New_York")
PROJECT_ROOT    = os.path.dirname(os.path.abspath(__file__))
PYTHON          = sys.executable        # same venv Python that runs this script
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(
            os.path.join(PROJECT_ROOT, "data", "scheduler.log"),
            encoding="utf-8"
        ),
    ]
)
log = logging.getLogger("scheduler")

_trading_proc = None   # subprocess handle


def now_et() -> datetime:
    return datetime.now(ET_ZONE)


def wait_for_market_open():
    """Block until 9:30 AM ET on the next trading day."""
    while True:
        current = now_et()
        current_time = current.time()
        weekday = current.weekday()  # 0=Mon … 4=Fri, 5=Sat, 6=Sun

        # Skip weekends
        if weekday >= 5:
            log.info("Weekend — waiting for Monday open...")
            time.sleep(3600)
            continue

        if current_time >= STOP_TIME_ET:
            # Past today's close — wait until tomorrow
            log.info("Past today's close — waiting until tomorrow's open...")
            time.sleep(3600)
            continue

        if current_time < START_TIME_ET:
            # Before open — calculate and log wait
            open_dt = current.replace(
                hour=START_TIME_ET.hour,
                minute=START_TIME_ET.minute,
                second=0, microsecond=0
            )
            wait_secs = (open_dt - current).total_seconds()
            mins, secs = divmod(int(wait_secs), 60)
            hours, mins = divmod(mins, 60)
            log.info(
                "Market opens in %02d:%02d:%02d  (at %s ET)",
                hours, mins, secs, open_dt.strftime("%H:%M")
            )
            time.sleep(min(CHECK_INTERVAL, wait_secs))
            continue

        # It's trading hours — break out
        log.info("MARKET OPEN: Starting trading at %s ET", current.strftime("%H:%M:%S"))
        return


def start_forward_mode():
    """Launch forward_mode as a subprocess with --alpaca flag."""
    global _trading_proc
    cmd = [
        PYTHON, "-m", "app.main",
        "--mode",   "forward",
        "--symbol", SYMBOL,
        "--alpaca",
    ]
    log.info("Launching: %s", " ".join(cmd))
    _trading_proc = subprocess.Popen(
        cmd,
        cwd=PROJECT_ROOT,
        stdout=sys.stdout,
        stderr=sys.stderr,
    )
    return _trading_proc


def stop_trading():
    """Gracefully terminate the trading subprocess."""
    global _trading_proc
    if _trading_proc and _trading_proc.poll() is None:
        log.info("Sending SIGTERM to trading process (pid=%d)...", _trading_proc.pid)
        _trading_proc.terminate()
        try:
            _trading_proc.wait(timeout=15)
            log.info("Trading process stopped cleanly.")
        except subprocess.TimeoutExpired:
            log.warning("Process did not exit — killing.")
            _trading_proc.kill()
    _trading_proc = None


def flatten_positions():
    """Close all Alpaca positions at end of day."""
    try:
        # Import here so we only need it at EOD
        sys.path.insert(0, PROJECT_ROOT)
        from app.execution.alpaca_paper import get_alpaca_client
        client = get_alpaca_client()
        client.cancel_all_orders()
        client.close_all_positions()
        log.info("EOD: all positions closed via Alpaca.")
    except Exception as e:
        log.error("EOD flatten failed: %s", e)


def handle_shutdown(sig, frame):
    """CTRL+C / system signal handler."""
    log.info("SHUTDOWN: stopping trading and flattening positions...")
    stop_trading()
    flatten_positions()
    sys.exit(0)


signal.signal(signal.SIGINT,  handle_shutdown)
signal.signal(signal.SIGTERM, handle_shutdown)


# ── Single-instance lock ──────────────────────────────────────────────────────
LOCK_FILE = os.path.join(PROJECT_ROOT, "data", ".scheduler.lock")

def _acquire_lock():
    """Prevent two scheduler instances running simultaneously."""
    os.makedirs(os.path.dirname(LOCK_FILE), exist_ok=True)
    if os.path.exists(LOCK_FILE):
        try:
            with open(LOCK_FILE) as f:
                old_pid = int(f.read().strip())
            # Check if that PID is still alive
            import ctypes
            handle = ctypes.windll.kernel32.OpenProcess(0x400, False, old_pid)
            if handle:
                ctypes.windll.kernel32.CloseHandle(handle)
                log.warning(
                    "Another scheduler is already running (pid=%d). Exiting.", old_pid
                )
                sys.exit(0)
        except Exception:
            pass  # stale lock — overwrite it
    with open(LOCK_FILE, "w") as f:
        f.write(str(os.getpid()))

def _release_lock():
    try:
        os.remove(LOCK_FILE)
    except Exception:
        pass


def main():
    _acquire_lock()
    log.info("=" * 60)
    log.info("  MES Paper Trading Scheduler")
    log.info("  Symbol  : %s  (via SPY proxy on Alpaca paper)", SYMBOL)
    log.info("  Start   : %s ET", START_TIME_ET.strftime("%H:%M"))
    log.info("  Stop    : %s ET", STOP_TIME_ET.strftime("%H:%M"))
    log.info("=" * 60)

    # Ensure data dir exists for log file
    os.makedirs(os.path.join(PROJECT_ROOT, "data"), exist_ok=True)

    try:
        wait_for_market_open()
        proc = start_forward_mode()

        # Monitor until stop time or process exits
        while True:
            current_time = now_et().time()

            if current_time >= STOP_TIME_ET:
                log.info("%s ET -- closing time, stopping trading.",
                         STOP_TIME_ET.strftime("%H:%M"))
                stop_trading()
                flatten_positions()
                log.info("Session complete. Goodbye.")
                break

            if proc.poll() is not None:
                log.warning("Trading process exited early (code=%d).",
                            proc.returncode)
                flatten_positions()
                break

            time.sleep(CHECK_INTERVAL)
    finally:
        _release_lock()


if __name__ == "__main__":
    main()
