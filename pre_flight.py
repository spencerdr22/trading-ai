"""
pre_flight.py — Trading-AI Pre-Flight Check
────────────────────────────────────────────
Run this BEFORE starting tomorrow's session to confirm everything is wired up.

Usage:
    .venv\Scripts\python.exe pre_flight.py

Checks:
  1. Python package imports (streamlit, sklearn, torch, ollama, aiohttp...)
  2. .env credentials present
  3. Alpaca paper account reachable + balance shown
  4. Ollama running + qwen3:8b available
  5. AI Orchestrator reachable (optional)
  6. Database initialises cleanly
  7. Data loader fetches real SPY bars from Alpaca
  8. Model training smoke test
  9. Dashboard file has no syntax errors
"""

import sys
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

# Colour codes — work on Windows 10+ with ANSI enabled
PASS  = "\033[92m  PASS\033[0m"
FAIL  = "\033[91m  FAIL\033[0m"
WARN  = "\033[93m  WARN\033[0m"
INFO  = "\033[96m  INFO\033[0m"
SKIP  = "\033[94m  SKIP\033[0m"

errors = 0


def check(label, fn, optional=False):
    """Run fn(). Treat exceptions and False as failures. Strings as info."""
    global errors
    try:
        result = fn()
        if result is True or result is None:
            print(f"{PASS}  {label}")
        elif result is False:
            tag = WARN if optional else FAIL
            if not optional:
                errors += 1
            print(f"{tag}  {label}")
        else:
            # String result — show as INFO (not a failure)
            print(f"{INFO}  {label}: {result}")
    except Exception as e:
        tag = WARN if optional else FAIL
        if not optional:
            errors += 1
        print(f"{tag}  {label}: {e}")


print("=" * 60)
print("  Trading-AI Pre-Flight Check")
print(f"  Python: {sys.executable}")
print("=" * 60)
print()

# ── 1. Core imports ───────────────────────────────────────────────────────────
print("[ Imports ]")


def _chk_streamlit():
    import streamlit
    v = tuple(int(x) for x in streamlit.__version__.split(".")[:2])
    if v < (1, 35):
        raise RuntimeError(f"version {streamlit.__version__} too old — run install_deps.bat")
    return f"v{streamlit.__version__}"


def _chk_sklearn():
    import sklearn
    return f"v{sklearn.__version__}"


def _chk_torch():
    import torch
    return f"v{torch.__version__}  cuda={torch.cuda.is_available()}"


def _chk_ollama():
    import ollama
    return True


def _chk_aiohttp():
    import aiohttp
    return f"v{aiohttp.__version__}"


def _chk_apscheduler():
    import apscheduler
    return True


def _chk_pytz():
    import pytz
    return True


def _chk_autorefresh():
    import streamlit_autorefresh
    return True


check("streamlit >=1.35",       _chk_streamlit)
check("streamlit_autorefresh",  _chk_autorefresh)
check("pandas",                 lambda: __import__("pandas").__version__)
check("numpy",                  lambda: __import__("numpy").__version__)
check("scikit-learn",           _chk_sklearn)
check("torch",                  _chk_torch)
check("ollama client",          _chk_ollama)
check("aiohttp",                _chk_aiohttp)
check("apscheduler",            _chk_apscheduler)
check("pytz",                   _chk_pytz)
print()

# ── 2. .env credentials ───────────────────────────────────────────────────────
print("[ Environment / .env ]")
from dotenv import load_dotenv
load_dotenv(os.path.join(ROOT, ".env"))


def _chk_alpaca_keys():
    k = os.getenv("ALPACA_API_KEY", "")
    s = os.getenv("ALPACA_SECRET_KEY", "")
    if not k or not s:
        raise RuntimeError("ALPACA_API_KEY / ALPACA_SECRET_KEY missing from .env")
    return f"key ends ...{k[-6:]}"


check("ALPACA keys set",        _chk_alpaca_keys)
check("ALPACA_BASE_URL",        lambda: os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets (default)"))
check("DB_NAME",                lambda: os.getenv("DB_NAME", "trading_ai (SQLite default)"))
check("OLLAMA_BASE_URL",        lambda: os.getenv("OLLAMA_BASE_URL", "http://localhost:11434 (default)"))
check("AI_ORCHESTRATOR_URL",    lambda: os.getenv("AI_ORCHESTRATOR_URL", "http://localhost:8000 (default)"))
print()

# ── 3. Alpaca paper account ───────────────────────────────────────────────────
print("[ Alpaca Paper Account ]")
import requests as _req


def _chk_account():
    key    = os.getenv("ALPACA_API_KEY")
    secret = os.getenv("ALPACA_SECRET_KEY")
    base   = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
    r = _req.get(
        f"{base}/v2/account",
        headers={"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret},
        timeout=8,
    )
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:100]}")
    acct = r.json()
    pv   = float(acct.get("portfolio_value", 0))
    return f"portfolio=${pv:,.2f}  status={acct.get('status', '?')}"


def _chk_clock():
    key    = os.getenv("ALPACA_API_KEY")
    secret = os.getenv("ALPACA_SECRET_KEY")
    base   = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
    r = _req.get(
        f"{base}/v2/clock",
        headers={"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret},
        timeout=8,
    )
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}")
    clk = r.json()
    nxt = str(clk.get("next_open", "?"))[:16]
    return f"market_open={clk.get('is_open')}  next_open={nxt}"


check("Account reachable",      _chk_account)
check("Market clock",           _chk_clock)
print()

# ── 4. Ollama ─────────────────────────────────────────────────────────────────
print("[ Ollama / LLM ]")


def _chk_ollama_server():
    r = _req.get("http://localhost:11434/api/tags", timeout=5)
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}")
    return True


def _chk_ollama_model():
    r = _req.get("http://localhost:11434/api/tags", timeout=5)
    models = [m.get("name", "") for m in r.json().get("models", [])]
    if any("qwen3:8b" in m for m in models):
        return "qwen3:8b found"
    available = [m for m in models if "qwen" in m.lower()]
    if available:
        return f"qwen3:8b not found — nearest: {available[0]}  (run: ollama pull qwen3:8b)"
    raise RuntimeError(f"No qwen model found. Run: ollama pull qwen3:8b  Available: {models}")


check("Ollama server running",  _chk_ollama_server)
check("qwen3:8b model",         _chk_ollama_model)
print()

# ── 5. AI Orchestrator (optional) ────────────────────────────────────────────
print("[ AI Orchestrator (optional) ]")


def _chk_orchestrator():
    url = os.getenv("AI_ORCHESTRATOR_URL", "http://localhost:8000")
    r = _req.get(f"{url}/health", timeout=3)
    if r.status_code != 200:
        return False
    data = r.json()
    return f"running at {url}  active_jobs={data.get('active_jobs', 0)}"


try:
    _req.get("http://localhost:8000/health", timeout=2)
    check("Orchestrator health", _chk_orchestrator, optional=True)
except Exception:
    print(f"{SKIP}  Orchestrator not running — OK, trading uses direct Ollama")
print()

# ── 6. Database ───────────────────────────────────────────────────────────────
print("[ Database ]")


def _chk_db():
    from app.db.init import get_engine
    from app.models.schema import Base
    engine = get_engine()
    Base.metadata.create_all(engine)
    return "SQLite OK"


check("DB init",                _chk_db)
print()

# ── 7. Market data ────────────────────────────────────────────────────────────
print("[ Market Data ]")


def _chk_data():
    from app.data.loader import load_sample
    df = load_sample(force_refresh=True)
    if df is None or len(df) < 100:
        raise RuntimeError(f"only {len(df) if df is not None else 0} rows — insufficient")
    return f"{len(df)} rows fetched from Alpaca (real SPY bars)"


check("load_sample() → Alpaca", _chk_data)
print()

# ── 8. Model training smoke test ──────────────────────────────────────────────
print("[ Model Training ]")


def _chk_train():
    import pandas as pd
    import numpy as np
    from app.ml.trainer import Trainer

    np.random.seed(42)
    n = 300
    price = np.cumsum(np.random.randn(n) * 0.3) + 560.0
    df = pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01", periods=n, freq="min"),
        "open":      price + np.random.randn(n) * 0.1,
        "high":      price + np.abs(np.random.randn(n)) * 0.3,
        "low":       price - np.abs(np.random.randn(n)) * 0.3,
        "close":     price + np.random.randn(n) * 0.1,
        "volume":    np.random.randint(200, 2000, size=n).astype(float),
    })
    trainer = Trainer(model_path="data/models/preflight_test.pkl")
    model = trainer.train(df)
    if model is None:
        raise RuntimeError("trainer.train() returned None")
    import sklearn
    return f"RF trained OK  sklearn={sklearn.__version__}"


check("Trainer smoke test",     _chk_train)
print()

# ── 9. Dashboard syntax ───────────────────────────────────────────────────────
print("[ Dashboard ]")


def _chk_dashboard():
    import py_compile
    path = os.path.join(ROOT, "app", "monitor", "dashboard.py")
    py_compile.compile(path, doraise=True)
    return "syntax OK"


check("dashboard.py syntax",    _chk_dashboard)
print()

# ── Summary ───────────────────────────────────────────────────────────────────
print("=" * 60)
if errors == 0:
    print("\033[92m  ALL CHECKS PASSED — system ready for tomorrow.\033[0m")
    print()
    print("  To start trading run:")
    print("    run_full_system.bat")
    print()
    print("  Or just the dashboard:")
    print("    run_dashboard.bat")
else:
    print(f"\033[91m  {errors} CHECK(S) FAILED — fix the issues above.\033[0m")
    print()
    print("  1. Run install_deps.bat to install missing packages")
    print("  2. Re-run this script to confirm all pass")
print("=" * 60)
sys.exit(0 if errors == 0 else 1)
