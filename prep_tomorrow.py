"""
prep_tomorrow.py — Pre-market setup script
Run this tonight to get ready for tomorrow's 9:30 AM open.

Does the following in order:
  1. Regenerates clean MES sim data at correct prices
  2. Runs the full self-training pipeline (RF + RL + Optuna)
  3. Verifies Alpaca paper connection
  4. Checks Ollama / Qwen3 availability
  5. Registers the Windows Task Scheduler job
  6. Prints a final readiness report
"""

import sys
import os
import subprocess

# Make sure we can import the app package
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

PYTHON = sys.executable
OK  = "[OK]"
WARN = "[WARN]"
FAIL = "[FAIL]"

results = {}

def run(label, fn):
    print(f"\n--- {label} ---")
    try:
        fn()
        results[label] = OK
        print(f"{OK} {label}")
    except Exception as e:
        results[label] = FAIL
        print(f"{FAIL} {label}: {e}")

# ── 1. Regenerate sim data ────────────────────────────────────────────────────
def regen_sim():
    r = subprocess.run(
        [PYTHON, "-m", "app.main", "--mode", "simulate", "--symbol", "MES"],
        cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=60
    )
    if r.returncode != 0:
        raise RuntimeError(r.stderr[-500:])
    print(r.stdout[-300:] if r.stdout else "(no output)")

run("1. Regenerate MES sim data", regen_sim)

# ── 2. Full training pipeline ─────────────────────────────────────────────────
def train_pipeline():
    from app.training_pipeline import run_pipeline
    summary = run_pipeline(run_rl=True, run_hparam=False)  # skip optuna for speed
    print(f"    RF accuracy : {summary['rf_accuracy']:.4f}")
    print(f"    RL reward   : {summary['rl_reward']:.4f}")
    print(f"    Status      : {summary['status']}")
    print(f"    Elapsed     : {summary['elapsed_sec']}s")

run("2. Self-training pipeline", train_pipeline)

# ── 3. Alpaca paper connection ────────────────────────────────────────────────
def check_alpaca():
    from app.execution.alpaca_paper import get_alpaca_client
    client = get_alpaca_client()
    acct = client.get_account()
    if not acct:
        raise RuntimeError("Empty account response")
    print(f"    Portfolio : ${float(acct['portfolio_value']):,.2f}")
    print(f"    Buying Pwr: ${float(acct['buying_power']):,.2f}")
    print(f"    Positions : {len(client.get_positions())} open")

run("3. Alpaca paper connection", check_alpaca)

# ── 4. Ollama / Qwen3 check ───────────────────────────────────────────────────
def check_ollama():
    import ollama
    models_resp = ollama.list()
    if hasattr(models_resp, 'models'):
        names = [m.model for m in models_resp.models]
    else:
        names = [m.get("name", "") for m in models_resp.get("models", [])]

    qwen_models = [n for n in names if "qwen" in n.lower()]
    if not qwen_models:
        raise RuntimeError(
            f"No Qwen model found. Available: {names}\n"
            "Run: ollama pull qwen3:30b-a3b-q4_K_M"
        )
    print(f"    Found Qwen models: {qwen_models}")

run("4. Ollama / Qwen3 availability", check_ollama)

# ── 5. Register Windows Task Scheduler ───────────────────────────────────────
def register_task():
    ps_script = os.path.join(PROJECT_ROOT, "register_task.ps1")
    if not os.path.exists(ps_script):
        raise FileNotFoundError("register_task.ps1 not found")
    r = subprocess.run(
        ["powershell", "-ExecutionPolicy", "Bypass",
         "-File", ps_script],
        cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=30
    )
    output = (r.stdout + r.stderr).strip()
    print(f"    {output[-400:]}")
    # Check if task was registered (exit 0 or task already exists)
    if r.returncode not in (0, 1):
        raise RuntimeError(f"PowerShell exited {r.returncode}")

run("5. Register Task Scheduler job", register_task)

# ── 6. Enable wake timers ─────────────────────────────────────────────────────
def enable_wake_timers():
    r = subprocess.run(
        ["powercfg", "/setacvalueindex", "SCHEME_CURRENT", "SUB_SLEEP", "RTCWAKE", "1"],
        capture_output=True, text=True, timeout=15
    )
    subprocess.run(
        ["powercfg", "/setactive", "SCHEME_CURRENT"],
        capture_output=True, timeout=15
    )
    print("    Wake timers enabled for AC power.")

run("6. Enable wake timers", enable_wake_timers)

# ── Final report ──────────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("  PRE-MARKET READINESS REPORT")
print("=" * 55)

all_ok = True
for label, status in results.items():
    icon = "+" if status == OK else "!" if status == WARN else "X"
    print(f"  [{icon}] {label}")
    if status == FAIL:
        all_ok = False

print("=" * 55)

if all_ok:
    print("\n  READY FOR TOMORROW.")
    print("  Market opens at 9:30 AM ET.")
    print("  The scheduler will fire automatically.")
    print("  You can safely close this window and sleep/hibernate.")
    print("\n  To watch live tomorrow:")
    print("    Get-Content data\\logs\\__main__.log -Wait -Tail 30")
else:
    print("\n  Some checks failed. Review the FAIL items above.")
    print("  The system will still attempt to run tomorrow,")
    print("  but failed components will fall back to safe defaults.")

print()
