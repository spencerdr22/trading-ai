"""
check_status.py — Quick trading system health check.

Run anytime to see:
  - Is the trading process running?
  - Alpaca account balance
  - Open positions
  - Recent orders
  - Last lines of the trading log

Usage:
    python check_status.py
"""

import os
import sys
import subprocess

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

SEP = "=" * 55


def section(title):
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)


# ── 1. Is the trading process alive? ─────────────────────────
section("PROCESS STATUS")
try:
    result = subprocess.run(
        ["tasklist", "/FI", "IMAGENAME eq python.exe", "/FO", "CSV", "/NH"],
        capture_output=True, text=True
    )
    lines = [l for l in result.stdout.strip().splitlines() if "python" in l.lower()]
    if lines:
        print(f"  ✅ {len(lines)} Python process(es) running:")
        for l in lines:
            parts = l.strip('"').split('","')
            if len(parts) >= 2:
                print(f"     PID {parts[1]}  |  {parts[0]}")
    else:
        print("  ❌ No Python processes found — trading is NOT running.")
except Exception as e:
    print(f"  Could not check processes: {e}")


# ── 2. Alpaca account ────────────────────────────────────────
section("ALPACA ACCOUNT")
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(PROJECT_ROOT, ".env"))
    from app.execution.alpaca_paper import get_alpaca_client

    client = get_alpaca_client()
    account = client.get_account()

    if account:
        pv   = float(account.get("portfolio_value", 0))
        bp   = float(account.get("buying_power", 0))
        cash = float(account.get("cash", 0))
        pnl  = float(account.get("equity", pv)) - float(account.get("last_equity", pv))
        status = account.get("status", "unknown")
        open_ = client.is_market_open()

        print(f"  Status        : {status.upper()}")
        print(f"  Market open   : {'✅ YES' if open_ else '🔴 NO (closed)'}")
        print(f"  Portfolio     : ${pv:,.2f}")
        print(f"  Cash          : ${cash:,.2f}")
        print(f"  Buying power  : ${bp:,.2f}")
        print(f"  Today P&L     : ${pnl:+,.2f}")
    else:
        print("  ❌ Could not reach Alpaca API.")
except Exception as e:
    print(f"  ❌ Alpaca error: {e}")


# ── 3. Open positions ─────────────────────────────────────────
section("OPEN POSITIONS")
try:
    positions = client.get_positions()
    if positions:
        for p in positions:
            sym    = p.get("symbol", "?")
            qty    = p.get("qty", "?")
            side   = p.get("side", "?")
            avg_px = float(p.get("avg_entry_price", 0))
            cur_px = float(p.get("current_price", 0))
            unreal = float(p.get("unrealized_pl", 0))
            print(f"  {sym:6s}  {side.upper():5s}  qty={qty:>4}  "
                  f"entry=${avg_px:.2f}  now=${cur_px:.2f}  "
                  f"P&L=${unreal:+.2f}")
    else:
        print("  No open positions (flat).")
except Exception as e:
    print(f"  ❌ Could not fetch positions: {e}")


# ── 4. Recent orders (last 5) ────────────────────────────────
section("RECENT ORDERS (last 5)")
try:
    import requests, os
    headers = {
        "APCA-API-KEY-ID":     os.getenv("ALPACA_API_KEY"),
        "APCA-API-SECRET-KEY": os.getenv("ALPACA_SECRET_KEY"),
    }
    base = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
    r = requests.get(
        f"{base}/v2/orders",
        headers=headers,
        params={"status": "all", "limit": 5, "direction": "desc"},
        timeout=10
    )
    if r.status_code == 200:
        orders = r.json()
        if orders:
            for o in orders:
                sym    = o.get("symbol", "?")
                side   = o.get("side", "?")
                qty    = o.get("qty", "?")
                status = o.get("status", "?")
                filled = o.get("filled_avg_price") or "pending"
                created = o.get("created_at", "")[:19].replace("T", " ")
                print(f"  {created}  {side.upper():4s} {qty:>4} {sym:6s}  "
                      f"status={status:10s}  fill=${filled}")
        else:
            print("  No orders found.")
    else:
        print(f"  ❌ Orders fetch failed: {r.status_code}")
except Exception as e:
    print(f"  ❌ Could not fetch orders: {e}")


# ── 5. Last 15 lines of trading log ─────────────────────────
section("RECENT LOG OUTPUT (last 15 lines)")
log_paths = [
    os.path.join(PROJECT_ROOT, "data", "scheduler.log"),
    os.path.join(PROJECT_ROOT, "data", "trading.log"),
]
found = False
for log_path in log_paths:
    if os.path.exists(log_path):
        found = True
        print(f"  [{os.path.basename(log_path)}]")
        try:
            with open(log_path, "r") as f:
                lines = f.readlines()
            for line in lines[-15:]:
                print(f"  {line.rstrip()}")
        except Exception as e:
            print(f"  Could not read log: {e}")
        break
if not found:
    print("  No log file found yet — has trading started?")

print(f"\n{SEP}\n")
