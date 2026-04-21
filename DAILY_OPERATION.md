# Trading-AI — Daily Operation Guide

## Do you need to do anything every morning?

**No — once you run `register_task.ps1` it is fully automatic.**

| What happens | When | How |
|---|---|---|
| PC wakes from sleep | 9:20 AM (your clock) | Windows Task Scheduler wake timer |
| Ollama starts with GPU settings | 9:20 AM | `autostart.py` step 2 |
| AI Orchestrator starts (Docker) | 9:20 AM | `autostart.py` step 1 |
| Ollama arbiter starts | 9:20 AM | `autostart.py` step 3 |
| Dashboard available at :8501 | 9:20 AM | `autostart.py` step 4 |
| Waits for market open | 9:20–9:30 AM | `start_trading.py` |
| Paper trading begins | 9:30 AM ET | Alpaca SPY proxy |
| Positions flattened | 3:55 PM ET | `start_trading.py` EOD |
| System idles | 3:55 PM+ | Dashboard stays up |

## One-time setup (do this once, tonight)

```
1. Run install_deps.bat          (installs missing packages)
2. Run pre_flight.py             (confirms all 9 checks pass)
3. Right-click register_task.ps1 → Run with PowerShell
   (registers both scheduled tasks and enables wake timers)
```

That's it. From tomorrow onwards the PC wakes itself and trades.

---

## What each task does

### TradingAI-Daily  (Mon–Fri 9:20 AM)
Runs `autostart.py` which:
1. Tries to start the AI Orchestrator via Docker (optional — skips gracefully if Docker not running)
2. Kills and restarts Ollama with FLASH_ATTENTION=1, GPU optimised
3. Starts the Ollama priority arbiter on port 11435
4. Starts the Streamlit dashboard on port 8501
5. Runs `start_trading.py` which waits until exactly 9:30 AM ET then trades

### TradingAI-Dashboard  (at every login)
Runs the Streamlit dashboard so `http://localhost:8501` is always available
when you are logged in, not just on trading days.

---

## If you want to start manually instead

Double-click `run_full_system.bat` — does the same as the scheduled task.

---

## AI Orchestrator on C:\

The orchestrator (`C:\Users\spenc\Documents\ai-orchestrator`) is started
automatically by `autostart.py` step 1 via Docker Compose.

If Docker Desktop is not running when the task fires, autostart.py logs a
warning and continues — trading falls back to calling Ollama directly on
port 11434. No trades are missed.

To start the orchestrator standalone at any time:
```
cd C:\Users\spenc\Documents\ai-orchestrator
docker compose -f docker\docker-compose.yml up
```

---

## Time zone note

The Task Scheduler trigger fires at 9:20 AM **in your PC's local clock**.
`start_trading.py` internally converts to Eastern Time via pytz, so it
correctly waits until 9:30 AM ET regardless of your PC's time zone.

If your PC clock IS Eastern Time — no adjustment needed.
If your PC clock is a different zone — the trigger still fires at 9:20 AM
local, and pytz handles the rest automatically.

---

## Logs

| Log | Location |
|---|---|
| Auto-launcher | `data/autostart.log` |
| Trading scheduler | `data/scheduler.log` |
| Trading system | `logs/trading_ai.log` |
| Alpaca signals | `data/logs/app_execution_alpaca_paper.log` |

---

## Removing the scheduled tasks

```powershell
Unregister-ScheduledTask -TaskName 'TradingAI-Daily'     -Confirm:$false
Unregister-ScheduledTask -TaskName 'TradingAI-Dashboard'  -Confirm:$false
```
