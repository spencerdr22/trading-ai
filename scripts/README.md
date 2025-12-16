# Scripts Directory

## Available Scripts

### 🚀 first_walkforward_backtest.py
Your first walk-forward validation backtest.

**Usage:**
```powershell
python scripts/first_walkforward_backtest.py
```

**What it does:**
- Loads sample data (2000 bars)
- Splits into 5 train/test periods
- Trains RandomForest model on each period
- Tests on out-of-sample data
- Tracks everything in MLflow
- Saves results to `data/validation/`

**Expected runtime:** 2-5 minutes

---

## Before Running

1. Make sure MLflow server is running:
```powershell
mlflow ui --port 5000
```

2. Activate your virtual environment:
```powershell
.venv\Scripts\activate
```

3. Ensure you're in the project root:
```powershell
cd C:\Users\spenc\Documents\trading-ai
```

---

## Understanding Results

- **Accuracy > 55%**: Good predictive power
- **Win Rate > 50%**: You have edge
- **Stable metrics**: Strategy is robust

---

## Next Steps After First Run

1. View results in MLflow: http://localhost:5000
2. Check `data/validation/walk_forward_first_wf_test.json`
3. Run with different configurations
4. Compare multiple runs in MLflow
