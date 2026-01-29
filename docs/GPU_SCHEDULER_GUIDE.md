# GPU Scheduler & Adaptive Config Guide

## 🎯 Overview

The GPU scheduler prevents resource contention between:
- **Trading models** (LSTM/RL) - HIGHEST PRIORITY
- **LLM analysis** (Qwen3) - LOWER PRIORITY

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install pytz
```

### 2. Test the Scheduler
```bash
# Test GPU scheduling
python -m app.llm.test_scheduling gpu

# Test adaptive config
python -m app.llm.test_scheduling config

# Test both
python -m app.llm.test_scheduling both
```

### 3. Integration
The scheduler is automatically active when you run:
```bash
python -m app.main --mode forward --symbol ES
```

## 📊 Expected Behavior

### Market Hours (9:30 AM - 4:00 PM ET)
- GPU: 100% reserved for trading
- LLM: Deferred (headlines buffered)
- CPU: 60% trading + features

### After Hours
- GPU: Available for sentiment analysis
- LLM: Active analysis of buffered headlines
- CPU: 40% background processing

### Weekends
- GPU: Batch processing + model training
- LLM: Full analysis of week's news

## ⚠️ Troubleshooting

### "GPU scheduler timeout"
**Cause:** Background task holding GPU too long  
**Fix:** Check `log_system_status()` output, restart if needed

### "Analysis deferred - GPU busy"
**Expected behavior** during market hours

### High CPU usage
**Fix:** System will auto-reduce workers when load > 70%

---

**Hardware:** AMD Ryzen 7 7800X3D, RTX 4070 Super, 32GB DDR5-6400
