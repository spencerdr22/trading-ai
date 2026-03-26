# 🚀 Real Data + Advanced Features Setup Guide

## What Was Added

You now have a **complete real-data + sentiment-enhanced trading system**:

### ✅ New Files Created:

1. **`app/ml/advanced_features.py`**
   - Integrates news sentiment into features
   - Adds market microstructure signals (order flow proxies)
   - Combines with your existing technical indicators
   
2. **`app/data/multi_source_loader.py`**
   - Loads REAL market data from multiple sources
   - Priority: Alpaca → Polygon → Yahoo → Synthetic (last resort)
   - Automatic caching to avoid API rate limits
   
3. **`app/ml/enhanced_training_pipeline.py`**
   - Complete training pipeline using real data
   - Proper time-series validation
   - Feature importance analysis
   - Reality-check warnings if model won't work

---

## 🎯 What's Different Now

### Before (Your Old System):
```python
# Used mostly synthetic data
df = load_sample()  # Generated random walk

# Basic technical features only
features = make_features(df)

# No sentiment integration
# Missing order flow signals
```

### After (New Enhanced System):
```python
# Uses REAL market data
df = get_training_data(symbol="ES", days=30)

# Advanced multi-source features
features = make_advanced_features(
    df,
    symbol="ES",
    include_sentiment=True,       # ← LLM news analysis
    include_microstructure=True   # ← Order flow proxies
)

# Features now include:
# - All your technical indicators (EMA, RSI, MACD, etc.)
# - News sentiment (bullish/bearish/neutral with confidence)
# - Sentiment shocks (rapid news-driven moves)
# - Order flow proxies (tail ratios, volume delta)
# - Session time effects (RTH vs overnight)
# - Liquidity proxies
```

---

## 📋 Setup Steps

### Step 1: Install Additional Dependencies

```bash
pip install yfinance  # For Yahoo Finance fallback
```

### Step 2: Verify API Keys in .env

Your `.env` file should have these (you may already have some):

```env
# Primary data source (REQUIRED)
ALPACA_API_KEY=your_alpaca_key
ALPACA_SECRET_KEY=your_alpaca_secret

# Optional (improves data quality)
POLYGON_API_KEY=your_polygon_key
ALPHA_VANTAGE_API_KEY=your_av_key

# News APIs (for sentiment features)
FINNHUB_API_KEY=your_finnhub_key
NEWSAPI_KEY=your_newsapi_key

# LLM for sentiment analysis
# (Already configured - Qwen3 via Ollama)
```

**Minimum requirement:** Just `ALPACA_API_KEY` and `ALPACA_SECRET_KEY` (free paper trading account)

Get free Alpaca keys: https://alpaca.markets/ (paper trading account)

### Step 3: Train Enhanced Model

```bash
# Basic training (30 days of SPY data, includes sentiment)
python -m app.ml.enhanced_training_pipeline --symbol ES --days 30

# Faster training (skip sentiment if news APIs not configured)
python -m app.ml.enhanced_training_pipeline --symbol ES --days 30 --no-sentiment

# Use gradient boosting instead of random forest
python -m app.ml.enhanced_training_pipeline --model-type gradient_boosting
```

### Step 4: Check the Results

You'll see output like:

```
📊 Loading 30 days of REAL data for ES...
✅ Loaded 11,700 bars of real data
🔧 Building advanced features...
✅ Created 67 features for 11,700 bars
📈 Features: 67 | Train: 9,360 | Test: 2,340
🤖 Training random_forest...

============================================================
📊 TRAINING RESULTS
============================================================
Train Accuracy:  0.6234
Test Accuracy:   0.5487
Test Precision:  0.5612
Test Recall:     0.6123
Test F1 Score:   0.5854

Top 20 Most Important Features:
  sentiment_1h                  : 0.0842  ← News sentiment matters!
  rsi                           : 0.0673
  ema_cross_fast                : 0.0591
  tail_imbalance                : 0.0543  ← Order flow signal
  sentiment_shock               : 0.0498  ← News shocks matter
  volume_delta_proxy            : 0.0467  ← Volume matters
  ...
```

---

## 🧪 Testing the New System

### Test 1: Verify Real Data Loading

```python
from app.data.multi_source_loader import get_training_data

# This should fetch REAL data (not synthetic)
df = get_training_data(symbol="ES", days=5)
print(f"Loaded {len(df)} real bars")
print(df.head())
```

### Test 2: Check Feature Engineering

```python
from app.ml.advanced_features import make_advanced_features

df = get_training_data("ES", days=5)
features = make_advanced_features(df, symbol="ES", include_sentiment=True)

print(f"Features created: {len(features.columns)}")
print("Sentiment features:", [c for c in features.columns if "sentiment" in c])
print("Microstructure features:", [c for c in features.columns if "tail" in c or "delta" in c])
```

### Test 3: Train and Evaluate

```python
from app.ml.enhanced_training_pipeline import EnhancedTrainer

trainer = EnhancedTrainer()
results = trainer.train(symbol="ES", days=30, include_sentiment=True)

print("Results:", results)
# If test_accuracy > 0.52, you have edge!
# If test_accuracy < 0.51, features aren't predictive
```

---

## 🎓 Understanding the Features

### 1. Technical Features (from features.py)
- Normalized by ATR (so patterns work across regimes)
- Trend: EMA crosses, slopes
- Momentum: RSI, MACD
- Volatility: Bollinger bands, ATR ratio

### 2. Sentiment Features (NEW)
- `sentiment_1h`: Weighted average sentiment last hour
- `sentiment_4h`: Longer-term sentiment trend
- `sentiment_shock`: Sudden news-driven moves (>2 std devs)
- `news_volume`: Number of headlines per hour
- `sentiment_momentum`: Change in sentiment direction

### 3. Microstructure Features (NEW)
- `tail_imbalance`: Buy vs sell pressure from bar tails
- `volume_delta_proxy`: Directional volume estimate
- `cum_volume_delta`: Running sum of volume delta
- `liquidity_proxy`: Wide bars = low liquidity
- `is_rth`, `is_asian`, `is_london`: Session time effects
- `price_vol_corr`: Price-volume correlation

---

## 🚨 Common Issues & Solutions

### Issue 1: "No real data available"
**Cause:** No API keys configured
**Fix:** Add `ALPACA_API_KEY` and `ALPACA_SECRET_KEY` to `.env`

### Issue 2: "Only 50 headlines fetched"
**Cause:** Free tier rate limits
**Normal:** This is expected. Model will still work.

### Issue 3: "Sentiment features failed"
**Cause:** Ollama not running or news APIs not configured
**Fix:** Either:
  - Run `ollama serve` before training
  - Use `--no-sentiment` flag to skip

### Issue 4: "Test accuracy = 0.50"
**Cause:** Features not predictive OR market too noisy
**What it means:**
  - 50% = coin flip = no edge
  - You need >52% for profitable trading after costs
  - Try different symbols, timeframes, or more days of data

---

## 🔧 Integration with Existing Code

### Replace Old Trainer Usage:

**Old way:**
```python
from app.ml.trainer import Trainer

trainer = Trainer()
model = trainer.train(df)  # Used synthetic data
```

**New way:**
```python
from app.ml.enhanced_training_pipeline import EnhancedTrainer

trainer = EnhancedTrainer()
results = trainer.train(symbol="ES", days=30)  # Uses real data + sentiment
model = trainer.model
```

### Use in Backtesting:

```python
from app.ml.enhanced_training_pipeline import EnhancedTrainer
from app.ml.advanced_features import make_advanced_features

# Load trained model
trainer = EnhancedTrainer(model_path="data/models/enhanced_model.pkl")
trainer.load()

# Prepare features the same way it was trained
test_data = get_training_data("ES", days=5)
features = make_advanced_features(test_data, "ES", include_sentiment=True)

# Get predictions
X = features[[col for col in features.columns 
             if col not in ["timestamp", "open", "high", "low", "close", "volume"]]]
predictions = trainer.model.predict(X)
```

---

## 📊 Expected Performance

### Realistic Expectations:

| Accuracy | Meaning | Action |
|----------|---------|--------|
| 50-51% | Random / no edge | Don't trade |
| 52-54% | Marginal edge | Profitable with tight risk management |
| 55-57% | Good edge | Can trade profitably |
| 58%+ | Strong edge | Rare but very profitable |

**Important:** Even 52% accuracy can be profitable if:
- Your wins are bigger than your losses (good risk:reward)
- You have tight stop losses
- You trade during high-quality setups only

### Feature Importance Insights:

If you see high importance for:
- **Sentiment features** → News drives your edges
- **Microstructure features** → Order flow matters
- **Technical features** → Classic TA works
- **Session features** → Time-of-day effects matter

This tells you WHERE your edge comes from.

---

## 🚀 Next Steps

1. **Run initial training:**
   ```bash
   python -m app.ml.enhanced_training_pipeline --symbol ES --days 30
   ```

2. **Check accuracy:**
   - If > 52%: Proceed to backtesting
   - If < 52%: Try different symbol or more data

3. **Analyze feature importance:**
   - Which features matter most?
   - Can you enhance them further?

4. **Backtest properly:**
   - Use walk-forward validation
   - Include transaction costs
   - Test on out-of-sample data

5. **Paper trade:**
   - Start with small size
   - Verify model works in real-time
   - Monitor performance daily

---

## 💡 Pro Tips

1. **More data = better model**
   - 30 days minimum
   - 90+ days ideal for stable features

2. **Sentiment helps most during:**
   - News-driven markets (Fed days, earnings)
   - High volatility periods
   - Trending markets

3. **Microstructure helps most during:**
   - Range-bound markets
   - Low volume periods
   - Session transitions

4. **Don't overtrain:**
   - If train accuracy >> test accuracy, you're overfitting
   - Reduce tree depth or number of features

5. **Reality check:**
   - Even professional models struggle to beat 55%
   - Focus on risk management, not prediction perfection

---

**Questions?** Check the logs - they're very verbose and will tell you exactly what's happening!
