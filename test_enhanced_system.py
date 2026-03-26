"""
Quick test script to verify the enhanced system works.

Run this to check:
  1. Real data loading
  2. Feature engineering
  3. Model training
  4. Results interpretation
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

print("=" * 70)
print("🧪 ENHANCED TRADING SYSTEM - QUICK TEST")
print("=" * 70)
print()

# ══════════════════════════════════════════════════════════════════════════════
# TEST 1: Check dependencies
# ══════════════════════════════════════════════════════════════════════════════

print("1️⃣  Checking dependencies...")
try:
    import pandas as pd
    import numpy as np
    import sklearn
    print("   ✅ Core packages OK")
except ImportError as e:
    print(f"   ❌ Missing dependency: {e}")
    sys.exit(1)

try:
    import yfinance
    print("   ✅ yfinance OK (fallback data source)")
except ImportError:
    print("   ⚠️  yfinance not installed - run: pip install yfinance")
    print("   ℹ️  Will still work if you have Alpaca keys")

print()

# ══════════════════════════════════════════════════════════════════════════════
# TEST 2: Check API keys
# ══════════════════════════════════════════════════════════════════════════════

print("2️⃣  Checking API configuration...")
from dotenv import load_dotenv
load_dotenv()

alpaca_key = os.getenv("ALPACA_API_KEY")
alpaca_secret = os.getenv("ALPACA_SECRET_KEY")

if alpaca_key and alpaca_secret:
    print(f"   ✅ Alpaca configured (key: {alpaca_key[:8]}...)")
else:
    print("   ⚠️  Alpaca not configured - will use Yahoo fallback")
    print("   ℹ️  Get free keys at: https://alpaca.markets")

finnhub_key = os.getenv("FINNHUB_API_KEY")
newsapi_key = os.getenv("NEWSAPI_KEY")

if finnhub_key or newsapi_key:
    print(f"   ✅ News APIs configured")
else:
    print("   ⚠️  No news APIs - sentiment features disabled")
    print("   ℹ️  Get keys at: https://finnhub.io and https://newsapi.org")

print()

# ══════════════════════════════════════════════════════════════════════════════
# TEST 3: Load real data
# ══════════════════════════════════════════════════════════════════════════════

print("3️⃣  Testing real data loading...")
try:
    from app.data.multi_source_loader import get_training_data
    
    df = get_training_data(symbol="ES", days=5, min_rows=100)
    
    if len(df) > 0:
        is_synthetic = (df["close"].std() / df["close"].mean()) < 0.001
        if is_synthetic:
            print(f"   ⚠️  Got {len(df)} bars but looks synthetic")
            print("   ℹ️  Add ALPACA_API_KEY to .env for real data")
        else:
            print(f"   ✅ Loaded {len(df)} bars of REAL data")
            print(f"   📊 Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    else:
        print("   ❌ No data loaded - check API keys")
        
except Exception as e:
    print(f"   ❌ Data loading failed: {e}")
    import traceback
    traceback.print_exc()

print()

# ══════════════════════════════════════════════════════════════════════════════
# TEST 4: Feature engineering
# ══════════════════════════════════════════════════════════════════════════════

print("4️⃣  Testing feature engineering...")
try:
    from app.ml.advanced_features import make_advanced_features
    
    features_df = make_advanced_features(
        df,
        symbol="ES",
        include_sentiment=False,  # Skip for speed
        include_microstructure=True
    )
    
    feature_cols = [c for c in features_df.columns 
                   if c not in ["timestamp", "open", "high", "low", "close", "volume"]]
    
    print(f"   ✅ Created {len(feature_cols)} features")
    print(f"   📋 Examples:")
    for col in feature_cols[:10]:
        print(f"      - {col}")
    if len(feature_cols) > 10:
        print(f"      ... and {len(feature_cols) - 10} more")
        
except Exception as e:
    print(f"   ❌ Feature engineering failed: {e}")
    import traceback
    traceback.print_exc()

print()

# ══════════════════════════════════════════════════════════════════════════════
# TEST 5: Quick model training
# ══════════════════════════════════════════════════════════════════════════════

print("5️⃣  Testing model training (quick test - 5 days)...")
try:
    from app.ml.enhanced_training_pipeline import EnhancedTrainer
    
    trainer = EnhancedTrainer(
        model_type="random_forest",
        model_path="data/models/test_model.pkl"
    )
    
    results = trainer.train(
        symbol="ES",
        days=5,  # Just 5 days for quick test
        include_sentiment=False  # Skip for speed
    )
    
    print()
    print("   📊 QUICK TEST RESULTS:")
    print(f"      Train Accuracy: {results['train_accuracy']:.4f}")
    print(f"      Test Accuracy:  {results['test_accuracy']:.4f}")
    print(f"      Precision:      {results['precision']:.4f}")
    print(f"      Recall:         {results['recall']:.4f}")
    print()
    
    if results['test_accuracy'] > 0.52:
        print("   ✅ Model shows promise (>52% accuracy)")
    elif results['test_accuracy'] > 0.50:
        print("   ⚠️  Marginal accuracy - needs more data or features")
    else:
        print("   ⚠️  Low accuracy - features not predictive on this sample")
    
    print()
    print("   ℹ️  This was just a quick test with 5 days.")
    print("   ℹ️  For real training, use 30+ days:")
    print("   ℹ️    python -m app.ml.enhanced_training_pipeline --days 30")
    
except Exception as e:
    print(f"   ❌ Training failed: {e}")
    import traceback
    traceback.print_exc()

print()

# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("📝 SUMMARY")
print("=" * 70)
print()
print("Next steps:")
print()
print("1. Configure API keys in .env (if not done):")
print("   - Alpaca: https://alpaca.markets (required for real data)")
print("   - Finnhub: https://finnhub.io (optional, for sentiment)")
print("   - NewsAPI: https://newsapi.org (optional, for sentiment)")
print()
print("2. Train with more data:")
print("   python -m app.ml.enhanced_training_pipeline --symbol ES --days 30")
print()
print("3. Check REAL_DATA_SETUP.md for full documentation")
print()
print("4. If accuracy > 52%, proceed to backtesting")
print("   If accuracy < 52%, try different symbol or more features")
print()
print("=" * 70)
