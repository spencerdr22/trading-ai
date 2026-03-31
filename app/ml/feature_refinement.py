"""
feature_refinement.py — Improve features based on importance analysis.

Based on your results, these features matter most:
  1. Volatility (bb_width, atr_ratio)
  2. Trend (EMA crosses)
  3. Order flow (cum_volume_delta, price_vol_corr)
  4. Time effects (minute)

This adds enhanced versions of high-importance features.
"""

import pandas as pd
import numpy as np
from app.ml.features import make_features


def add_volatility_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enhanced volatility features based on bb_width importance.
    """
    df = df.copy()
    
    # BB width is important - add more volatility context
    close = df["close"]
    
    # Historical volatility (std of returns)
    returns = close.pct_change()
    df["hist_vol_10"] = returns.rolling(10).std()
    df["hist_vol_30"] = returns.rolling(30).std()
    df["vol_ratio_10_30"] = df["hist_vol_10"] / (df["hist_vol_30"] + 1e-9)
    
    # Volatility of volatility (vol clustering)
    df["vol_of_vol"] = df["hist_vol_10"].rolling(10).std()
    
    # Range expansion/contraction
    bar_range = (df["high"] - df["low"]) / df["close"]
    df["range_expansion"] = bar_range / bar_range.rolling(20).mean()
    
    return df


def add_enhanced_trend_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enhanced trend features based on EMA cross importance.
    """
    df = df.copy()
    
    # EMA crosses are important - add trend strength
    close = df["close"]
    
    # Multiple timeframe EMAs
    ema_5 = close.ewm(span=5).mean()
    ema_10 = close.ewm(span=10).mean()
    ema_20 = close.ewm(span=20).mean()
    ema_50 = close.ewm(span=50).mean()
    ema_100 = close.ewm(span=100).mean()
    
    # Trend alignment score (all EMAs in same direction)
    df["trend_alignment"] = (
        ((ema_5 > ema_10).astype(int) +
         (ema_10 > ema_20).astype(int) +
         (ema_20 > ema_50).astype(int) +
         (ema_50 > ema_100).astype(int)) / 4.0
    ) * 2 - 1  # Scale to [-1, 1]
    
    # Distance from moving averages (pullback detection)
    df["dist_from_ema20"] = (close - ema_20) / ema_20
    df["dist_from_ema50"] = (close - ema_50) / ema_50
    
    # Trend momentum (rate of change of trend)
    df["ema20_momentum"] = ema_20.pct_change(5)
    
    return df


def add_enhanced_order_flow_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enhanced order flow based on cum_volume_delta importance.
    """
    df = df.copy()
    
    # Volume delta already works - add more context
    if "cum_volume_delta" in df.columns:
        cvd = df["cum_volume_delta"]
        
        # CVD momentum (acceleration)
        df["cvd_momentum"] = cvd.diff(5)
        df["cvd_acceleration"] = df["cvd_momentum"].diff(5)
        
        # CVD divergence from price
        price_change = df["close"].pct_change(10)
        cvd_change = cvd.pct_change(10)
        df["cvd_divergence"] = (price_change * cvd_change) < 0  # Opposite signs
        
    # Volume imbalance
    if "volume" in df.columns:
        vol = df["volume"]
        
        # Unusual volume
        vol_ma = vol.rolling(20).mean()
        df["volume_surge"] = (vol / vol_ma) > 1.5
        
        # Volume trend
        df["volume_trend"] = vol.rolling(10).mean() / vol.rolling(30).mean()
    
    return df


def add_time_based_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enhanced time features based on 'minute' importance.
    """
    df = df.copy()
    
    if "timestamp" not in df.columns:
        return df
    
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    
    # Minute was important - add more time context
    hour = df["timestamp"].dt.hour
    minute = df["timestamp"].dt.minute
    
    # Critical times for ES futures
    df["is_open"] = ((hour == 9) & (minute < 45)).astype(float)  # First 45 min
    df["is_lunch"] = ((hour == 11) | (hour == 12)).astype(float)  # Lunch doldrums
    df["is_close"] = ((hour == 15) & (minute >= 30)).astype(float)  # Last 30 min
    df["is_overnight"] = ((hour < 9) | (hour >= 16)).astype(float)
    
    # Time since market open (normalized)
    minutes_since_open = (hour - 9) * 60 + minute
    df["time_ratio"] = np.clip(minutes_since_open / 390, 0, 1)  # 390 min trading day
    
    return df


def make_refined_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Complete refined feature set based on importance analysis.
    
    Focuses on:
      1. Volatility regime detection
      2. Trend strength and alignment
      3. Enhanced order flow
      4. Time-of-day optimization
    """
    # Start with base features
    df = make_features(df)
    
    # Add refined features
    df = add_volatility_regime_features(df)
    df = add_enhanced_trend_features(df)
    df = add_enhanced_order_flow_features(df)
    df = add_time_based_features(df)
    
    # Fill NaNs
    df = df.ffill().bfill().fillna(0)
    
    return df


# ══════════════════════════════════════════════════════════════════════════════
# Test script
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from app.data.multi_source_loader import get_training_data
    
    print("Testing refined features...")
    
    df = get_training_data("ES", days=5)
    refined_df = make_refined_features(df)
    
    feature_cols = [c for c in refined_df.columns 
                   if c not in ["timestamp", "open", "high", "low", "close", "volume"]]
    
    print(f"\nCreated {len(feature_cols)} features")
    print("\nNew refined features:")
    new_features = [c for c in feature_cols if c not in df.columns]
    for feat in new_features:
        print(f"  - {feat}")
