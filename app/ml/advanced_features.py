"""
advanced_features.py — Enhanced feature engineering combining:
  - Technical indicators (price/volume)
  - News sentiment from LLM
  - Market microstructure signals
  - Order flow proxies
  
Designed specifically for ES/MES futures trading profitability.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from typing import Optional

from ..llm.news_feeds import NewsFeedManager
from ..llm.news_analyzer import NewsFlowAnalyzer
from ..monitor.logger import get_logger
from .features import make_features as make_technical_features

logger = get_logger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# SENTIMENT FEATURES
# ══════════════════════════════════════════════════════════════════════════════

class SentimentFeatureBuilder:
    """
    Builds time-aligned sentiment features from news headlines.
    
    Key insight: News sentiment has predictive power for 15-60 minutes,
    then mean-reverts. We capture both immediate shock and decay.
    """
    
    def __init__(self, use_cache: bool = True):
        self.news_manager = NewsFeedManager()
        self.analyzer = NewsFlowAnalyzer(use_cache=use_cache)
        
    def build_sentiment_features(
        self,
        df: pd.DataFrame,
        symbol: str = "ES",
        lookback_hours: int = 24
    ) -> pd.DataFrame:
        """
        Add news sentiment features to OHLCV DataFrame.
        
        Features created:
          - sentiment_1h: avg sentiment last 1 hour
          - sentiment_4h: avg sentiment last 4 hours  
          - sentiment_shock: spike in sentiment (> 2 std devs)
          - news_volume: number of headlines per hour
          - sentiment_momentum: change in sentiment direction
        """
        df = df.copy()
        
        # Fetch news
        try:
            headlines_df = self.news_manager.get_recent_headlines(
                symbols=[symbol],
                hours=lookback_hours
            )
            
            if headlines_df.empty:
                logger.warning("No news data - using neutral sentiment")
                return self._add_neutral_sentiment(df)
                
            # Analyze sentiment
            analyzed = self.analyzer.analyze_batch(
                headlines_df["headline"].tolist(),
                symbol=symbol
            )
            
            if analyzed.empty:
                return self._add_neutral_sentiment(df)
                
            # Time-align sentiment to bars
            return self._time_align_sentiment(df, analyzed)
            
        except Exception as e:
            logger.error(f"Sentiment feature error: {e}")
            return self._add_neutral_sentiment(df)
            
    def _add_neutral_sentiment(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add neutral sentiment when news unavailable."""
        df["sentiment_1h"] = 0.0
        df["sentiment_4h"] = 0.0
        df["sentiment_shock"] = 0.0
        df["news_volume"] = 0.0
        df["sentiment_momentum"] = 0.0
        return df
        
    def _time_align_sentiment(
        self,
        bars_df: pd.DataFrame,
        sentiment_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Time-align news sentiment to price bars.
        
        Strategy:
          - For each bar, look back 1h and 4h
          - Weight recent news more heavily (exponential decay)
          - Detect sentiment shocks (rapid changes)
        """
        bars_df = bars_df.copy()
        
        # Ensure timestamp columns are datetime
        if "timestamp" not in bars_df.columns:
            logger.error("bars_df missing 'timestamp' column")
            return self._add_neutral_sentiment(bars_df)
            
        bars_df["timestamp"] = pd.to_datetime(bars_df["timestamp"])
        
        if "timestamp" not in sentiment_df.columns:
            logger.error("sentiment_df missing 'timestamp'")
            return self._add_neutral_sentiment(bars_df)
            
        sentiment_df["timestamp"] = pd.to_datetime(sentiment_df["timestamp"])
        
        # Get weighted sentiment (confidence * relevance * direction)
        if "weighted_score" in sentiment_df.columns:
            sentiment_df["score"] = sentiment_df["weighted_score"]
        else:
            # Compute manually
            sent_map = {"bullish": 1.0, "neutral": 0.0, "bearish": -1.0}
            sentiment_df["score"] = (
                sentiment_df["sentiment"].map(sent_map).fillna(0.0)
                * sentiment_df["confidence"]
                * sentiment_df["relevance"]
            )
        
        # Initialize feature columns
        bars_df["sentiment_1h"] = 0.0
        bars_df["sentiment_4h"] = 0.0
        bars_df["sentiment_shock"] = 0.0
        bars_df["news_volume"] = 0.0
        bars_df["sentiment_momentum"] = 0.0
        
        # For each bar, compute sentiment features
        for idx, row in bars_df.iterrows():
            bar_time = row["timestamp"]
            if pd.isna(bar_time):
                continue
                
            # Ensure bar_time is timezone-aware
            if bar_time.tzinfo is None:
                bar_time = bar_time.replace(tzinfo=timezone.utc)
                
            # 1-hour lookback
            cutoff_1h = bar_time - timedelta(hours=1)
            recent_1h = sentiment_df[sentiment_df["timestamp"] > cutoff_1h]
            
            # 4-hour lookback  
            cutoff_4h = bar_time - timedelta(hours=4)
            recent_4h = sentiment_df[sentiment_df["timestamp"] > cutoff_4h]
            
            # Weighted average sentiment (more recent = higher weight)
            if not recent_1h.empty:
                # Exponential decay: weight = exp(-minutes_ago / 30)
                recent_1h = recent_1h.copy()
                recent_1h["minutes_ago"] = (
                    (bar_time - recent_1h["timestamp"]).dt.total_seconds() / 60
                )
                recent_1h["weight"] = np.exp(-recent_1h["minutes_ago"] / 30)
                weighted_sum = (recent_1h["score"] * recent_1h["weight"]).sum()
                weight_sum = recent_1h["weight"].sum()
                bars_df.at[idx, "sentiment_1h"] = weighted_sum / weight_sum if weight_sum > 0 else 0.0
                bars_df.at[idx, "news_volume"] = len(recent_1h)
                
            if not recent_4h.empty:
                recent_4h = recent_4h.copy()
                recent_4h["minutes_ago"] = (
                    (bar_time - recent_4h["timestamp"]).dt.total_seconds() / 60
                )
                recent_4h["weight"] = np.exp(-recent_4h["minutes_ago"] / 120)  # slower decay
                weighted_sum = (recent_4h["score"] * recent_4h["weight"]).sum()
                weight_sum = recent_4h["weight"].sum()
                bars_df.at[idx, "sentiment_4h"] = weighted_sum / weight_sum if weight_sum > 0 else 0.0
        
        # Compute sentiment shock (rapid change)
        sent_1h_series = bars_df["sentiment_1h"]
        sent_std = sent_1h_series.rolling(60, min_periods=10).std().fillna(0)
        sent_change = sent_1h_series.diff(5).fillna(0)
        bars_df["sentiment_shock"] = (sent_change / (sent_std + 0.01)).fillna(0).clip(-5, 5)
        
        # Sentiment momentum (direction change)
        bars_df["sentiment_momentum"] = sent_1h_series.diff(15).fillna(0)
        
        logger.info(f"Added sentiment features: avg={bars_df['sentiment_1h'].mean():.3f}")
        
        return bars_df


# ══════════════════════════════════════════════════════════════════════════════
# MARKET MICROSTRUCTURE FEATURES
# ══════════════════════════════════════════════════════════════════════════════

def add_microstructure_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add market microstructure features that capture order flow.
    
    These are derived from OHLCV only (we don't have L2 book data),
    but still provide valuable signals:
      - Tail ratios (buy vs sell pressure)
      - Volume delta proxy
      - Liquidity proxies
      - Session time effects
    """
    df = df.copy()
    
    # ── Tail ratios (order flow proxy) ──────────────────────────────
    # Upper tail = high - close (sellers absorbed buying)
    # Lower tail = close - low (buyers absorbed selling)
    bar_range = (df["high"] - df["low"]).replace(0, 1e-9)
    df["upper_tail"] = (df["high"] - df["close"]) / bar_range
    df["lower_tail"] = (df["close"] - df["low"]) / bar_range
    df["tail_imbalance"] = df["lower_tail"] - df["upper_tail"]  # >0 = buying pressure
    
    # ── Volume delta proxy ──────────────────────────────────────────
    # Close near high = buying, close near low = selling
    # Weight volume by position in bar
    df["volume_delta_proxy"] = (
        df["volume"] * (df["close"] - df["low"]) / bar_range
    ) - (
        df["volume"] * (df["high"] - df["close"]) / bar_range
    )
    
    # Cumulative volume delta (running sum)
    df["cum_volume_delta"] = df["volume_delta_proxy"].rolling(20, min_periods=1).sum()
    
    # ── Liquidity / spread proxy ────────────────────────────────────
    # Wide bars = low liquidity, tight bars = high liquidity
    atr_20 = (df["high"] - df["low"]).rolling(20, min_periods=1).mean().replace(0, 1e-9)
    df["liquidity_proxy"] = bar_range / atr_20  # >1 = low liquidity
    
    # ── Time-of-day effects ─────────────────────────────────────────
    # ES is most liquid during RTH (9:30-16:00 ET)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["hour"] = df["timestamp"].dt.hour
        df["minute"] = df["timestamp"].dt.minute
        
        # Session indicators
        df["is_rth"] = ((df["hour"] >= 9) & (df["hour"] < 16)).astype(float)
        df["is_asian"] = ((df["hour"] >= 18) | (df["hour"] < 1)).astype(float)
        df["is_london"] = ((df["hour"] >= 2) & (df["hour"] < 9)).astype(float)
        
        # First/last hour (usually more volatile)
        df["is_opening"] = ((df["hour"] == 9) & (df["minute"] < 60)).astype(float)
        df["is_closing"] = ((df["hour"] == 15) & (df["minute"] >= 30)).astype(float)
    
    # ── Order flow momentum ─────────────────────────────────────────
    # Rolling correlation of price and volume
    df["price_vol_corr"] = (
        df["close"]
        .rolling(20, min_periods=10)
        .corr(df["volume"])
        .fillna(0)
    )
    
    return df


# ══════════════════════════════════════════════════════════════════════════════
# MASTER FEATURE BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def make_advanced_features(
    df: pd.DataFrame,
    symbol: str = "ES",
    include_sentiment: bool = True,
    include_microstructure: bool = True
) -> pd.DataFrame:
    """
    Build complete feature set combining:
      1. Technical indicators (from features.py)
      2. News sentiment (from LLM)
      3. Market microstructure
      
    Args:
        df: OHLCV DataFrame
        symbol: Trading symbol for news lookup
        include_sentiment: Add news features (slower, requires APIs)
        include_microstructure: Add order flow features
        
    Returns:
        DataFrame with all features
    """
    logger.info(f"Building advanced features for {len(df)} bars...")
    
    # Step 1: Technical features (base)
    df = make_technical_features(df)
    
    # Step 2: Sentiment features (requires news APIs)
    if include_sentiment:
        try:
            builder = SentimentFeatureBuilder()
            df = builder.build_sentiment_features(df, symbol=symbol)
        except Exception as e:
            logger.warning(f"Sentiment features failed: {e} - continuing without")
    
    # Step 3: Microstructure features
    if include_microstructure:
        df = add_microstructure_features(df)
    
    # Final cleanup
    df = df.ffill().bfill().fillna(0)
    
    feature_count = len([c for c in df.columns if c not in [
        "timestamp", "open", "high", "low", "close", "volume"
    ]])
    
    logger.info(f"✅ Created {feature_count} features for {len(df)} bars")
    
    return df


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE IMPORTANCE ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def analyze_feature_importance(
    model,
    feature_names: list,
    top_n: int = 20
) -> pd.DataFrame:
    """
    Extract and rank feature importance from trained model.
    Works with RandomForest, GradientBoosting, etc.
    """
    if not hasattr(model, "feature_importances_"):
        logger.warning("Model doesn't support feature_importances_")
        return pd.DataFrame()
    
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)
    
    logger.info(f"\nTop {top_n} Most Important Features:")
    for idx, row in importance_df.head(top_n).iterrows():
        logger.info(f"  {row['feature']:30s}: {row['importance']:.4f}")
    
    return importance_df
