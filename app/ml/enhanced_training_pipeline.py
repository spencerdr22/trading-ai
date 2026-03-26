"""
enhanced_training_pipeline.py — Complete training pipeline with:
  - Real data loading from multiple sources
  - Advanced feature engineering (technical + sentiment + microstructure)
  - Model training with proper validation
  - Feature importance analysis
  - Performance metrics

Run this instead of the old trainer.py for real predictions.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

from ..data.multi_source_loader import get_training_data
from ..ml.advanced_features import make_advanced_features, analyze_feature_importance
from ..monitor.logger import get_logger

logger = get_logger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# ENHANCED TRAINER
# ══════════════════════════════════════════════════════════════════════════════

class EnhancedTrainer:
    """
    Improved trainer that uses REAL data + advanced features.
    
    Key improvements over old trainer:
      1. Multi-source real data (not synthetic)
      2. News sentiment integration
      3. Market microstructure features
      4. Proper time-series validation
      5. Feature importance analysis
      6. Multiple model types
    """
    
    def __init__(
        self,
        model_type: str = "random_forest",
        model_path: str = "data/models/enhanced_model.pkl"
    ):
        """
        Args:
            model_type: "random_forest", "gradient_boosting", "hybrid"
            model_path: Where to save trained model
        """
        self.model_type = model_type
        self.model_path = model_path
        self.model = None
        self.feature_names = None
        self.feature_importance = None
        
    def prepare_data(
        self,
        symbol: str = "ES",
        days: int = 30,
        include_sentiment: bool = True,
        include_microstructure: bool = True
    ) -> tuple:
        """
        Load and prepare training data.
        
        Returns:
            (X_train, X_test, y_train, y_test, feature_names)
        """
        logger.info(f"📊 Loading {days} days of REAL data for {symbol}...")
        
        # Load real market data
        df = get_training_data(symbol=symbol, days=days, min_rows=500)
        
        if df.empty or len(df) < 100:
            raise ValueError(
                "Not enough data! Check API keys in .env file.\n"
                "Need at least: ALPACA_API_KEY and ALPACA_SECRET_KEY"
            )
        
        logger.info(f"✅ Loaded {len(df)} bars of real data")
        
        # Build features
        logger.info("🔧 Building advanced features...")
        df = make_advanced_features(
            df,
            symbol=symbol,
            include_sentiment=include_sentiment,
            include_microstructure=include_microstructure
        )
        
        # Create target
        # Predict if price will be higher in 5 bars (5 minutes)
        df["future_return"] = df["close"].pct_change(5).shift(-5)
        df["target"] = (df["future_return"] > 0.0001).astype(int)  # >1bp = buy
        
        # Remove rows with NaN target
        df = df.dropna(subset=["target"])
        
        if len(df) < 100:
            raise ValueError("Too few valid samples after feature engineering")
        
        # Separate features and target
        feature_cols = [
            c for c in df.columns
            if c not in ["timestamp", "target", "future_return",
                        "open", "high", "low", "close", "volume"]
        ]
        
        X = df[feature_cols].values
        y = df["target"].values
        
        # Time-series split (don't shuffle - preserves temporal order)
        split_idx = int(len(X) * 0.8)
        X_train = X[:split_idx]
        X_test = X[split_idx:]
        y_train = y[:split_idx]
        y_test = y[split_idx:]
        
        logger.info(
            f"📈 Features: {len(feature_cols)} | "
            f"Train: {len(X_train)} | Test: {len(X_test)}"
        )
        
        self.feature_names = feature_cols
        
        return X_train, X_test, y_train, y_test
        
    def train(
        self,
        symbol: str = "ES",
        days: int = 30,
        include_sentiment: bool = True
    ):
        """
        Train the model with proper validation.
        """
        # Prepare data
        X_train, X_test, y_train, y_test = self.prepare_data(
            symbol=symbol,
            days=days,
            include_sentiment=include_sentiment
        )
        
        # Create model
        logger.info(f"🤖 Training {self.model_type}...")
        
        if self.model_type == "random_forest":
            self.model = RandomForestClassifier(
                n_estimators=300,
                max_depth=15,
                min_samples_split=20,
                min_samples_leaf=10,
                max_features="sqrt",
                n_jobs=-1,
                random_state=42,
                class_weight="balanced"  # Handle imbalanced data
            )
        elif self.model_type == "gradient_boosting":
            self.model = GradientBoostingClassifier(
                n_estimators=200,
                max_depth=7,
                learning_rate=0.05,
                subsample=0.8,
                random_state=42
            )
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")
        
        # Train
        self.model.fit(X_train, y_train)
        
        # Evaluate
        train_preds = self.model.predict(X_train)
        test_preds = self.model.predict(X_test)
        
        # Metrics
        train_acc = accuracy_score(y_train, train_preds)
        test_acc = accuracy_score(y_test, test_preds)
        test_precision = precision_score(y_test, test_preds, zero_division=0)
        test_recall = recall_score(y_test, test_preds, zero_division=0)
        test_f1 = f1_score(y_test, test_preds, zero_division=0)
        
        logger.info("=" * 60)
        logger.info("📊 TRAINING RESULTS")
        logger.info("=" * 60)
        logger.info(f"Train Accuracy:  {train_acc:.4f}")
        logger.info(f"Test Accuracy:   {test_acc:.4f}")
        logger.info(f"Test Precision:  {test_precision:.4f}")
        logger.info(f"Test Recall:     {test_recall:.4f}")
        logger.info(f"Test F1 Score:   {test_f1:.4f}")
        
        # Feature importance
        if hasattr(self.model, "feature_importances_"):
            self.feature_importance = analyze_feature_importance(
                self.model,
                self.feature_names,
                top_n=20
            )
        
        # Save model
        self._save_model()
        
        # Reality check
        if test_acc < 0.51:
            logger.warning(
                "⚠️ Model accuracy is barely above random (50%). "
                "This suggests:\n"
                "  1. Features may not be predictive\n"
                "  2. Market is too noisy for this timeframe\n"
                "  3. Need more data or better features"
            )
        elif test_acc > 0.55:
            logger.info(f"✅ Model shows promise (accuracy > 55%)")
        
        return {
            "train_accuracy": train_acc,
            "test_accuracy": test_acc,
            "precision": test_precision,
            "recall": test_recall,
            "f1": test_f1
        }
    
    def _save_model(self):
        """Save model to disk."""
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        
        model_package = {
            "model": self.model,
            "feature_names": self.feature_names,
            "model_type": self.model_type,
            "trained_at": datetime.utcnow().isoformat()
        }
        
        joblib.dump(model_package, self.model_path)
        logger.info(f"💾 Model saved to: {self.model_path}")
        
    def load(self):
        """Load trained model."""
        if not os.path.exists(self.model_path):
            logger.error(f"Model not found: {self.model_path}")
            return None
        
        package = joblib.load(self.model_path)
        self.model = package["model"]
        self.feature_names = package["feature_names"]
        logger.info(f"📂 Model loaded from: {self.model_path}")
        return self.model


# ══════════════════════════════════════════════════════════════════════════════
# CLI INTERFACE
# ══════════════════════════════════════════════════════════════════════════════

def main():
    """Command-line interface for training."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced Trading Model Trainer")
    parser.add_argument("--symbol", default="ES", help="Trading symbol")
    parser.add_argument("--days", type=int, default=30, help="Days of data")
    parser.add_argument("--model-type", default="random_forest",
                       choices=["random_forest", "gradient_boosting"],
                       help="Model type")
    parser.add_argument("--no-sentiment", action="store_true",
                       help="Skip sentiment features (faster)")
    parser.add_argument("--output", default="data/models/enhanced_model.pkl",
                       help="Output path")
    
    args = parser.parse_args()
    
    # Train
    trainer = EnhancedTrainer(
        model_type=args.model_type,
        model_path=args.output
    )
    
    results = trainer.train(
        symbol=args.symbol,
        days=args.days,
        include_sentiment=not args.no_sentiment
    )
    
    logger.info("\n🎉 Training complete!")
    logger.info(f"Model saved to: {args.output}")
    logger.info("\nTo use in backtesting:")
    logger.info(f"  from app.ml.enhanced_training_pipeline import EnhancedTrainer")
    logger.info(f"  trainer = EnhancedTrainer(model_path='{args.output}')")
    logger.info(f"  trainer.load()")


if __name__ == "__main__":
    main()
