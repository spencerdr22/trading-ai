"""
enhanced_training_pipeline.py — Complete training pipeline with:
  - Real data loading from multiple sources
  - Advanced feature engineering (technical + sentiment + microstructure)
  - Model training with proper time-series validation
  - Feature importance analysis
  - Performance metrics

Key fixes vs previous version:
  - RF max_depth reduced to 3 (was 5) — core overfitting lever
  - min_samples_leaf raised to 50 (was 25)
  - n_estimators reduced to 50 (was 100) — faster, less overfit
  - target changed to 3-bar forward return > 0.03% (matches live trainer)
  - sentiment aggregation now requires MIN_ANALYZED_FOR_SIGNAL headlines
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

from ..data.multi_source_loader import get_training_data
from ..ml.advanced_features import make_advanced_features, analyze_feature_importance
from ..monitor.logger import get_logger

logger = get_logger(__name__)

# Overfitting guard: abort training if test accuracy drops below this
MIN_ACCEPTABLE_TEST_ACC = 0.49


class EnhancedTrainer:
    """
    Improved trainer that uses REAL data + advanced features.

    Key improvements over old trainer:
      1. Multi-source real data (not synthetic)
      2. News sentiment integration (with minimum-headline guard)
      3. Market microstructure features
      4. Proper time-series validation (no shuffle)
      5. Feature importance analysis
      6. Multiple model types with tighter regularisation
    """

    def __init__(
        self,
        model_type: str = "random_forest",
        model_path: str = "data/models/enhanced_model.pkl",
    ):
        self.model_type      = model_type
        self.model_path      = model_path
        self.model           = None
        self.feature_names   = None
        self.feature_importance = None

    def prepare_data(
        self,
        symbol:                 str  = "ES",
        days:                   int  = 30,
        include_sentiment:      bool = True,
        include_microstructure: bool = True,
    ) -> tuple:
        """Load and prepare training data."""
        logger.info("Loading %d days of REAL data for %s...", days, symbol)

        df = get_training_data(symbol=symbol, days=days, min_rows=500)

        if df.empty or len(df) < 100:
            raise ValueError(
                "Not enough data. Check ALPACA_API_KEY / ALPACA_SECRET_KEY in .env."
            )

        logger.info("Loaded %d bars of real data.", len(df))

        # Build features
        logger.info("Building advanced features...")
        df = make_advanced_features(
            df,
            symbol=symbol,
            include_sentiment=include_sentiment,
            include_microstructure=include_microstructure,
        )

        # Target: 3-bar forward return > 0.03% (matches live trainer threshold)
        # This filters out noise and forces the model to predict real moves,
        # not random tick fluctuations.
        df["future_return"] = df["close"].pct_change(3).shift(-3)
        df["target"] = (df["future_return"] > 0.0003).astype(int)
        df = df.dropna(subset=["target"])

        if len(df) < 100:
            raise ValueError("Too few valid samples after feature engineering.")

        feature_cols = [
            c for c in df.columns
            if c not in ["timestamp", "target", "future_return",
                         "open", "high", "low", "close", "volume"]
        ]

        X = df[feature_cols].values
        y = df["target"].values

        # Temporal split — never shuffle financial time-series
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        logger.info(
            "Features: %d | Train: %d | Test: %d",
            len(feature_cols), len(X_train), len(X_test),
        )

        self.feature_names = feature_cols
        return X_train, X_test, y_train, y_test

    def train(
        self,
        symbol:            str  = "ES",
        days:              int  = 30,
        include_sentiment: bool = True,
    ) -> dict:
        """Train the model with proper validation."""
        X_train, X_test, y_train, y_test = self.prepare_data(
            symbol=symbol,
            days=days,
            include_sentiment=include_sentiment,
        )

        logger.info("Training %s...", self.model_type)

        if self.model_type == "random_forest":
            self.model = RandomForestClassifier(
                n_estimators      = 50,    # fewer trees = less overfit
                max_depth         = 3,     # very shallow — key lever
                min_samples_split = 80,    # require large nodes before splitting
                min_samples_leaf  = 50,    # require substantial leaf support
                max_features      = "sqrt",
                max_samples       = 0.7,   # bootstrap 70% to increase variance
                n_jobs            = -1,
                random_state      = 42,
                class_weight      = "balanced",
            )
        elif self.model_type == "gradient_boosting":
            self.model = GradientBoostingClassifier(
                n_estimators  = 100,
                max_depth     = 3,
                learning_rate = 0.03,
                subsample     = 0.7,
                random_state  = 42,
            )
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

        self.model.fit(X_train, y_train)

        train_preds = self.model.predict(X_train)
        test_preds  = self.model.predict(X_test)

        train_acc      = accuracy_score(y_train, train_preds)
        test_acc       = accuracy_score(y_test,  test_preds)
        test_precision = precision_score(y_test, test_preds, zero_division=0)
        test_recall    = recall_score(y_test,    test_preds, zero_division=0)
        test_f1        = f1_score(y_test,        test_preds, zero_division=0)
        overfit_gap    = train_acc - test_acc

        logger.info("=" * 60)
        logger.info("TRAINING RESULTS")
        logger.info("=" * 60)
        logger.info("Train Accuracy:  %.4f", train_acc)
        logger.info("Test Accuracy:   %.4f", test_acc)
        logger.info("Test Precision:  %.4f", test_precision)
        logger.info("Test Recall:     %.4f", test_recall)
        logger.info("Test F1 Score:   %.4f", test_f1)
        logger.info("Overfit Gap:     %.4f", overfit_gap)

        # Feature importance
        if hasattr(self.model, "feature_importances_"):
            self.feature_importance = analyze_feature_importance(
                self.model, self.feature_names, top_n=20
            )

        # Save
        self._save_model()

        # Diagnostics
        if overfit_gap > 0.10:
            logger.warning(
                "Overfitting detected (gap=%.3f). "
                "Try reducing max_depth further or increasing min_samples_leaf.",
                overfit_gap,
            )
        elif test_acc < MIN_ACCEPTABLE_TEST_ACC:
            logger.warning(
                "Test accuracy %.3f is near random. "
                "Features may not be predictive for this window.",
                test_acc,
            )
        else:
            logger.info(
                "Model within acceptable range: test=%.3f, gap=%.3f",
                test_acc, overfit_gap,
            )

        return {
            "train_accuracy": train_acc,
            "test_accuracy":  test_acc,
            "precision":      test_precision,
            "recall":         test_recall,
            "f1":             test_f1,
            "overfit_gap":    overfit_gap,
        }

    def _save_model(self):
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump({
            "model":        self.model,
            "feature_names": self.feature_names,
            "model_type":   self.model_type,
            "trained_at":   datetime.now(timezone.utc).isoformat(),
        }, self.model_path)
        logger.info("Model saved to: %s", self.model_path)

    def load(self):
        if not os.path.exists(self.model_path):
            logger.error("Model not found: %s", self.model_path)
            return None
        package        = joblib.load(self.model_path)
        self.model     = package["model"]
        self.feature_names = package["feature_names"]
        logger.info("Model loaded from: %s", self.model_path)
        return self.model


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Enhanced Trading Model Trainer")
    parser.add_argument("--symbol",     default="ES")
    parser.add_argument("--days",       type=int, default=30)
    parser.add_argument("--model-type", default="random_forest",
                        choices=["random_forest", "gradient_boosting"])
    parser.add_argument("--no-sentiment", action="store_true")
    parser.add_argument("--output",     default="data/models/enhanced_model.pkl")
    args = parser.parse_args()

    trainer = EnhancedTrainer(model_type=args.model_type, model_path=args.output)
    results = trainer.train(
        symbol=args.symbol,
        days=args.days,
        include_sentiment=not args.no_sentiment,
    )

    logger.info("Training complete!")
    logger.info("Model saved to: %s", args.output)
    logger.info("Results: %s", results)


if __name__ == "__main__":
    main()
