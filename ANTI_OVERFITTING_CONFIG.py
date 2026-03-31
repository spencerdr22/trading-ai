"""
Anti-overfitting configuration for enhanced_training_pipeline.py

Replace the RandomForestClassifier creation in your training pipeline
with these settings to reduce overfitting.
"""

from sklearn.ensemble import RandomForestClassifier

# PROBLEM: Your current settings
# RandomForestClassifier(
#     n_estimators=300,      ← Too many trees
#     max_depth=15,          ← Too deep (memorizes noise)
#     min_samples_split=20,  ← Too low
#     min_samples_leaf=10,   ← Too low
#     max_features="sqrt",
#     n_jobs=-1,
#     random_state=42,
#     class_weight="balanced"
# )

# SOLUTION: Anti-overfitting settings
ANTI_OVERFIT_RF = RandomForestClassifier(
    n_estimators=100,          # ← Reduced from 300 (fewer trees = less overfitting)
    max_depth=5,               # ← Reduced from 15 (shallow trees can't memorize)
    min_samples_split=50,      # ← Increased from 20 (require more samples to split)
    min_samples_leaf=25,       # ← Increased from 10 (leaves must have 25+ samples)
    max_features="sqrt",       # ← Keep this (good default)
    bootstrap=True,            # ← Ensure bootstrap sampling
    n_jobs=-1,
    random_state=42,
    class_weight="balanced",
    max_samples=0.8            # ← Use only 80% of data per tree (adds randomness)
)

# Expected results:
# - Train accuracy: 55-65% (down from 96%)
# - Test accuracy: 53-56% (up from 52.5%)
# - Gap: <10 percentage points (healthy!)
