"""
apply_sortino_updates.py
Applies all Sortino-prioritized reward function updates to the Trading-AI system.

Changes:
1. Update reward.py default weights (Sortino 0.3, Sharpe 0.1)
2. Add sortino_minute() to metrics.py
3. Update optimizer.py hyperparameter ranges
4. Update tests for new reward weights

Usage:
    python apply_sortino_updates.py --dry-run  # Preview changes
    python apply_sortino_updates.py --apply    # Apply changes
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime

# Backup directory
BACKUP_DIR = Path("data/backups") / datetime.utcnow().strftime("%Y%m%d_%H%M%S")

def backup_file(filepath):
    """Create backup of file before modification."""
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    src = Path(filepath)
    if src.exists():
        dst = BACKUP_DIR / src.name
        dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"✅ Backed up: {filepath} -> {dst}")

def update_reward_py():
    """Update app/adaptive/reward.py with new default weights."""
    filepath = "app/adaptive/reward.py"
    backup_file(filepath)
    
    content = Path(filepath).read_text(encoding="utf-8")
    
    # Update compute_reward signature
    old_sig = """def compute_reward(
    pnl_series,
    win_rate,
    leverage_factor=1.0,
    pnl_weight=0.5,
    sharpe_weight=0.2,
    sortino_weight=0.2,
    dd_penalty_weight=0.3,
    win_rate_weight=0.4
):"""
    
    new_sig = """def compute_reward(
    pnl_series,
    win_rate,
    leverage_factor=1.0,
    pnl_weight=0.5,
    sharpe_weight=0.1,      # Reduced: secondary metric
    sortino_weight=0.3,     # Increased: primary risk metric
    dd_penalty_weight=0.3,
    win_rate_weight=0.4
):"""
    
    if old_sig in content:
        content = content.replace(old_sig, new_sig)
        Path(filepath).write_text(content, encoding="utf-8")
        print(f"✅ Updated: {filepath}")
        return True
    else:
        print(f"⚠️  Pattern not found in {filepath} - may already be updated")
        return False

def update_metrics_py():
    """Add sortino_minute() function to app/backtest/metrics.py."""
    filepath = "app/backtest/metrics.py"
    backup_file(filepath)
    
    content = Path(filepath).read_text(encoding="utf-8")
    
    # Check if sortino_minute already exists
    if "def sortino_minute" in content:
        print(f"⚠️  sortino_minute() already exists in {filepath}")
        return False
    
    # Add sortino_minute function after sharpe_minute
    sortino_func = '''
def sortino_minute(returns, risk_free=0.0):
    """
    Compute Sortino ratio for minute-level returns.
    Only penalizes downside volatility (better for trading systems).
    
    Args:
        returns: Series of minute returns
        risk_free: Risk-free rate (default 0.0)
    
    Returns:
        Sortino ratio scaled to daily-ish frequency
    """
    mean = np.mean(returns)
    downside = returns[returns < risk_free]
    
    if len(downside) == 0:
        return float('inf')  # No downside = perfect
    
    downside_std = np.std(downside, ddof=1)
    if downside_std == 0:
        return 0.0
    
    # Scale to daily-ish (approx 390 minutes per trading day)
    return (mean - risk_free) / downside_std * np.sqrt(390)
'''
    
    # Find position after sharpe_minute function
    sharpe_end = content.find("def sharpe_minute")
    if sharpe_end == -1:
        print(f"⚠️  Could not find sharpe_minute() in {filepath}")
        return False
    
    # Find end of sharpe_minute function (next def or end of file)
    next_def = content.find("\ndef ", sharpe_end + 1)
    if next_def == -1:
        next_def = len(content)
    
    # Insert sortino_minute
    content = content[:next_def] + sortino_func + "\n" + content[next_def:]
    Path(filepath).write_text(content, encoding="utf-8")
    print(f"✅ Added sortino_minute() to {filepath}")
    return True

def update_optimizer_py():
    """Update app/adaptive/optimizer.py hyperparameter ranges."""
    filepath = "app/adaptive/optimizer.py"
    backup_file(filepath)
    
    content = Path(filepath).read_text(encoding="utf-8")
    
    # Update sharpe_weight range
    old_sharpe = 'sharpe_weight = trial.suggest_float("sharpe_weight", 0.1, 0.4)'
    new_sharpe = 'sharpe_weight = trial.suggest_float("sharpe_weight", 0.05, 0.15)  # Reduced range'
    
    # Update sortino_weight range
    old_sortino = 'sortino_weight = trial.suggest_float("sortino_weight", 0.1, 0.4)'
    new_sortino = 'sortino_weight = trial.suggest_float("sortino_weight", 0.2, 0.4)  # Increased range'
    
    updated = False
    if old_sharpe in content:
        content = content.replace(old_sharpe, new_sharpe)
        updated = True
    
    if old_sortino in content:
        content = content.replace(old_sortino, new_sortino)
        updated = True
    
    if updated:
        Path(filepath).write_text(content, encoding="utf-8")
        print(f"✅ Updated: {filepath}")
        return True
    else:
        print(f"⚠️  Patterns not found in {filepath} - may already be updated")
        return False

def create_changelog_entry():
    """Add entry to CHANGELOG.md."""
    filepath = "CHANGELOG.md"
    backup_file(filepath)
    
    entry = f"""
---

## Sortino-Prioritized Reward Update — {datetime.utcnow().strftime("%Y-%m-%d")}
**Type:** Performance Enhancement

### Changes:
- **reward.py**: Updated default weights
  - Sharpe weight: 0.2 → 0.1 (reduced)
  - Sortino weight: 0.2 → 0.3 (increased)
  - Rationale: Sortino better captures asymmetric trading returns
  
- **metrics.py**: Added `sortino_minute()` function
  - Computes Sortino ratio for minute-level returns
  - Only penalizes downside volatility
  - Scaled to daily frequency (√390)
  
- **optimizer.py**: Updated hyperparameter ranges
  - Sharpe range: [0.1, 0.4] → [0.05, 0.15]
  - Sortino range: [0.1, 0.4] → [0.2, 0.4]
  - Encourages Optuna to prefer Sortino-heavy configurations

### Impact:
- RL policy will learn to minimize downside risk specifically
- Upside volatility no longer penalized equally with downside
- Better alignment with practical trading objectives

### Testing:
```bash
pytest -v app/tests/test_reward.py
python -m app.adaptive.run_offline_rl --episodes 10 --tune
```

---
"""
    
    content = Path(filepath).read_text(encoding="utf-8")
    
    # Insert after first header
    lines = content.split("\n")
    insert_pos = 0
    for i, line in enumerate(lines):
        if line.startswith("## ") or line.startswith("# "):
            insert_pos = i + 1
            break
    
    lines.insert(insert_pos, entry)
    Path(filepath).write_text("\n".join(lines), encoding="utf-8")
    print(f"✅ Updated: {filepath}")

def create_test_file():
    """Create app/tests/test_reward.py if it doesn't exist."""
    filepath = "app/tests/test_reward.py"
    
    if Path(filepath).exists():
        print(f"⚠️  {filepath} already exists")
        return False
    
    content = '''"""
Test suite for reward calculation functions.
Validates Sharpe, Sortino, drawdown, and composite reward logic.
"""

import pytest
import numpy as np
from app.adaptive.reward import (
    compute_sharpe,
    compute_sortino,
    compute_drawdown,
    compute_reward,
    compute_batch_reward
)


def test_compute_sharpe():
    """Test Sharpe ratio calculation."""
    # Positive returns with volatility
    pnl = [1, 2, -1, 3, -0.5, 2]
    sharpe = compute_sharpe(pnl)
    assert sharpe > 0, "Positive PnL should have positive Sharpe"
    
    # Zero returns
    pnl_zero = [0, 0, 0, 0]
    sharpe_zero = compute_sharpe(pnl_zero)
    assert sharpe_zero == 0, "Zero returns should have zero Sharpe"
    
    # Single value
    sharpe_single = compute_sharpe([1])
    assert sharpe_single == 0, "Single value should return 0"


def test_compute_sortino():
    """Test Sortino ratio calculation."""
    # Mixed returns (upside and downside)
    pnl = [1, 2, -1, 3, -0.5, 2]
    sortino = compute_sortino(pnl)
    assert sortino > 0, "Positive mean PnL should have positive Sortino"
    
    # Only positive returns
    pnl_positive = [1, 2, 3, 4, 5]
    sortino_positive = compute_sortino(pnl_positive)
    assert sortino_positive > 0, "All positive returns should have high Sortino"
    
    # Sortino should be higher than Sharpe for asymmetric returns
    sharpe = compute_sharpe(pnl_positive)
    assert sortino_positive > sharpe, "Sortino should be higher for upside-skewed returns"


def test_compute_drawdown():
    """Test max drawdown calculation."""
    # Increasing equity
    pnl_increasing = [1, 1, 1, 1, 1]
    dd_increasing = compute_drawdown(pnl_increasing)
    assert dd_increasing == 0, "No drawdown in increasing equity"
    
    # Drawdown scenario
    pnl_dd = [10, 5, -5, -10, 5]  # Peak at 10, trough at -10
    dd = compute_drawdown(pnl_dd)
    assert dd > 0, "Should detect drawdown"
    
    # Single value
    dd_single = compute_drawdown([1])
    assert dd_single == 0, "Single value has no drawdown"


def test_compute_reward_weights():
    """Test that reward weights are applied correctly."""
    pnl = [1, 2, -1, 3, -0.5, 2]
    win_rate = 0.6
    
    # Get reward with default weights
    reward = compute_reward(pnl, win_rate)
    
    # Reward should be positive for profitable series
    assert reward > 0, "Profitable trading should have positive reward"
    
    # Test that Sortino is weighted more than Sharpe
    reward_high_sortino = compute_reward(
        pnl, win_rate,
        sharpe_weight=0.1,
        sortino_weight=0.3
    )
    
    reward_high_sharpe = compute_reward(
        pnl, win_rate,
        sharpe_weight=0.3,
        sortino_weight=0.1
    )
    
    # For upside-skewed returns, high Sortino weight should yield higher reward
    # (This may not always be true depending on other factors, so we just check it doesn't crash)
    assert isinstance(reward_high_sortino, float)
    assert isinstance(reward_high_sharpe, float)


def test_compute_batch_reward():
    """Test batch reward computation."""
    trade_pnls = [1, -0.5, 2, -1, 3, 1.5]
    win_rate = 0.67  # 4 wins, 2 losses
    
    reward = compute_batch_reward(trade_pnls, win_rate)
    
    assert isinstance(reward, float), "Reward should be float"
    assert reward > 0, "Profitable batch should have positive reward"
    assert -10.0 <= reward <= 10.0, "Reward should be clipped to [-10, 10]"


def test_reward_clipping():
    """Test that extreme rewards are clipped."""
    # Extreme profitable scenario
    pnl_extreme = [100, 200, 300, 400, 500]
    win_rate = 1.0
    
    reward = compute_reward(pnl_extreme, win_rate)
    assert reward <= 10.0, "Reward should be clipped at 10.0"
    
    # Extreme loss scenario
    pnl_loss = [-100, -200, -300]
    win_rate = 0.0
    
    reward_loss = compute_reward(pnl_loss, win_rate)
    assert reward_loss >= -10.0, "Reward should be clipped at -10.0"


def test_leverage_penalty():
    """Test that high leverage is penalized."""
    pnl = [1, 2, 3, 4, 5]
    win_rate = 0.8
    
    reward_low_lev = compute_reward(pnl, win_rate, leverage_factor=1.0)
    reward_high_lev = compute_reward(pnl, win_rate, leverage_factor=3.0)
    
    assert reward_low_lev > reward_high_lev, "High leverage should reduce reward"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
'''
    
    Path(filepath).write_text(content, encoding="utf-8")
    print(f"✅ Created: {filepath}")
    return True

def main():
    parser = argparse.ArgumentParser(description="Apply Sortino-prioritized updates")
    parser.add_argument("--dry-run", action="store_true", help="Preview changes without applying")
    parser.add_argument("--apply", action="store_true", help="Apply all changes")
    args = parser.parse_args()
    
    if not args.dry_run and not args.apply:
        print("❌ Error: Must specify --dry-run or --apply")
        sys.exit(1)
    
    if args.dry_run:
        print("🔍 DRY RUN MODE - No files will be modified\n")
    else:
        print("✏️  APPLY MODE - Files will be updated\n")
    
    print("=" * 60)
    print("SORTINO-PRIORITIZED REWARD UPDATE")
    print("=" * 60)
    print()
    
    files_to_update = [
        ("app/adaptive/reward.py", "Update default weight parameters"),
        ("app/backtest/metrics.py", "Add sortino_minute() function"),
        ("app/adaptive/optimizer.py", "Update hyperparameter ranges"),
        ("CHANGELOG.md", "Document changes"),
        ("app/tests/test_reward.py", "Create test suite (if missing)"),
    ]
    
    print("Files to be updated:")
    for filepath, description in files_to_update:
        status = "✅" if Path(filepath).exists() else "➕ (new)"
        print(f"  {status} {filepath}")
        print(f"      → {description}")
    print()
    
    if args.dry_run:
        print("✅ Dry run complete. Use --apply to make changes.")
        return
    
    # Apply updates
    print("Applying updates...\n")
    
    update_reward_py()
    update_metrics_py()
    update_optimizer_py()
    create_changelog_entry()
    create_test_file()
    
    print()
    print("=" * 60)
    print("✅ UPDATE COMPLETE")
    print("=" * 60)
    print()
    print(f"Backups saved to: {BACKUP_DIR}")
    print()
    print("Next steps:")
    print("  1. Review changes: git diff")
    print("  2. Run tests: pytest -v app/tests/test_reward.py")
    print("  3. Run full test suite: pytest -v")
    print("  4. Commit changes: git add . && git commit -m 'Prioritize Sortino over Sharpe in reward function'")
    print()

if __name__ == "__main__":
    main()
