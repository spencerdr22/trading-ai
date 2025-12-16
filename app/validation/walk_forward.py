#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
walk_forward.py

Walk-forward validation for time-series strategies.
Implements anchored and rolling window walk-forward optimization.

Version 2.0: Updated to support VectorbtEngine for accurate backtesting.
"""

import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable, Any

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit


def walk_forward_split(
    n_samples: int,
    n_splits: int = 5,
    test_size: Optional[int] = None,
    mode: str = 'rolling'
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Generate walk-forward train/test splits.
    
    Args:
        n_samples: Total number of samples
        n_splits: Number of splits
        test_size: Size of test set (None = auto)
        mode: 'rolling' (fixed train size) or 'anchored' (expanding train size)
        
    Returns:
        List of (train_indices, test_indices) tuples
    """
    if mode == 'anchored':
        # Anchored: training set expands over time
        tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size)
        return list(tscv.split(np.arange(n_samples)))
    elif mode == 'rolling':
        # Rolling: fixed training window size
        if test_size is None:
            test_size = n_samples // (n_splits + 1)
        
        train_size = n_samples // (n_splits + 1)
        splits = []
        
        for i in range(n_splits):
            test_start = (i + 1) * train_size
            test_end = test_start + test_size
            
            if test_end > n_samples:
                break
            
            train_start = max(0, test_start - train_size)
            train_indices = np.arange(train_start, test_start)
            test_indices = np.arange(test_start, test_end)
            
            splits.append((train_indices, test_indices))
        
        return splits
    else:
        raise ValueError(f"Unknown mode: {mode}")


def walk_forward_validation(
    df: pd.DataFrame,
    train_func: Callable,
    test_func: Callable,
    n_splits: int = 5,
    test_size: Optional[int] = None,
    mode: str = 'anchored',
    output_dir: Optional[str] = None,
    strategy_id: Optional[str] = None,
    engine: str = 'vectorbt'
) -> Dict:
    """
    Perform walk-forward validation.
    
    Args:
        df: DataFrame with features and labels
        train_func: Function(train_df) -> model that trains and returns a model
        test_func: Function(model, test_df) -> metrics that tests and returns metrics dict
        n_splits: Number of walk-forward splits
        test_size: Size of test set
        mode: 'anchored' or 'rolling'
        output_dir: Directory to save results
        strategy_id: Strategy identifier
        engine: 'vectorbt' (recommended) or 'legacy' for BacktestEngine
        
    Returns:
        Dictionary with walk-forward results
        
    Note:
        Using engine='vectorbt' is recommended for accurate backtesting results.
        The 'legacy' engine is deprecated and may produce different metrics.
    """
    # Warn if using legacy engine
    if engine == 'legacy':
        warnings.warn(
            "Using deprecated legacy BacktestEngine. Consider switching to "
            "engine='vectorbt' for more accurate results.",
            DeprecationWarning,
            stacklevel=2
        )
    
    print(f"\n🚶 WALK-FORWARD VALIDATION")
    print(f"{'='*60}")
    print(f"Mode: {mode}")
    print(f"Splits: {n_splits}")
    print(f"Engine: {engine}")
    print(f"Total samples: {len(df)}")
    
    # Generate splits
    splits = walk_forward_split(len(df), n_splits, test_size, mode)
    
    print(f"\n📊 Generated {len(splits)} walk-forward periods:")
    for i, (train_idx, test_idx) in enumerate(splits, 1):
        print(f"  Period {i}: Train[{len(train_idx)}] -> Test[{len(test_idx)}]")
    
    # Run walk-forward
    results = []
    all_test_metrics = []
    
    for period, (train_idx, test_idx) in enumerate(splits, 1):
        print(f"\n{'='*60}")
        print(f"Period {period}/{len(splits)}")
        print(f"{'='*60}")
        
        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]
        
        print(f"Training on {len(train_df)} samples...")
        model = train_func(train_df)
        
        print(f"Testing on {len(test_df)} samples...")
        test_metrics = test_func(model, test_df)
        
        period_result = {
            'period': period,
            'train_size': len(train_idx),
            'test_size': len(test_idx),
            'train_start_idx': int(train_idx[0]),
            'train_end_idx': int(train_idx[-1]),
            'test_start_idx': int(test_idx[0]),
            'test_end_idx': int(test_idx[-1]),
            'metrics': test_metrics
        }
        
        results.append(period_result)
        all_test_metrics.append(test_metrics)
        
        print(f"\nPeriod {period} Results:")
        for key, value in test_metrics.items():
            if isinstance(value, (int, float)):
                print(f"  {key}: {value:.4f}")
    
    # Aggregate metrics
    print(f"\n{'='*60}")
    print(f"AGGREGATE RESULTS ACROSS ALL PERIODS")
    print(f"{'='*60}")
    
    aggregated = {}
    if all_test_metrics:
        # Get all metric keys
        metric_keys = all_test_metrics[0].keys()
        
        for key in metric_keys:
            values = [m[key] for m in all_test_metrics if isinstance(m.get(key), (int, float))]
            if values:
                aggregated[key] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'periods': values
                }
                print(f"\n{key}:")
                print(f"  Mean: {aggregated[key]['mean']:.4f} ± {aggregated[key]['std']:.4f}")
                print(f"  Range: [{aggregated[key]['min']:.4f}, {aggregated[key]['max']:.4f}]")
    
    output = {
        'mode': mode,
        'n_splits': len(splits),
        'periods': results,
        'aggregated_metrics': aggregated,
        'stability': {
            key: 'stable' if aggregated[key]['std'] / abs(aggregated[key]['mean']) < 0.3 
            else 'unstable'
            for key in aggregated.keys()
            if aggregated[key]['mean'] != 0
        }
    }
    
    # Save results
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        filename = f"walk_forward_{strategy_id}.json" if strategy_id else "walk_forward_results.json"
        results_file = output_path / filename
        
        with open(results_file, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"\n💾 Results saved to: {results_file}")
    
    print(f"\n✅ Walk-forward validation complete!")
    
    return output


if __name__ == "__main__":
    print("Walk-forward validation module loaded.")
    print("Use walk_forward_validation() with custom train_func and test_func.")
