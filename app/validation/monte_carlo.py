#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
monte_carlo.py

Monte Carlo analysis for backtest robustness assessment.
Implements bootstrap resampling of trades to generate equity curve distributions.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def bootstrap_trades(
    trades_df: pd.DataFrame,
    n_sequences: int = 500,
    block_size: Optional[int] = None,
    seed: int = 42
) -> List[pd.DataFrame]:
    """
    Bootstrap resample trades to create multiple equity sequences.
    
    Uses block bootstrap to preserve temporal structure if block_size is specified.
    
    Args:
        trades_df: DataFrame with trade results (must have 'R' column)
        n_sequences: Number of bootstrap sequences to generate
        block_size: Size of blocks for block bootstrap (None = standard bootstrap)
        seed: Random seed for reproducibility
        
    Returns:
        List of DataFrames, each representing one bootstrap sequence
    """
    np.random.seed(seed)
    n_trades = len(trades_df)
    sequences = []
    
    for _ in range(n_sequences):
        if block_size:
            # Block bootstrap
            n_blocks = int(np.ceil(n_trades / block_size))
            block_starts = np.random.choice(
                n_trades - block_size + 1, 
                size=n_blocks, 
                replace=True
            )
            indices = []
            for start in block_starts:
                indices.extend(range(start, min(start + block_size, n_trades)))
            indices = indices[:n_trades]  # Trim to original length
        else:
            # Standard bootstrap
            indices = np.random.choice(n_trades, size=n_trades, replace=True)
        
        bootstrap_seq = trades_df.iloc[indices].copy()
        bootstrap_seq['cumulative_R'] = bootstrap_seq['R'].cumsum()
        sequences.append(bootstrap_seq)
    
    return sequences


def calculate_sequence_metrics(sequence: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate performance metrics for a single bootstrap sequence.
    
    Args:
        sequence: DataFrame with 'R' and 'cumulative_R' columns
        
    Returns:
        Dictionary of metrics
    """
    returns = sequence['R'].values
    cumulative = sequence['cumulative_R'].values
    
    # Basic stats
    mean_r = float(np.mean(returns))
    std_r = float(np.std(returns))
    sharpe = mean_r / std_r if std_r > 0 else 0.0
    
    # Drawdown
    running_max = np.maximum.accumulate(cumulative)
    drawdown = cumulative - running_max
    max_dd = float(np.min(drawdown))
    
    # Final equity
    final_r = float(cumulative[-1])
    
    # Win rate
    win_rate = float(np.mean(returns > 0))
    
    return {
        'mean_R': mean_r,
        'std_R': std_r,
        'sharpe': sharpe,
        'max_drawdown': max_dd,
        'final_R': final_r,
        'win_rate': win_rate,
        'total_trades': len(returns)
    }


def monte_carlo_analysis(
    trades_df: pd.DataFrame,
    n_sequences: int = 500,
    block_size: Optional[int] = None,
    percentiles: List[int] = [5, 25, 50, 75, 95],
    output_dir: Optional[str] = None,
    strategy_id: Optional[str] = None
) -> Dict:
    """
    Perform Monte Carlo analysis on backtest trades.
    
    Args:
        trades_df: DataFrame with trade results (must have 'R' column)
        n_sequences: Number of bootstrap sequences
        block_size: Block size for temporal structure preservation
        percentiles: Percentile levels for equity curves
        output_dir: Directory to save results (None = don't save)
        strategy_id: Strategy identifier for file naming
        
    Returns:
        Dictionary with analysis results
    """
    print(f"\n🎲 MONTE CARLO ANALYSIS")
    print(f"{'='*60}")
    print(f"Original trades: {len(trades_df)}")
    print(f"Bootstrap sequences: {n_sequences}")
    if block_size:
        print(f"Block size: {block_size}")
    
    # Generate bootstrap sequences
    print(f"\n📊 Generating {n_sequences} bootstrap sequences...")
    sequences = bootstrap_trades(trades_df, n_sequences, block_size)
    
    # Calculate metrics for each sequence
    print(f"📈 Computing metrics for each sequence...")
    metrics_list = [calculate_sequence_metrics(seq) for seq in sequences]
    metrics_df = pd.DataFrame(metrics_list)
    
    # Summary statistics
    summary = {
        'mean_R': {
            'mean': float(metrics_df['mean_R'].mean()),
            'std': float(metrics_df['mean_R'].std()),
            'percentiles': {p: float(metrics_df['mean_R'].quantile(p/100)) 
                          for p in percentiles}
        },
        'sharpe': {
            'mean': float(metrics_df['sharpe'].mean()),
            'std': float(metrics_df['sharpe'].std()),
            'percentiles': {p: float(metrics_df['sharpe'].quantile(p/100)) 
                          for p in percentiles}
        },
        'max_drawdown': {
            'mean': float(metrics_df['max_drawdown'].mean()),
            'std': float(metrics_df['max_drawdown'].std()),
            'percentiles': {p: float(metrics_df['max_drawdown'].quantile(p/100)) 
                          for p in percentiles}
        },
        'final_R': {
            'mean': float(metrics_df['final_R'].mean()),
            'std': float(metrics_df['final_R'].std()),
            'percentiles': {p: float(metrics_df['final_R'].quantile(p/100)) 
                          for p in percentiles}
        },
        'win_rate': {
            'mean': float(metrics_df['win_rate'].mean()),
            'std': float(metrics_df['win_rate'].std()),
            'percentiles': {p: float(metrics_df['win_rate'].quantile(p/100)) 
                          for p in percentiles}
        }
    }
    
    # Equity curve percentiles
    max_length = max(len(seq) for seq in sequences)
    equity_matrix = np.full((n_sequences, max_length), np.nan)
    
    for i, seq in enumerate(sequences):
        equity_matrix[i, :len(seq)] = seq['cumulative_R'].values
    
    equity_percentiles = {}
    for p in percentiles:
        equity_percentiles[f'p{p}'] = np.nanpercentile(equity_matrix, p, axis=0).tolist()
    
    # Original equity curve
    original_equity = trades_df['R'].cumsum().tolist()
    
    results = {
        'n_sequences': n_sequences,
        'n_original_trades': len(trades_df),
        'block_size': block_size,
        'summary_metrics': summary,
        'equity_curves': {
            'original': original_equity,
            'percentiles': equity_percentiles
        },
        'stability_assessment': {
            'mean_R_stability': 'stable' if summary['mean_R']['std'] / abs(summary['mean_R']['mean']) < 0.3 else 'unstable',
            'sharpe_stability': 'stable' if summary['sharpe']['std'] / abs(summary['sharpe']['mean']) < 0.5 else 'unstable',
            'max_dd_variability': summary['max_drawdown']['std']
        }
    }
    
    # Print summary
    print(f"\n📊 RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"\nMean R-multiple:")
    print(f"  Mean: {summary['mean_R']['mean']:.3f} ± {summary['mean_R']['std']:.3f}")
    print(f"  5th-95th percentile: [{summary['mean_R']['percentiles'][5]:.3f}, {summary['mean_R']['percentiles'][95]:.3f}]")
    
    print(f"\nSharpe Ratio:")
    print(f"  Mean: {summary['sharpe']['mean']:.3f} ± {summary['sharpe']['std']:.3f}")
    print(f"  5th-95th percentile: [{summary['sharpe']['percentiles'][5]:.3f}, {summary['sharpe']['percentiles'][95]:.3f}]")
    
    print(f"\nMax Drawdown:")
    print(f"  Mean: {summary['max_drawdown']['mean']:.3f} ± {summary['max_drawdown']['std']:.3f}")
    print(f"  5th-95th percentile: [{summary['max_drawdown']['percentiles'][5]:.3f}, {summary['max_drawdown']['percentiles'][95]:.3f}]")
    
    print(f"\nFinal R:")
    print(f"  Mean: {summary['final_R']['mean']:.1f} ± {summary['final_R']['std']:.1f}")
    print(f"  5th-95th percentile: [{summary['final_R']['percentiles'][5]:.1f}, {summary['final_R']['percentiles'][95]:.1f}]")
    
    print(f"\n✅ Stability Assessment:")
    for key, value in results['stability_assessment'].items():
        print(f"  {key}: {value}")
    
    # Save results
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        filename = f"monte_carlo_{strategy_id}.json" if strategy_id else "monte_carlo_results.json"
        results_file = output_path / filename
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n💾 Results saved to: {results_file}")
        
        # Save detailed metrics
        metrics_file = output_path / filename.replace('.json', '_metrics.csv')
        metrics_df.to_csv(metrics_file, index=False)
        print(f"💾 Detailed metrics saved to: {metrics_file}")
    
    return results


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python monte_carlo.py <trades_csv_path> [strategy_id] [n_sequences]")
        sys.exit(1)
    
    trades_path = sys.argv[1]
    strategy_id = sys.argv[2] if len(sys.argv) > 2 else None
    n_sequences = int(sys.argv[3]) if len(sys.argv) > 3 else 500
    
    # Load trades
    trades_df = pd.read_csv(trades_path)
    
    if 'R' not in trades_df.columns:
        print("Error: trades CSV must have 'R' column")
        sys.exit(1)
    
    # Determine output directory
    output_dir = Path(trades_path).parent / 'monte_carlo'
    
    # Run analysis
    results = monte_carlo_analysis(
        trades_df=trades_df,
        n_sequences=n_sequences,
        output_dir=str(output_dir),
        strategy_id=strategy_id
    )
    
    print(f"\n✅ Monte Carlo analysis complete!")
