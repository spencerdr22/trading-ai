"""
Package: app.validation
Advanced validation utilities for trading-ai.
"""

from .walk_forward import walk_forward_validation, walk_forward_split
from .monte_carlo import monte_carlo_analysis, bootstrap_trades

__version__ = "1.0.0"
__all__ = [
    "walk_forward_validation",
    "walk_forward_split",
    "monte_carlo_analysis",
    "bootstrap_trades"
]
