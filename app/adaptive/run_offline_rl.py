"""
Module: run_offline_rl.py
Author: Adaptive Framework Generator

Description:
    Standalone CLI runner for offline reinforcement-learning retraining.
    Integrates with ReinforcementLearner, RLHyperOptimizer, and ModelHub.
    Saves Plotly visualizations (reward & loss curves) for analysis.

Usage:
    python app/adaptive/run_offline_rl.py --episodes 25 --tune
"""

import argparse
import os
import datetime
import numpy as np
# Attempt to import plotly; if unavailable provide a minimal fallback to avoid import errors.
try:
    import plotly.graph_objects as go
except Exception:
    # Minimal fallback implementation that mimics the parts of plotly used in this module.
    class _FallbackFigure:
        def __init__(self):
            self.traces = []
            self.layout = {}

        def add_trace(self, trace):
            self.traces.append(trace)

        def update_layout(self, **kwargs):
            self.layout.update(kwargs)

        def write_html(self, path):
            # Create a very simple HTML representation of the data so callers won't fail.
            title = self.layout.get("title", "plot")
            xaxis = self.layout.get("xaxis_title", "x")
            yaxis = self.layout.get("yaxis_title", "y")
            with open(path, "w", encoding="utf-8") as f:
                f.write(f"<html><head><meta charset='utf-8'><title>{title}</title></head><body>")
                f.write(f"<h2>{title}</h2>")
                f.write(f"<p><strong>{xaxis}</strong> vs <strong>{yaxis}</strong></p>")
                f.write("<pre>")
                for t in self.traces:
                    f.write(str(t))
                    f.write("\n")
                f.write("</pre></body></html>")

    class _FallbackTrace(dict):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)

        def __repr__(self):
            # Provide concise readable representation for fallback HTML
            items = ", ".join(f"{k}={v}" for k, v in self.items())
            return f"Trace({items})"

    class _GoModule:
        def Figure(self):
            return _FallbackFigure()

        def Scatter(self, **kwargs):
            return _FallbackTrace(type="scatter", **kwargs)

        def Bar(self, **kwargs):
            return _FallbackTrace(type="bar", **kwargs)

    go = _GoModule()

from .learner import ReinforcementLearner
from .optimizer import RLHyperOptimizer
from ..monitor.logger import get_logger

logger = get_logger(__name__)


# -------------------------------------------------------------------
# Plotting helpers
# -------------------------------------------------------------------

def save_plot(x, y, title, file_prefix):
    """
    Saves a Plotly HTML interactive chart.
    """
    os.makedirs("data/plots", exist_ok=True)
    timestamp = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    file_path = f"data/plots/{file_prefix}_{timestamp}.html"

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(len(y))), y=y, mode="lines+markers"))
    fig.update_layout(
        title=title,
        xaxis_title="Episode",
        yaxis_title="Value",
        template="plotly_white"
    )

    fig.write_html(file_path)
    logger.info(f"Plot saved: {file_path}")


# -------------------------------------------------------------------
# MAIN EXECUTION LOGIC
# -------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Offline RL retrainer")

    parser.add_argument(
        "--feature-dim",
        type=int,
        required=False,
        default=4,
        help="Dimensionality of RL state vector"
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="Number of RL episodes to run"
    )

    parser.add_argument(
        "--tune",
        action="store_true",
        help="Enable Optuna hyperparameter tuning"
    )

    args = parser.parse_args()

    feature_dim = args.feature_dim
    episodes = args.episodes

    logger.info("============================================")
    logger.info("   OFFLINE ADAPTIVE RL RETRAINING STARTING   ")
    logger.info("============================================")

    # ----------------------------------------------------------
    # RL Learner
    # ----------------------------------------------------------
    learner = ReinforcementLearner(feature_dim=feature_dim)

    # Load latest policy if exists
    learner.load_latest_policy()

    # Run RL training
    result = learner.train_from_history(episodes=episodes)

    if not result:
        logger.warning("Offline RL training aborted — no data.")
        return

    reward = result["reward"]
    losses = result["losses"]

    # Plot losses
    save_plot(
        x=list(range(len(losses))),
        y=losses,
        title="RL Training Loss Curve",
        file_prefix="rl_loss_curve"
    )

    # Plot reward (static per training session)
    save_plot(
        x=[0, 1],
        y=[reward, reward],
        title=f"RL Reward: {reward:.4f}",
        file_prefix="rl_reward_curve"
    )

    # ----------------------------------------------------------
    # Hyperparameter Optimization (Optional)
    # ----------------------------------------------------------
    if args.tune:
        logger.info("Launching Optuna hyperparameter tuning...")
        optimizer = RLHyperOptimizer(feature_dim=feature_dim)

        best_params, best_score = optimizer.optimize(n_trials=5)

        # Save hyperparameter summary plot
        summary_x = list(best_params.keys())
        summary_y = list(best_params.values())

        timestamp = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        file_path = f"data/plots/rl_hyperparam_summary_{timestamp}.html"

        fig = go.Figure()
        fig.add_trace(go.Bar(x=summary_x, y=summary_y))
        fig.update_layout(
            title=f"Best Hyperparameters (Score {best_score:.3f})",
            xaxis_title="Hyperparameter",
            yaxis_title="Value",
            template="plotly_white"
        )
        fig.write_html(file_path)

        logger.info(f"Hyperparameter summary saved: {file_path}")

    logger.info("============================================")
    logger.info("   OFFLINE ADAPTIVE RL RETRAINING COMPLETE   ")
    logger.info("============================================")


if __name__ == "__main__":
    main()
