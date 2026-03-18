"""
run_offline_rl.py — CLI runner for offline RL retraining + Optuna tuning.

Usage:
    python -m app.adaptive.run_offline_rl --episodes 10
    python -m app.adaptive.run_offline_rl --episodes 25 --tune
"""

import argparse
import os
import datetime

from .learner import ReinforcementLearner
from .optimizer import RLHyperOptimizer
from ..monitor.logger import get_logger

logger = get_logger(__name__)

# ── Plotly with ASCII fallback ────────────────────────────────────────────────
try:
    import plotly.graph_objects as go
    _PLOTLY = True
except ImportError:
    _PLOTLY = False


def _save_plot(y: list, title: str, file_prefix: str):
    """Save an HTML line chart (Plotly if available, plain HTML otherwise)."""
    os.makedirs("data/plots", exist_ok=True)
    ts        = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    file_path = f"data/plots/{file_prefix}_{ts}.html"

    if _PLOTLY:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(x=list(range(len(y))), y=y, mode="lines+markers")
        )
        fig.update_layout(
            title       = title,
            xaxis_title = "Episode",
            yaxis_title = "Value",
        )
        fig.write_html(file_path)
    else:
        # Minimal HTML table fallback — no external dependencies
        rows = "".join(
            f"<tr><td>{i}</td><td>{v:.6f}</td></tr>" for i, v in enumerate(y)
        )
        with open(file_path, "w", encoding="utf-8") as fh:
            fh.write(
                f"<html><head><title>{title}</title></head><body>"
                f"<h2>{title}</h2><table border='1'>"
                f"<tr><th>Step</th><th>Value</th></tr>{rows}"
                f"</table></body></html>"
            )

    logger.info("Plot saved: %s", file_path)


# ── Bar-chart variant for hyperparams ─────────────────────────────────────────
def _save_bar_plot(keys: list, values: list, title: str, file_prefix: str):
    os.makedirs("data/plots", exist_ok=True)
    ts        = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    file_path = f"data/plots/{file_prefix}_{ts}.html"

    if _PLOTLY:
        fig = go.Figure(go.Bar(x=keys, y=values))
        fig.update_layout(title=title, xaxis_title="Parameter", yaxis_title="Value")
        fig.write_html(file_path)
    else:
        rows = "".join(
            f"<tr><td>{k}</td><td>{v:.6f}</td></tr>" for k, v in zip(keys, values)
        )
        with open(file_path, "w", encoding="utf-8") as fh:
            fh.write(
                f"<html><head><title>{title}</title></head><body>"
                f"<h2>{title}</h2><table border='1'>"
                f"<tr><th>Param</th><th>Value</th></tr>{rows}"
                f"</table></body></html>"
            )

    logger.info("Hyperparameter plot saved: %s", file_path)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Offline RL retrainer")
    parser.add_argument("--feature-dim", type=int, default=4,
                        help="RL state vector dimension (default 4)")
    parser.add_argument("--episodes",    type=int, default=10,
                        help="REINFORCE episodes per session (default 10)")
    parser.add_argument("--tune",        action="store_true",
                        help="Run Optuna hyperparameter search after RL")
    args = parser.parse_args()

    logger.info("=" * 50)
    logger.info("OFFLINE RL RETRAINING STARTED")
    logger.info("=" * 50)

    # ── RL training ───────────────────────────────────────────
    learner = ReinforcementLearner(feature_dim=args.feature_dim)
    learner.load_latest_policy()

    result = learner.train_from_history(episodes=args.episodes)

    if not result:
        logger.warning("RL training aborted — insufficient trade history (need >= 20 trades).")
        return

    _save_plot(result["losses"], "RL Training Loss",  "rl_loss_curve")
    _save_plot([result["reward"]] * 2, f"RL Reward {result['reward']:.4f}", "rl_reward")

    # ── Optional Optuna tuning ────────────────────────────────
    if args.tune:
        logger.info("Running Optuna hyperparameter search...")
        optimizer  = RLHyperOptimizer(feature_dim=args.feature_dim)
        best, score = optimizer.optimize(n_trials=5)

        _save_bar_plot(
            list(best.keys()),
            [float(v) for v in best.values()],
            f"Best Hyperparams (score={score:.3f})",
            "rl_hyperparams",
        )
        logger.info("Optuna best score=%.4f  params=%s", score, best)

    logger.info("=" * 50)
    logger.info("OFFLINE RL RETRAINING COMPLETE")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
