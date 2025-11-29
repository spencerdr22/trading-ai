import argparse
import json
import logging
import os
import pandas as pd
from datetime import datetime
from apscheduler.schedulers.background import BackgroundScheduler

from .config import load_config
from .data.simulator import stream_bars
from .data.loader import load_sample
from .backtest.backtester import Backtester
from .monitor.logger import get_logger
from app.db.init import get_engine, get_session
from .models.schema import Base, Metric
from .ml.trainer import Trainer
from .ml.training import retrain_on_recent

logger = get_logger(__name__)
cfg = load_config()
engine = get_engine()
session = get_session()

# === Database Initialization ===
def init_db():
    Base.metadata.create_all(bind=engine)
    logger.info("DB initialized with ORM tables.")

# === Simulation ===
def simulate_mode(args):
    minutes = args.minutes or 1440
    bars = list(stream_bars(symbol=args.symbol, minutes=minutes, fast=args.fast, seed=cfg["simulator"]["seed"]))
    df = pd.DataFrame(bars)
    if "timestamp" not in df.columns:
        df["timestamp"] = pd.date_range(datetime.utcnow(), periods=len(df), freq="T")
    out = os.path.join(os.getcwd(), "data", f"sim_{args.symbol}.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_csv(out, index=False)
    logger.info("Simulated %d bars to %s", len(df), out)
    return df

# === Backtesting ===
def backtest_mode(args):
    if args.fast:
        df = load_sample()
    else:
        p = os.path.join(os.getcwd(), "data", f"sim_{args.symbol}.csv")
        df = pd.read_csv(p, parse_dates=["timestamp"]) if os.path.exists(p) else load_sample()
    b = Backtester(cfg)
    res = b.run(df)
    logger.info("Backtest complete. Win rate: %.2f, Max Drawdown: %.2f", res["win_rate"], res["max_drawdown"])
    import joblib
    out = os.path.join(os.getcwd(), "data", f"backtest_{args.symbol}.pkl")
    joblib.dump(res, out)
    logger.info("Saved backtest results to %s", out)

# === Forward / Paper Testing with Logging & Metrics ===
def forward_mode(args):
    """
    Paper trading mode — runs continuously on live or simulated data.
    Trades are logged, PnL tracked, retrained periodically, and results exported.
    """
    import time
    import pandas as pd
    from sqlalchemy import select
    from .execution.paper_executor import PaperExecutor
    from .ml.trainer import Trainer
    from .ml.training import retrain_on_recent
    from .strategy.engine import StrategyEngine
    from .strategy.adaption import Adaptor
    from .ml.features import make_features
    from .data.loader import load_sample
    from .data.simulator import stream_bars
    from .db import engine, get_session
    from .models.schema import TradeMetric

    logger.info("=== Forward Paper Trading Mode ===")
    logger.info("Running in simulated/paper mode — no live funds required.")

    # === Step 1: Load or Simulate Data ===
    df = load_sample()
    if df is None or len(df) < 120:
        logger.warning("Sample data too small. Generating simulated data...")
        bars = list(stream_bars(symbol=args.symbol, minutes=1440, fast=True, seed=42))
        df = pd.DataFrame(bars)
        logger.info(f"Simulated {len(df)} bars for forward test.")

    # === Step 2: Train initial model ===
    trainer = Trainer()
    model = trainer.train(df)
    if model is None:
        logger.warning("Not enough data to train model — skipping forward run.")
        return

    adaptor = Adaptor()
    strat = StrategyEngine(model, adaptor)
    exe = PaperExecutor(cfg)

    feat = make_features(df)
    X = feat.drop(columns=["timestamp", "open", "high", "low", "close", "volume"], errors="ignore")

    trades = []
    trade_count = 0
    retrain_interval = 100  # retrain every 100 trades
    start_time = time.time()
    retrain_time = 3600  # retrain every hour
    last_retrain = start_time

    for i, row in feat.iterrows():
        signal = strat.on_bar(X.loc[[i]])
        exe.place_order(row, signal)
        trade_count += 1

        # Log trade (in memory + DB)
        trade_entry = {
            "timestamp": row["timestamp"],
            "symbol": args.symbol,
            "side": signal.get("side") if isinstance(signal, dict) else signal,
            "confidence": signal.get("confidence") if isinstance(signal, dict) else None,
            "pnl": row.get("pnl", 0.0),
            "status": "FILLED"
        }
        trades.append(trade_entry)

        try:
            with get_session() as s:
                t = TradeMetric(
                    symbol=args.symbol,
                    timestamp=row["timestamp"],
                    side={"side": trade_entry["side"], "confidence": trade_entry["confidence"]},
                    pnl=trade_entry["pnl"],
                    status="FILLED",
                )
                s.add(t)
                s.commit()
        except Exception as e:
            logger.error(f"Trade logging failed: {e}")

        # Retraining logic
        now = time.time()
        if trade_count % retrain_interval == 0 or (now - last_retrain > retrain_time):
            logger.info("Triggering periodic retrain_on_recent()...")
            retrain_on_recent(n_bars=1440)
            last_retrain = now
            trade_count = 0

        # Simulate 0.1s per bar
        time.sleep(0.1)

    # === Step 3: Save Forward Results to CSV ===
    out_dir = os.path.join(os.getcwd(), "data")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"forward_{args.symbol}.csv")
    pd.DataFrame(trades).to_csv(out_path, index=False)
    logger.info(f"Saved forward results to {out_path}")

    # === Step 4: Export DB trades to CSV for backup ===
    try:
        with get_session() as s:
            rows = s.execute(select(TradeMetric)).scalars().all()
            if rows:
                db_export_path = os.path.join(out_dir, f"db_export_{args.symbol}.csv")
                # when exporting DB rows to CSV:
                pd.DataFrame([{
                    "id": t.id,
                    "symbol": t.symbol,
                    "timestamp": t.timestamp,
                    "side": (t.side if isinstance(t.side, str) else json.dumps(t.side)),
                    "pnl": t.pnl,
                    "status": t.status
                } for t in rows]).to_csv(db_export_path, index=False)
            else:
                logger.warning("No trades found in DB to export.")
    except Exception as e:
        logger.error(f"DB export failed: {e}")

    logger.info(f"Paper run complete. Executed {len(trades)} trades.")
    out = os.path.join("data", f"forward_results_{args.symbol}.csv")
    pd.DataFrame(exe.positions).to_csv(out, index=False)
    logger.info(f"Saved forward results to {out}")



# === Live Trading Placeholder ===
def live_mode(args):
    from .execution.tradovate_client import TradovateAPI
    api = TradovateAPI()
    if not api.ready:
        logger.error("Tradovate credentials missing. Live mode disabled.")
        return
    logger.info("Live mode started (stub).")

# === Live Feed Simulation ===
def livefeed_mode(args):
    import asyncio
    from .data.live_feed import LiveFeed
    from .execution.tradovate_client import MockTradovate
    client = MockTradovate()
    feed = LiveFeed(symbol=args.symbol, tradovate_client=client)
    try:
        asyncio.run(feed.stream())
    except KeyboardInterrupt:
        logger.info("Livefeed stopped by user.")

# === Retraining (manual or scheduled) ===
def retrain_on_recent(n_bars: int = 1440):
    """Reloads most recent data, retrains model, logs metrics."""
    try:
        logger.info("Starting retraining on recent data...")
        df = load_sample()
        trainer = Trainer()
        result = trainer.train(df)
        win_rate = getattr(result, "win_rate", 0.0)

        with get_session() as s:
            s.add(Metric(
                name="nightly_retrain",
                value=win_rate,
                timestamp=datetime.utcnow()
            ))
            s.commit()
        logger.info("Retraining complete. Win rate: %.2f", win_rate)
    except Exception as e:
        logger.error(f"Retraining failed: {e}")

def retrain_mode(args):
    retrain_on_recent()

# === Scheduler Setup ===
def schedule_jobs():
    scheduler = BackgroundScheduler()
    scheduler.add_job(retrain_on_recent, "cron", hour=0, minute=0)
    scheduler.start()
    logger.info("Scheduler started — nightly retrain at 00:00 UTC.")

# === CLI and Main Entrypoint ===
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["simulate", "backtest", "forward", "live", "livefeed", "init_db", "retrain"], required=True)
    p.add_argument("--fast", action="store_true")
    p.add_argument("--symbol", default=cfg.get("symbol", "MES"))
    p.add_argument("--minutes", type=int, default=1440)
    p.add_argument("--start")
    p.add_argument("--end")
    p.add_argument("--use-pytorch", action="store_true", help="Use PyTorch LSTM model")
    return p.parse_args()

def main():
    args = parse_args()
    if args.use_pytorch:
        cfg["model"]["use_pytorch"] = True

    if args.mode == "init_db":
        init_db()
        schedule_jobs()
    elif args.mode == "simulate":
        simulate_mode(args)
    elif args.mode == "backtest":
        backtest_mode(args)
    elif args.mode == "forward":
        forward_mode(args)
    elif args.mode == "live":
        live_mode(args)
    elif args.mode == "livefeed":
        livefeed_mode(args)
    elif args.mode == "retrain":
        retrain_mode(args)

if __name__ == "__main__":
    main()
