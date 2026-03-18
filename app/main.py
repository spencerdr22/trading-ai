import argparse
import json
import itertools
import os
import time
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pkg_resources")

from datetime import datetime, timezone as _tz
from apscheduler.schedulers.background import BackgroundScheduler

from .config import load_config
from .data.simulator import stream_bars
from .data.loader import load_sample
from .backtest.backtester import Backtester
from .monitor.logger import get_logger
from app.db.init import get_engine, get_session
from .models.schema import Base, Metric
from .ml.trainer import Trainer

logger = get_logger(__name__)
cfg    = load_config()
engine = get_engine()


# ── DB init ───────────────────────────────────────────────────────────────────

def init_db():
    Base.metadata.create_all(bind=engine)
    logger.info("DB initialized with ORM tables.")


# ── Simulate ──────────────────────────────────────────────────────────────────

def simulate_mode(args):
    minutes = args.minutes or 1440
    bars = list(stream_bars(
        symbol     = args.symbol,
        minutes    = minutes,
        fast       = args.fast,
        seed       = cfg["simulator"]["seed"],
    ))
    df = pd.DataFrame(bars)
    if "timestamp" not in df.columns:
        df["timestamp"] = pd.date_range(datetime.utcnow(), periods=len(df), freq="T")
    out = os.path.join(os.getcwd(), "data", f"sim_{args.symbol}.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_csv(out, index=False)
    logger.info("Simulated %d bars to %s", len(df), out)
    return df


# ── Backtest ──────────────────────────────────────────────────────────────────

def backtest_mode(args):
    if args.fast:
        df = load_sample()
    else:
        p  = os.path.join(os.getcwd(), "data", f"sim_{args.symbol}.csv")
        df = pd.read_csv(p, parse_dates=["timestamp"]) if os.path.exists(p) else load_sample()
    b   = Backtester(cfg)
    res = b.run(df)
    logger.info("Backtest complete. Win rate: %.2f  Max DD: %.4f",
                res["win_rate"], res["max_drawdown"])
    import joblib
    out = os.path.join(os.getcwd(), "data", f"backtest_{args.symbol}.pkl")
    joblib.dump(res, out)
    logger.info("Saved backtest results to %s", out)


# ── Forward / Paper trading ───────────────────────────────────────────────────

def forward_mode(args):
    """
    Continuous bar-by-bar trading loop.

    Data flow each bar:
      load_sample -> Trainer -> StrategyEngine -> sentiment filter
      -> Alpaca paper (or Tradovate live) + local PaperExecutor log
    """
    from sqlalchemy import select
    from .execution.paper_executor import PaperExecutor
    from .strategy.engine import StrategyEngine
    from .strategy.adaption import Adaptor
    from .ml.features import make_features
    from .models.schema import TradeMetric
    from .llm.news_feeds import NewsFeedManager
    from .llm.news_analyzer import NewsFlowAnalyzer

    logger.info("=== Forward Trading Mode ===")

    # ── Step 1: Data ──────────────────────────────────────────────────
    df = load_sample()
    if df is None or len(df) < 120:
        logger.warning("Sample data too small — generating simulated data...")
        bars = list(stream_bars(symbol=args.symbol, minutes=1440, fast=True, seed=42))
        df   = pd.DataFrame(bars)
        logger.info("Simulated %d bars.", len(df))

    # ── Step 2: Train ─────────────────────────────────────────────────
    trainer = Trainer()
    model   = trainer.train(df)
    if model is None:
        logger.error("Model training failed — cannot start forward mode.")
        return

    adaptor = Adaptor()
    strat   = StrategyEngine(model, adaptor, cfg)

    # ── Step 3: Execution client ──────────────────────────────────────
    use_live   = getattr(args, "live",   False)
    use_alpaca = getattr(args, "alpaca", False)
    live_client = None

    if use_live:
        try:
            from .execution.tradovate_client import TradovateAPI
            client = TradovateAPI()
            if client.ready:
                acct = client.get_account()
                logger.info("LIVE Tradovate ACTIVE | balance=$%.2f",
                            acct.get("cashBalance", 0))
                live_client = client
            else:
                logger.error("Tradovate credentials missing — check .env")
        except Exception as e:
            logger.error("Tradovate init failed: %s", e)

    elif use_alpaca:
        try:
            from .execution.alpaca_paper import get_alpaca_client
            client = get_alpaca_client()
            acct   = client.get_account()
            if acct:
                logger.info("Alpaca paper ACTIVE | portfolio=$%.2f",
                            float(acct.get("portfolio_value", 0)))
                live_client = client
            else:
                logger.warning("Alpaca account check failed — simulator only.")
        except Exception as e:
            logger.error("Alpaca init failed: %s", e)

    exe = PaperExecutor(cfg)  # always logs locally

    # ── Step 4: Sentiment ─────────────────────────────────────────────
    news_manager    = NewsFeedManager(symbols=[args.symbol])
    analyzer        = NewsFlowAnalyzer()
    sentiment_score = 0.0
    sentiment_label = "neutral"
    SENTIMENT_REFRESH_BARS = 50

    def refresh_sentiment():
        nonlocal sentiment_score, sentiment_label
        try:
            logger.info("Fetching news headlines...")
            headlines_df = news_manager.get_recent_headlines(hours=4)
            if headlines_df.empty:
                logger.info("No headlines — holding sentiment at %s.", sentiment_label)
                return
            headlines    = headlines_df["headline"].dropna().tolist()
            logger.info("Fetched %d headlines — running Qwen3 sentiment...", len(headlines))
            sentiment_df = analyzer.analyze_batch(headlines, symbol=args.symbol)
            agg          = analyzer.get_aggregated_sentiment(sentiment_df)
            sentiment_score = agg["sentiment_score"]
            sentiment_label = agg["overall_sentiment"]
            logger.info(
                "Sentiment: %s (score=%.3f  bull=%.0f%%  bear=%.0f%%)",
                sentiment_label.upper(), sentiment_score,
                agg["bullish_pct"] * 100, agg["bearish_pct"] * 100,
            )
        except Exception as e:
            logger.warning("Sentiment refresh failed: %s — keeping previous.", e)

    refresh_sentiment()

    # ── Step 5: Feature matrix ────────────────────────────────────────
    feat = make_features(df)
    X    = feat.drop(
        columns=["timestamp", "open", "high", "low", "close", "volume"],
        errors="ignore",
    )

    # ── Step 6: Live bar helper ───────────────────────────────────────
    def get_current_bar(row):
        if live_client:
            live = live_client.get_latest_bar()
            if live:
                for col in ["open", "high", "low", "close", "volume"]:
                    if col in row.index:
                        row[col] = live[col]
                row["timestamp"] = live["timestamp"]
        return row

    # ── Step 7: Main loop ─────────────────────────────────────────────
    trades       = []
    trade_count  = 0
    bar_count    = 0
    last_retrain = time.time()
    RETRAIN_BARS = 100
    RETRAIN_SECS = 3600
    BAR_SLEEP    = 60 if live_client else 0.1

    # Live: cycle indefinitely on real prices. Sim: single pass.
    row_source = (
        itertools.cycle(feat.iterrows()) if live_client
        else feat.iterrows()
    )

    for i, row in row_source:
        bar_count += 1

        if bar_count % SENTIMENT_REFRESH_BARS == 0:
            refresh_sentiment()

        row    = get_current_bar(row)
        signal = strat.on_bar(X.loc[[i]])

        # Sentiment filter
        if isinstance(signal, dict):
            side = signal.get("side", "HOLD")
            if side == "BUY" and sentiment_score < -0.15:
                signal["side"] = "HOLD"
                signal["sentiment_override"] = True
                logger.debug("Bar %d: BUY -> HOLD (bearish %.2f)", bar_count, sentiment_score)
            elif side == "SELL" and sentiment_score > 0.15:
                signal["side"] = "HOLD"
                signal["sentiment_override"] = True
                logger.debug("Bar %d: SELL -> HOLD (bullish %.2f)", bar_count, sentiment_score)

        # Execute
        if live_client:
            live_client.execute_mes_signal(signal, row.to_dict())
        exe.place_order(row, signal)
        trade_count += 1

        # Log trade
        trade_entry = {
            "timestamp":       row["timestamp"],
            "symbol":          args.symbol,
            "side":            signal.get("side") if isinstance(signal, dict) else signal,
            "confidence":      signal.get("confidence") if isinstance(signal, dict) else None,
            "sentiment":       sentiment_label,
            "sentiment_score": sentiment_score,
            "pnl":             float(row.get("pnl", 0.0)),
            "status":          "FILLED",
        }
        trades.append(trade_entry)

        try:
            # Normalise timestamp — Alpaca returns ISO strings, pandas returns Timestamps
            raw_ts = row["timestamp"]
            if isinstance(raw_ts, str):
                raw_ts = datetime.fromisoformat(
                    raw_ts.replace("Z", "+00:00")
                ).replace(tzinfo=None)
            elif hasattr(raw_ts, "to_pydatetime"):
                raw_ts = raw_ts.to_pydatetime().replace(tzinfo=None)
            elif isinstance(raw_ts, datetime) and raw_ts.tzinfo is not None:
                raw_ts = raw_ts.replace(tzinfo=None)
            with get_session() as s:
                s.add(TradeMetric(
                    symbol    = args.symbol,
                    timestamp = raw_ts,
                    side      = {"side": trade_entry["side"],
                                 "confidence": trade_entry["confidence"]},
                    pnl       = trade_entry["pnl"],
                    status    = "FILLED",
                ))
                s.commit()
        except Exception as e:
            logger.error("Trade DB log failed: %s", e)

        # Periodic retrain
        now = time.time()
        if trade_count % RETRAIN_BARS == 0 or (now - last_retrain) > RETRAIN_SECS:
            logger.info("Intraday quick retrain triggered...")
            try:
                from .training_pipeline import quick_retrain
                quick_retrain()
            except Exception as e:
                logger.warning("Quick retrain failed: %s", e)
            last_retrain = now
            trade_count  = 0

        time.sleep(BAR_SLEEP)

    # ── Step 8: Save results ──────────────────────────────────────────
    out_dir = os.path.join(os.getcwd(), "data")
    os.makedirs(out_dir, exist_ok=True)

    trades_path = os.path.join(out_dir, f"forward_{args.symbol}.csv")
    pd.DataFrame(trades).to_csv(trades_path, index=False)
    logger.info("Saved %d trade records to %s", len(trades), trades_path)

    positions_path = os.path.join(out_dir, f"forward_results_{args.symbol}.csv")
    pd.DataFrame(exe.positions).to_csv(positions_path, index=False)
    logger.info("Saved position log to %s", positions_path)

    # DB export backup
    try:
        from sqlalchemy import select as sa_select
        with get_session() as s:
            rows = s.execute(sa_select(TradeMetric)).scalars().all()
            if rows:
                db_path = os.path.join(out_dir, f"db_export_{args.symbol}.csv")
                pd.DataFrame([{
                    "id":        t.id,
                    "symbol":    t.symbol,
                    "timestamp": t.timestamp,
                    "side":      t.side if isinstance(t.side, str) else json.dumps(t.side),
                    "pnl":       t.pnl,
                    "status":    t.status,
                } for t in rows]).to_csv(db_path, index=False)
                logger.info("DB export saved to %s", db_path)
    except Exception as e:
        logger.error("DB export failed: %s", e)

    logger.info("Forward run complete. %d bars processed.", bar_count)


# ── Live mode (Tradovate direct) ──────────────────────────────────────────────

def live_mode(args):
    """Tradovate live — delegates to forward_mode with --live flag."""
    args.live = True
    forward_mode(args)


# ── Livefeed simulation ───────────────────────────────────────────────────────

def livefeed_mode(args):
    import asyncio
    from .data.live_feed import LiveFeed
    from .execution.tradovate_client import MockTradovate
    client = MockTradovate()
    feed   = LiveFeed(symbol=args.symbol, tradovate_client=client)
    try:
        asyncio.run(feed.stream())
    except KeyboardInterrupt:
        logger.info("Livefeed stopped.")


# ── Manual retrain ────────────────────────────────────────────────────────────

def retrain_mode(args):
    from .training_pipeline import run_pipeline
    result = run_pipeline(run_rl=True, run_hparam=True)
    logger.info("Manual retrain complete: %s", result)


# ── Scheduler ─────────────────────────────────────────────────────────────────

def schedule_jobs():
    from .training_pipeline import run_pipeline
    from .llm.monitor import log_system_status
    scheduler = BackgroundScheduler()
    scheduler.add_job(
        run_pipeline, "cron", hour=0, minute=0,
        kwargs={"run_rl": True, "run_hparam": True},
    )
    scheduler.add_job(log_system_status, "interval", minutes=15)
    scheduler.start()
    logger.info("Scheduler started: nightly pipeline at 00:00 UTC, status every 15 min.")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Trading-AI — MES paper/live trading")
    p.add_argument("--mode", required=True,
                   choices=["simulate", "backtest", "forward", "live",
                            "livefeed", "init_db", "retrain"])
    p.add_argument("--symbol",      default=cfg.get("symbol", "MES"))
    p.add_argument("--minutes",     type=int, default=1440)
    p.add_argument("--fast",        action="store_true")
    p.add_argument("--start")
    p.add_argument("--end")
    p.add_argument("--use-pytorch", action="store_true")
    p.add_argument("--alpaca",      action="store_true",
                   help="Route signals to Alpaca paper trading")
    p.add_argument("--live",        action="store_true",
                   help="Route signals to Tradovate live/demo trading")
    return p.parse_args()


def main():
    args = parse_args()

    if args.use_pytorch:
        cfg["model"]["use_pytorch"] = True

    dispatch = {
        "init_db":  lambda: (init_db(), schedule_jobs()),
        "simulate": lambda: simulate_mode(args),
        "backtest": lambda: backtest_mode(args),
        "forward":  lambda: forward_mode(args),
        "live":     lambda: live_mode(args),
        "livefeed": lambda: livefeed_mode(args),
        "retrain":  lambda: retrain_mode(args),
    }
    dispatch[args.mode]()


if __name__ == "__main__":
    main()
