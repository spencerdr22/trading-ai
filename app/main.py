import argparse
import json
import os
import time
import collections
import pandas as pd
import numpy as np
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
from .ml.features import make_features

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


# ── Live feature buffer ───────────────────────────────────────────────────────

class LiveBarBuffer:
    """
    Maintains a rolling window of real live bars and recomputes features
    fresh on each new bar. This ensures the model always sees features
    derived from actual current market data, not cached simulated values.

    MIN_BARS: minimum bars needed before we produce features.
    MAX_BARS: maximum bars to keep in the rolling window.
    """

    MIN_BARS = 60    # need at least 60 bars for EMA-50 / ATR-50 to stabilise
    MAX_BARS = 500   # keep up to ~8 hours of 1-min bars

    def __init__(self):
        self._bars = collections.deque(maxlen=self.MAX_BARS)
        self._feature_cols = None   # determined on first successful feature build

    def push(self, bar: dict):
        """Add a new bar dict {timestamp, open, high, low, close, volume}."""
        self._bars.append({
            "timestamp": bar["timestamp"],
            "open":      float(bar["open"]),
            "high":      float(bar["high"]),
            "low":       float(bar["low"]),
            "close":     float(bar["close"]),
            "volume":    float(bar["volume"]),
        })

    def ready(self) -> bool:
        return len(self._bars) >= self.MIN_BARS

    def get_feature_row(self, trained_feature_cols: list):
        """
        Recompute features on the current rolling window and return a
        single-row DataFrame aligned to trained_feature_cols.
        Returns None if not enough bars yet.
        """
        if not self.ready():
            return None

        df = pd.DataFrame(list(self._bars))
        feat = make_features(df)

        # Take the LAST row (most recent bar)
        last = feat.iloc[[-1]].copy()

        # Align columns to what the model was trained on
        # Add any missing cols as 0, drop any extras
        for col in trained_feature_cols:
            if col not in last.columns:
                last[col] = 0.0
        last = last[trained_feature_cols]

        return last

    def get_trend_signal(self) -> int:
        """
        Simple trend filter based on EMA slope.
        Returns +1 (uptrend), -1 (downtrend), or 0 (neutral).
        Used to block counter-trend trades.
        """
        if len(self._bars) < 30:
            return 0
        closes = pd.Series([b["close"] for b in self._bars])
        ema_fast = closes.ewm(span=9,  adjust=False).mean()
        ema_slow = closes.ewm(span=21, adjust=False).mean()
        # Slope over last 5 bars
        slope = ema_fast.iloc[-1] - ema_fast.iloc[-5]
        cross = ema_fast.iloc[-1] - ema_slow.iloc[-1]
        if slope > 0 and cross > 0:
            return 1    # uptrend
        elif slope < 0 and cross < 0:
            return -1   # downtrend
        return 0        # neutral / choppy


# ── Forward / Paper trading ───────────────────────────────────────────────────

def forward_mode(args):
    """
    Continuous live trading loop.

    Key improvements over previous version:
      1. LiveBarBuffer — features recomputed from real live bars each tick.
      2. Trend filter — BUY only allowed in uptrend, SELL only in downtrend.
      3. Consecutive-signal cap — max 3 same-direction signals before forcing
         a pause, preventing the system from doubling into a losing trade.
      4. Market-hours guard — stops generating signals after EOD.
      5. init_llm() called at startup — logs GPU config to app_llm.log.
    """
    from sqlalchemy import select
    from .execution.paper_executor import PaperExecutor
    from .strategy.engine import StrategyEngine
    from .strategy.adaption import Adaptor
    from .models.schema import TradeMetric
    from .llm.news_feeds import NewsFeedManager
    from .llm.news_analyzer import NewsFlowAnalyzer
    from .llm import init_llm, check_gpu_utilisation
    import pytz

    ET_ZONE = pytz.timezone("America/New_York")

    def now_et():
        return datetime.now(ET_ZONE)

    def is_market_hours() -> bool:
        t = now_et().time()
        import datetime as _dt
        return _dt.time(9, 30) <= t <= _dt.time(16, 0)

    logger.info("=== Forward Trading Mode ===")

    # ── LLM / GPU health check ────────────────────────────────────────
    # Logs Ollama model availability, OLLAMA_NUM_GPU config, and GPU
    # utilisation to app_llm.log. Warns if Ollama is not running or
    # if the GPU layer split looks wrong.
    llm_ok = init_llm()
    if not llm_ok:
        logger.warning(
            "Ollama unavailable or model missing — sentiment will be neutral. "
            "Ensure Ollama is running via start_trading.bat."
        )

    # ── Step 1: Load historical data for training ─────────────────────
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

    # Determine the feature columns the model was trained on
    # (needed to align live feature rows)
    feat_sample  = make_features(df)
    TRAINED_COLS = [
        c for c in feat_sample.columns
        if c not in ("timestamp", "open", "high", "low", "close", "volume")
    ]
    logger.info("Model trained on %d feature columns.", len(TRAINED_COLS))

    adaptor = Adaptor()
    strat   = StrategyEngine(model, adaptor, cfg)

    # ── Step 3: Execution client ──────────────────────────────────────
    use_alpaca  = getattr(args, "alpaca", False)
    use_live    = getattr(args, "live",   False)
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

    # ── Step 4: Live bar buffer (replaces simulated feature cycling) ──
    bar_buffer = LiveBarBuffer()

    # Seed the buffer with recent bars from Alpaca (up to MAX_BARS)
    if live_client and hasattr(live_client, "get_latest_bar"):
        logger.info("Seeding live bar buffer...")
        seed_bar = live_client.get_latest_bar()
        if seed_bar:
            for _ in range(LiveBarBuffer.MIN_BARS):
                bar_buffer.push(seed_bar)
            logger.info("Bar buffer seeded with latest live bar (x%d).",
                        LiveBarBuffer.MIN_BARS)

    # ── Step 5: Sentiment ─────────────────────────────────────────────
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
                logger.info("No headlines — keeping sentiment at %s.", sentiment_label)
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

    # ── Step 6: Main loop ─────────────────────────────────────────────
    trades              = []
    trade_count         = 0
    bar_count           = 0
    last_retrain        = time.time()
    RETRAIN_SECS        = 3600
    BAR_SLEEP           = 60 if live_client else 0.1

    # Consecutive-signal cap: max same-direction signals in a row
    MAX_CONSECUTIVE     = 3
    consecutive_side    = None
    consecutive_count   = 0

    logger.info(
        "Live loop starting | bar_sleep=%ds | trend_filter=ON | "
        "consecutive_cap=%d | market_hours_guard=ON",
        BAR_SLEEP, MAX_CONSECUTIVE,
    )

    while True:
        bar_count += 1

        # ── Market hours guard ────────────────────────────────────────
        if live_client and not is_market_hours():
            logger.info("Outside market hours — pausing (bar %d).", bar_count)
            time.sleep(60)
            continue

        # ── Fetch latest live bar ─────────────────────────────────────
        current_bar = None
        if live_client:
            try:
                current_bar = live_client.get_latest_bar()
            except Exception as e:
                logger.warning("Bar fetch failed: %s", e)

        if current_bar is None:
            logger.debug("No bar available at bar %d — sleeping.", bar_count)
            time.sleep(BAR_SLEEP)
            continue

        # ── Update bar buffer ─────────────────────────────────────────
        bar_buffer.push(current_bar)

        # ── Sentiment refresh ─────────────────────────────────────────
        if bar_count % SENTIMENT_REFRESH_BARS == 0:
            refresh_sentiment()

        # ── Build live feature row ────────────────────────────────────
        if not bar_buffer.ready():
            logger.info(
                "Bar buffer warming up (%d/%d bars) — HOLD.",
                len(bar_buffer._bars), LiveBarBuffer.MIN_BARS,
            )
            time.sleep(BAR_SLEEP)
            continue

        X_live = bar_buffer.get_feature_row(TRAINED_COLS)
        if X_live is None:
            time.sleep(BAR_SLEEP)
            continue

        # ── Trend filter ──────────────────────────────────────────────
        trend = bar_buffer.get_trend_signal()   # +1 up, -1 down, 0 neutral

        # ── Model signal ──────────────────────────────────────────────
        signal = strat.on_bar(X_live)

        if isinstance(signal, dict):
            side = signal.get("side", "HOLD")

            # -- Trend filter: block counter-trend entries -------------
            if side == "BUY" and trend == -1:
                logger.info(
                    "Bar %d: BUY blocked by downtrend filter (trend=%d)",
                    bar_count, trend,
                )
                signal["side"] = "HOLD"
                signal["trend_override"] = True
                side = "HOLD"

            elif side == "SELL" and trend == 1:
                logger.info(
                    "Bar %d: SELL blocked by uptrend filter (trend=%d)",
                    bar_count, trend,
                )
                signal["side"] = "HOLD"
                signal["trend_override"] = True
                side = "HOLD"

            # -- Consecutive-signal cap --------------------------------
            if side != "HOLD":
                if side == consecutive_side:
                    consecutive_count += 1
                else:
                    consecutive_side  = side
                    consecutive_count = 1

                if consecutive_count > MAX_CONSECUTIVE:
                    logger.info(
                        "Bar %d: %s blocked — %d consecutive same-direction "
                        "signals (cap=%d).",
                        bar_count, side, consecutive_count, MAX_CONSECUTIVE,
                    )
                    signal["side"] = "HOLD"
                    signal["consecutive_override"] = True
                    side = "HOLD"
            else:
                consecutive_side  = None
                consecutive_count = 0

            # -- Sentiment filter -------------------------------------
            if side == "BUY" and sentiment_score < -0.15:
                signal["side"] = "HOLD"
                signal["sentiment_override"] = True
                logger.debug(
                    "Bar %d: BUY -> HOLD (bearish sentiment %.2f)",
                    bar_count, sentiment_score,
                )
                side = "HOLD"
            elif side == "SELL" and sentiment_score > 0.15:
                signal["side"] = "HOLD"
                signal["sentiment_override"] = True
                logger.debug(
                    "Bar %d: SELL -> HOLD (bullish sentiment %.2f)",
                    bar_count, sentiment_score,
                )
                side = "HOLD"

        # ── Execute ───────────────────────────────────────────────────
        if live_client:
            live_client.execute_mes_signal(signal, current_bar)

        exe.place_order(pd.Series(current_bar), signal)
        trade_count += 1

        # ── Log trade ─────────────────────────────────────────────────
        trade_entry = {
            "timestamp":       current_bar["timestamp"],
            "symbol":          args.symbol,
            "side":            signal.get("side") if isinstance(signal, dict) else signal,
            "confidence":      signal.get("strength") if isinstance(signal, dict) else None,
            "sentiment":       sentiment_label,
            "sentiment_score": sentiment_score,
            "trend":           trend,
            "pnl":             0.0,
            "status":          "FILLED",
        }
        trades.append(trade_entry)

        try:
            raw_ts = current_bar["timestamp"]
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
                    side      = {
                        "side":       trade_entry["side"],
                        "confidence": trade_entry["confidence"],
                        "trend":      trend,
                    },
                    pnl    = 0.0,
                    status = "FILLED",
                ))
                s.commit()
        except Exception as e:
            logger.error("Trade DB log failed: %s", e)

        # ── Periodic retrain ──────────────────────────────────────────
        now = time.time()
        if (now - last_retrain) > RETRAIN_SECS:
            logger.info("Intraday retrain triggered...")
            try:
                from .training_pipeline import quick_retrain
                quick_retrain()
                loaded = trainer.load()
                if loaded is not None:
                    strat.model = loaded
                    logger.info("Intraday retrain complete — model updated.")
            except Exception as e:
                logger.warning("Intraday retrain failed: %s", e)
            last_retrain = now
            trade_count  = 0

        time.sleep(BAR_SLEEP)

    # ── Save results (reached only if loop exits non-interactively) ───
    out_dir = os.path.join(os.getcwd(), "data")
    os.makedirs(out_dir, exist_ok=True)

    trades_path = os.path.join(out_dir, f"forward_{args.symbol}.csv")
    pd.DataFrame(trades).to_csv(trades_path, index=False)
    logger.info("Saved %d trade records to %s", len(trades), trades_path)

    positions_path = os.path.join(out_dir, f"forward_results_{args.symbol}.csv")
    pd.DataFrame(exe.positions).to_csv(positions_path, index=False)
    logger.info("Saved position log to %s", positions_path)


# ── Live mode (Tradovate direct) ──────────────────────────────────────────────

def live_mode(args):
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
