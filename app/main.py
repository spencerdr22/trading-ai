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

# Run migration on every startup so schema additions never break a running session
from sqlalchemy import text, inspect as _inspect
try:
    _missing = {"trade_metrics": [("reward", "FLOAT"), ("symbol", "VARCHAR")]}
    with engine.connect() as _conn:
        _insp = _inspect(engine)
        for _tbl, _cols in _missing.items():
            try:
                _existing = {c["name"] for c in _insp.get_columns(_tbl)}
                for _cn, _ct in _cols:
                    if _cn not in _existing:
                        _conn.execute(text(f"ALTER TABLE {_tbl} ADD COLUMN {_cn} {_ct}"))
                        _conn.commit()
            except Exception:
                pass
except Exception:
    pass  # table may not exist yet; create_all in init_db() handles it


# ── DB init ───────────────────────────────────────────────────────────────────

def init_db():
    Base.metadata.create_all(bind=engine)
    # ── Migrate existing SQLite tables that predate schema additions ──
    # SQLite does not support ALTER TABLE DROP COLUMN, but does support
    # ADD COLUMN. We probe and add any missing columns safely.
    _migrate_db(engine)
    logger.info("DB initialized with ORM tables.")


def _migrate_db(eng):
    """Add any columns missing from pre-existing tables without data loss."""
    from sqlalchemy import text, inspect
    missing_cols = {
        "trade_metrics": [
            ("reward",  "FLOAT"),
            ("symbol",  "VARCHAR"),
        ],
    }
    with eng.connect() as conn:
        inspector = inspect(eng)
        for table, cols in missing_cols.items():
            try:
                existing = {c["name"] for c in inspector.get_columns(table)}
            except Exception:
                continue  # table doesn't exist yet—create_all will handle it
            for col_name, col_type in cols:
                if col_name not in existing:
                    try:
                        conn.execute(text(
                            f"ALTER TABLE {table} ADD COLUMN {col_name} {col_type}"
                        ))
                        conn.commit()
                        logger.info("Migrated: added %s.%s", table, col_name)
                    except Exception as e:
                        logger.warning("Migration skipped %s.%s: %s",
                                       table, col_name, e)


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
    fresh on each new bar.

    EMA Anchoring Fix
    -----------------
    The previous version seeded with MIN_BARS=60 bars, but make_features()
    uses a 50-period EMA (ema_cross_slow). With only 60 seed bars the
    50-EMA's first ~50 values were anchored to the seed price, making
    ema_cross_slow near-zero for the first ~50 live bars. This caused the
    trend filter to return 0 (neutral) and SELL signals to be passed through
    incorrectly on days where there was a genuine uptrend.

    Fix applied here:
      - MIN_BARS raised from 60 -> 120 so the 50-EMA has at least 70 bars
        of real price history before the first signal is produced.
      - EMA_WARMUP_BARS=120: get_feature_row() refuses to return features
        until this many bars are in the buffer. The live loop logs a
        "warming up" message and holds.
      - seed_from_alpaca() requests the last 120 rows from real_SPY.csv
        (instead of MIN_BARS=60) so on a normal startup the buffer is
        already warm and trading starts immediately.
      - get_trend_signal() still uses only TREND_LOOKBACK=20 bars so
        the trend is always based on recent price action, not session-start
        prices.
      - STRONG_SIGNAL_MARGIN bypass still applies at 0.30.
    """

    MIN_BARS             = 120   # raised from 60 — need 70+ bars past the 50-EMA
    MAX_BARS             = 500
    EMA_WARMUP_BARS      = 120   # refuse feature rows until this many bars loaded
    TREND_LOOKBACK       = 20    # bars used for trend computation
    STRONG_SIGNAL_MARGIN = 0.30  # bypass trend filter above this margin

    def __init__(self):
        self._bars = collections.deque(maxlen=self.MAX_BARS)

    def push(self, bar: dict):
        self._bars.append({
            "timestamp": bar["timestamp"],
            "open":      float(bar["open"]),
            "high":      float(bar["high"]),
            "low":       float(bar["low"]),
            "close":     float(bar["close"]),
            "volume":    float(bar["volume"]),
        })

    def ready(self) -> bool:
        """True once EMA_WARMUP_BARS real bars have been accumulated."""
        return len(self._bars) >= self.EMA_WARMUP_BARS

    def seed_from_alpaca(self, live_client) -> int:
        """
        Seed the buffer with real recent bars from the real_SPY.csv cache.

        Requests EMA_WARMUP_BARS rows (120) so that on a normal startup
        trading can begin immediately — the 50-EMA is already warm.

        Falls back to repeating the latest live bar only if the cache
        is unavailable (but that path will leave EMAs anchored, so the
        live loop will still wait for EMA_WARMUP_BARS live bars to arrive
        before producing signals).
        """
        try:
            real_path = os.path.join(os.getcwd(), "data", "real_SPY.csv")
            if os.path.exists(real_path):
                seed_df = pd.read_csv(real_path, parse_dates=["timestamp"])
                # Request EMA_WARMUP_BARS rows so 50-EMA is fully warm
                seed_df = seed_df.tail(self.EMA_WARMUP_BARS)
                for _, row in seed_df.iterrows():
                    self.push({
                        "timestamp": str(row["timestamp"]),
                        "open":      float(row["open"]),
                        "high":      float(row["high"]),
                        "low":       float(row["low"]),
                        "close":     float(row["close"]),
                        "volume":    float(row.get("volume", 1000)),
                    })
                logger.info(
                    "Bar buffer seeded with %d real SPY bars from cache "
                    "(50-EMA warm-up complete).",
                    len(seed_df),
                )
                return len(seed_df)
        except Exception as e:
            logger.warning("Buffer seed from cache failed: %s", e)

        # Fallback: single live bar repeated — EMAs will be anchored.
        # The ready() guard (EMA_WARMUP_BARS=120) ensures we won't produce
        # signals until 120 live bars have pushed the anchored seed out.
        try:
            seed_bar = live_client.get_latest_bar()
            if seed_bar:
                for _ in range(self.MIN_BARS):
                    self.push(seed_bar)
                logger.warning(
                    "Bar buffer seeded with latest live bar x%d (fallback). "
                    "EMAs will be anchored — waiting for %d live bars before "
                    "producing signals.",
                    self.MIN_BARS, self.EMA_WARMUP_BARS,
                )
                return self.MIN_BARS
        except Exception as e:
            logger.warning("Buffer fallback seed failed: %s", e)

        return 0

    def get_feature_row(self, trained_feature_cols: list):
        """
        Return a single-row DataFrame of features for the latest bar,
        or None if the buffer has not yet accumulated EMA_WARMUP_BARS bars.

        The EMA_WARMUP_BARS check is the primary guard against anchored EMAs:
        even if the buffer was seeded with repeated identical prices (fallback
        path), we wait until enough real live bars have pushed those seed
        bars out of the EMA calculation window.
        """
        if not self.ready():
            return None

        df   = pd.DataFrame(list(self._bars))
        feat = make_features(df)
        last = feat.iloc[[-1]].copy()
        for col in trained_feature_cols:
            if col not in last.columns:
                last[col] = 0.0
        return last[trained_feature_cols]

    def get_trend_signal(self) -> int:
        """
        Trend direction based on the most recent TREND_LOOKBACK bars only.

        Using a short recent window prevents stale seed prices from
        anchoring the EMA and misrepresenting the current trend.

        Returns +1 (uptrend), -1 (downtrend), 0 (neutral/choppy).
        """
        if len(self._bars) < self.TREND_LOOKBACK:
            return 0

        # Use only the most recent TREND_LOOKBACK bars
        recent = list(self._bars)[-self.TREND_LOOKBACK:]
        closes = pd.Series([b["close"] for b in recent])

        ema_fast = closes.ewm(span=9,  adjust=False).mean()
        ema_slow = closes.ewm(span=21, adjust=False).mean()

        slope = ema_fast.iloc[-1] - ema_fast.iloc[-3]   # 3-bar momentum
        cross = ema_fast.iloc[-1] - ema_slow.iloc[-1]   # fast vs slow

        if slope > 0 and cross > 0:
            return 1
        elif slope < 0 and cross < 0:
            return -1
        return 0


# ── Forward / Paper trading ───────────────────────────────────────────────────

def forward_mode(args):
    """
    Continuous live trading loop.

    EMA fix summary (see LiveBarBuffer docstring for full details):
      - Buffer seeded with 120 real SPY bars from cache (was 60).
      - get_feature_row() blocks until 120 bars loaded (EMA_WARMUP_BARS).
      - Trend filter still uses only 20-bar window.
      - Sentiment threshold 0.25, strong signal bypass at 0.30.
    """
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

    exe = PaperExecutor(cfg)

    # ── Step 4: Live bar buffer ───────────────────────────────────────
    bar_buffer = LiveBarBuffer()

    if live_client:
        seeded = bar_buffer.seed_from_alpaca(live_client)
        if seeded == 0:
            logger.warning(
                "Bar buffer could not be seeded from cache — will warm up live. "
                "Signals will be held until %d bars accumulated.",
                LiveBarBuffer.EMA_WARMUP_BARS,
            )
        elif seeded < LiveBarBuffer.EMA_WARMUP_BARS:
            logger.warning(
                "Bar buffer only seeded with %d bars (need %d for full EMA warm-up). "
                "Will top up from live bars.",
                seeded, LiveBarBuffer.EMA_WARMUP_BARS,
            )

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
            logger.info("Fetched %d headlines — running sentiment...", len(headlines))
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

    # Signal caps — strong signals get a higher cap before being blocked
    MAX_CONSECUTIVE_WEAK   = 3
    MAX_CONSECUTIVE_STRONG = 5
    consecutive_side       = None
    consecutive_count      = 0

    # Sentiment filter threshold
    SENTIMENT_BLOCK_THRESHOLD = 0.25

    logger.info(
        "Live loop starting | bar_sleep=%ds | trend_filter=ON | "
        "ema_warmup=%d | weak_cap=%d | strong_cap=%d | sentiment_threshold=%.2f",
        BAR_SLEEP, LiveBarBuffer.EMA_WARMUP_BARS,
        MAX_CONSECUTIVE_WEAK, MAX_CONSECUTIVE_STRONG,
        SENTIMENT_BLOCK_THRESHOLD,
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

        # ── EMA warm-up guard — hold until buffer is ready ────────────
        if not bar_buffer.ready():
            logger.info(
                "EMA warm-up: %d/%d bars — HOLD (need %d for reliable 50-EMA).",
                len(bar_buffer._bars),
                LiveBarBuffer.EMA_WARMUP_BARS,
                LiveBarBuffer.EMA_WARMUP_BARS,
            )
            time.sleep(BAR_SLEEP)
            continue

        # ── Build live feature row ────────────────────────────────────
        X_live = bar_buffer.get_feature_row(TRAINED_COLS)
        if X_live is None:
            time.sleep(BAR_SLEEP)
            continue

        # ── Trend signal (short lookback — not affected by seed) ──────
        trend = bar_buffer.get_trend_signal()

        # ── Model signal ──────────────────────────────────────────────
        signal   = strat.on_bar(X_live)
        decision = strat.decide(X_live.iloc[0].to_numpy())
        margin   = abs(
            decision["final"].get("buy", 0.5) - decision["final"].get("sell", 0.5)
        )
        is_strong = margin >= LiveBarBuffer.STRONG_SIGNAL_MARGIN

        if isinstance(signal, dict):
            side = signal.get("side", "HOLD")

            # -- Trend filter ------------------------------------------
            if side == "BUY" and trend == -1 and not is_strong:
                logger.info(
                    "Bar %d: BUY blocked by downtrend filter "
                    "(trend=%d, margin=%.3f < strong threshold %.2f)",
                    bar_count, trend, margin, LiveBarBuffer.STRONG_SIGNAL_MARGIN,
                )
                signal["side"] = "HOLD"
                signal["trend_override"] = True
                side = "HOLD"

            elif side == "SELL" and trend == 1 and not is_strong:
                logger.info(
                    "Bar %d: SELL blocked by uptrend filter "
                    "(trend=%d, margin=%.3f < strong threshold %.2f)",
                    bar_count, trend, margin, LiveBarBuffer.STRONG_SIGNAL_MARGIN,
                )
                signal["side"] = "HOLD"
                signal["trend_override"] = True
                side = "HOLD"

            elif side != "HOLD" and is_strong and trend != 0:
                direction_agrees = (side == "BUY" and trend == 1) or \
                                   (side == "SELL" and trend == -1)
                if not direction_agrees:
                    logger.info(
                        "Bar %d: %s allowed despite trend=%d — strong signal "
                        "(margin=%.3f >= %.2f)",
                        bar_count, side, trend, margin,
                        LiveBarBuffer.STRONG_SIGNAL_MARGIN,
                    )

            # -- Consecutive-signal cap --------------------------------
            if side != "HOLD":
                if side == consecutive_side:
                    consecutive_count += 1
                else:
                    consecutive_side  = side
                    consecutive_count = 1

                cap = MAX_CONSECUTIVE_STRONG if is_strong else MAX_CONSECUTIVE_WEAK
                if consecutive_count > cap:
                    logger.info(
                        "Bar %d: %s blocked — %d consecutive signals "
                        "(cap=%d, strong=%s).",
                        bar_count, side, consecutive_count, cap, is_strong,
                    )
                    signal["side"] = "HOLD"
                    signal["consecutive_override"] = True
                    side = "HOLD"
            else:
                consecutive_side  = None
                consecutive_count = 0

            # -- Sentiment filter --------------------------------------
            # Strong signals (margin >= STRONG_SIGNAL_MARGIN) bypass
            # sentiment — the model is too confident to be overridden
            # by a news score. Sentiment only blocks weak signals.
            if not is_strong:
                if side == "BUY" and sentiment_score < -SENTIMENT_BLOCK_THRESHOLD:
                    signal["side"] = "HOLD"
                    signal["sentiment_override"] = True
                    logger.info(
                        "Bar %d: BUY -> HOLD (bearish sentiment %.3f < -%.2f)",
                        bar_count, sentiment_score, SENTIMENT_BLOCK_THRESHOLD,
                    )
                    side = "HOLD"
                elif side == "SELL" and sentiment_score > SENTIMENT_BLOCK_THRESHOLD:
                    signal["side"] = "HOLD"
                    signal["sentiment_override"] = True
                    logger.info(
                        "Bar %d: SELL -> HOLD (bullish sentiment %.3f > %.2f)",
                        bar_count, sentiment_score, SENTIMENT_BLOCK_THRESHOLD,
                    )
                    side = "HOLD"
            else:
                if side != "HOLD":
                    logger.info(
                        "Bar %d: %s keeping strong signal — sentiment overridden "
                        "(margin=%.3f >= %.2f, sentiment=%.3f)",
                        bar_count, side, margin,
                        LiveBarBuffer.STRONG_SIGNAL_MARGIN, sentiment_score,
                    )

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
            "margin":          round(margin, 4),
            "strong":          is_strong,
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
                        "side":     trade_entry["side"],
                        "margin":   trade_entry["margin"],
                        "strong":   is_strong,
                        "trend":    trend,
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

    # ── Save results ──────────────────────────────────────────────────
    out_dir = os.path.join(os.getcwd(), "data")
    os.makedirs(out_dir, exist_ok=True)

    pd.DataFrame(trades).to_csv(
        os.path.join(out_dir, f"forward_{args.symbol}.csv"), index=False
    )
    logger.info("Saved %d trade records.", len(trades))

    pd.DataFrame(exe.positions).to_csv(
        os.path.join(out_dir, f"forward_results_{args.symbol}.csv"), index=False
    )


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
