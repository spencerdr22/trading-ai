"""
test_news_feeds.py — CLI test for news API connections and sentiment pipeline.

Usage:
    python -m app.llm.test_news_feeds api
    python -m app.llm.test_news_feeds full
"""

import asyncio
import sys

from .news_feeds import NewsFeedManager
from ..monitor.logger import get_logger

logger = get_logger(__name__)


async def test_api_connections():
    print("=" * 60)
    print("Testing News API Connections")
    print("=" * 60)

    manager = NewsFeedManager(symbols=["MES"])

    print("\nAPI Status:")
    print(f"  Alpaca:  {'[OK] Enabled' if manager.alpaca_enabled else '[--] Disabled'}")
    print(f"  Finnhub: {'[OK] Enabled' if manager.finnhub_enabled else '[--] Disabled'}")
    print(f"  NewsAPI: {'[OK] Enabled' if manager.newsapi_enabled else '[--] Disabled'}")
    print()

    if not any([manager.alpaca_enabled, manager.finnhub_enabled, manager.newsapi_enabled]):
        print("[FAIL] No APIs configured.  Add keys to .env:")
        print("  ALPACA_API_KEY / ALPACA_SECRET_KEY")
        print("  FINNHUB_API_KEY")
        print("  NEWSAPI_KEY")
        return

    print("Fetching headlines from all sources...")
    headlines = await manager.fetch_all_sources()

    if not headlines:
        print("[FAIL] No headlines returned.  Check API keys and connectivity.")
        return

    print(f"\n[OK] Fetched {len(headlines)} unique headlines\n")
    print("=" * 60)
    print("Sample Headlines by Source")
    print("=" * 60)

    from collections import defaultdict
    by_source = defaultdict(list)
    for a in headlines:
        by_source[a["source"]].append(a)

    for source, articles in by_source.items():
        print(f"\n[{source.upper()}]  ({len(articles)} total)")
        print("-" * 60)
        for i, art in enumerate(articles[:5], 1):
            print(f"  {i}. {art['headline'][:72]}")
            print(f"     {art['timestamp'][:19]}")

    print("\n[OK] API connection test complete.")


async def test_with_sentiment_analysis():
    print("=" * 60)
    print("Full Pipeline Test: News + Qwen3 Sentiment")
    print("=" * 60)

    try:
        from .news_analyzer import NewsFlowAnalyzer
    except Exception as e:
        print(f"[WARN] Sentiment analyzer unavailable: {e}")
        print("Running news-only test instead...")
        await test_api_connections()
        return

    manager  = NewsFeedManager(symbols=["MES"])
    analyzer = NewsFlowAnalyzer()

    print("\nFetching headlines...")
    headlines = await manager.fetch_all_sources()
    if not headlines:
        print("[FAIL] No headlines fetched.")
        return

    print(f"[OK] Fetched {len(headlines)} headlines")
    print("\nAnalyzing sentiment (may take 30-60 seconds)...")

    sample_texts = [h["headline"] for h in headlines[:10]]
    sentiment_df = analyzer.analyze_batch(sample_texts, symbol="SPY")

    if sentiment_df.empty:
        print("[FAIL] Sentiment analysis returned no results.")
        return

    print("\n" + "=" * 60)
    print("Sentiment Results (sample)")
    print("=" * 60)

    for i, row in sentiment_df.iterrows():
        print(f"\n  [{i+1}] {headlines[i]['headline'][:60]}")
        print(f"       Sentiment : {row['sentiment'].upper()} "
              f"(conf={row['confidence']:.2f}, rel={row['relevance']:.2f})")
        print(f"       Summary   : {row.get('summary', '')[:80]}")

    agg = analyzer.get_aggregated_sentiment(sentiment_df)
    print("\n" + "=" * 60)
    print("Aggregated Market Sentiment")
    print("=" * 60)
    print(f"  Overall : {agg['overall_sentiment'].upper()}")
    print(f"  Score   : {agg['sentiment_score']:.3f}  (-1=bearish .. +1=bullish)")
    print(f"  Bull    : {agg['bullish_pct']:.1%}")
    print(f"  Bear    : {agg['bearish_pct']:.1%}")
    print(f"  Neutral : {agg['neutral_pct']:.1%}")
    print("\n[OK] Full pipeline test complete.")


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ("api", "full"):
        print("Usage: python -m app.llm.test_news_feeds [api|full]")
        print("  api  — test API connections only")
        print("  full — test news fetch + Qwen3 sentiment")
        sys.exit(1)

    mode = sys.argv[1]
    if mode == "api":
        asyncio.run(test_api_connections())
    else:
        asyncio.run(test_with_sentiment_analysis())


if __name__ == "__main__":
    main()
