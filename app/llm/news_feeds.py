"""
Real-time news feed integration with Alpaca, Finnhub, and NewsAPI.
Optimized for Ryzen 7800X3D with high-bandwidth DDR5.
"""

import os
import asyncio
import aiohttp
import json
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor
from collections import deque

from ..monitor.logger import get_logger
from .system_config import system_config, get_system_config
from .symbol_mapping import get_news_symbols
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = get_logger(__name__)

# ============================================================
# API CREDENTIALS (from .env)
# ============================================================

ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")

# API Endpoints
ALPACA_NEWS_URL = "https://data.alpaca.markets/v1beta1/news"
FINNHUB_NEWS_URL = "https://finnhub.io/api/v1/news"
NEWSAPI_URL = "https://newsapi.org/v2/everything"

# Rate limits (requests per minute)
ALPACA_RATE_LIMIT = 200
FINNHUB_RATE_LIMIT = 60
NEWSAPI_RATE_LIMIT = 100


# ============================================================
# NEWS FEED MANAGER
# ============================================================

class NewsFeedManager:
    """
    Manages multiple news API sources with intelligent rate limiting.
    Optimized for high-throughput async I/O on Ryzen 7800X3D.
    """
    
    def __init__(
        self,
        symbols: List[str] = None,
        cache_ttl_minutes: int = 15
    ):
        self.trading_symbols = symbols or ["ES", "NQ", "YM", "RTY"]
        # Convert to news-friendly symbols (SPY for ES, QQQ for NQ, etc.)
        self.symbols = []
        for sym in self.trading_symbols:
            self.symbols.extend(get_news_symbols(sym))
        self.cache_ttl = timedelta(minutes=cache_ttl_minutes)
        
        # Use adaptive configuration
        self.config = get_system_config()
        
        # Dynamic worker allocation
        limits = self.config.get_resource_limits()
        self.max_workers = limits.news_api_workers
        
        # High-speed in-memory headline buffer (leverages 7800X3D cache)
        self.headline_buffer = deque(maxlen=limits.max_headlines_buffer)
        self.analysis_queue = deque(maxlen=10000)
        
        # API availability check
        self.alpaca_enabled = bool(ALPACA_API_KEY and ALPACA_SECRET_KEY)
        self.finnhub_enabled = bool(FINNHUB_API_KEY)
        self.newsapi_enabled = bool(NEWSAPI_KEY)
        
        if not any([self.alpaca_enabled, self.finnhub_enabled, self.newsapi_enabled]):
            logger.warning("⚠️ No news API keys configured. Set them in .env file.")
        
        logger.info(
            f"NewsFeedManager initialized: "
            f"Alpaca={self.alpaca_enabled}, "
            f"Finnhub={self.finnhub_enabled}, "
            f"NewsAPI={self.newsapi_enabled}"
        )
    
    # --------------------------------------------------------
    # ALPACA NEWS API
    # --------------------------------------------------------
    
    async def fetch_alpaca_news(
        self,
        session: aiohttp.ClientSession,
        symbols: List[str] = None,
        limit: int = 50
    ) -> List[Dict]:
        """
        Fetch news from Alpaca (best for US equities/futures).
        Rate limit: 200 req/min
        """
        if not self.alpaca_enabled:
            return []
        
        symbols = symbols or self.symbols
        
        headers = {
            "APCA-API-KEY-ID": ALPACA_API_KEY,
            "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY
        }
        
        params = {
            "symbols": ",".join(symbols),
            "limit": limit,
            "sort": "desc"  # Most recent first
        }
        
        try:
            async with session.get(
                ALPACA_NEWS_URL,
                headers=headers,
                params=params,
                timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    articles = data.get("news", [])
                    
                    # Normalize format
                    normalized = []
                    for article in articles:
                        normalized.append({
                            "source": "alpaca",
                            "headline": article.get("headline", ""),
                            "summary": article.get("summary", ""),
                            "url": article.get("url", ""),
                            "timestamp": article.get("created_at", datetime.utcnow().isoformat()),
                            "symbols": article.get("symbols", []),
                            "author": article.get("author", "")
                        })
                    
                    logger.debug(f"✅ Alpaca: fetched {len(normalized)} articles")
                    return normalized
                else:
                    logger.warning(f"Alpaca API error: {response.status}")
                    return []
        
        except asyncio.TimeoutError:
            logger.warning("Alpaca API timeout")
            return []
        except Exception as e:
            logger.error(f"Alpaca fetch error: {e}")
            return []
    
    # --------------------------------------------------------
    # FINNHUB NEWS API
    # --------------------------------------------------------
    
    async def fetch_finnhub_news(
        self,
        session: aiohttp.ClientSession,
        category: str = "general",
        limit: int = 50
    ) -> List[Dict]:
        """
        Fetch news from Finnhub (good for macro/market news).
        Rate limit: 60 req/min (free tier)
        """
        if not self.finnhub_enabled:
            return []
        
        params = {
            "category": category,  # general, forex, crypto, merger
            "token": FINNHUB_API_KEY
        }
        
        try:
            async with session.get(
                FINNHUB_NEWS_URL,
                params=params,
                timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                if response.status == 200:
                    articles = await response.json()
                    
                    # Normalize format
                    normalized = []
                    for article in articles[:limit]:
                        normalized.append({
                            "source": "finnhub",
                            "headline": article.get("headline", ""),
                            "summary": article.get("summary", ""),
                            "url": article.get("url", ""),
                            "timestamp": datetime.fromtimestamp(
                                article.get("datetime", 0)
                            ).isoformat(),
                            "symbols": [],  # Finnhub doesn't provide symbols directly
                            "category": article.get("category", "")
                        })
                    
                    logger.debug(f"✅ Finnhub: fetched {len(normalized)} articles")
                    return normalized
                else:
                    logger.warning(f"Finnhub API error: {response.status}")
                    return []
        
        except asyncio.TimeoutError:
            logger.warning("Finnhub API timeout")
            return []
        except Exception as e:
            logger.error(f"Finnhub fetch error: {e}")
            return []
    
    # --------------------------------------------------------
    # NEWSAPI.ORG
    # --------------------------------------------------------
    
    async def fetch_newsapi(
        self,
        session: aiohttp.ClientSession,
        query: str = "stock market OR futures OR trading",
        limit: int = 50
    ) -> List[Dict]:
        """
        Fetch news from NewsAPI.org (broad coverage).
        Rate limit: 100 req/day (free tier), 1000 req/day (paid)
        """
        if not self.newsapi_enabled:
            return []
        
        params = {
            "q": query,
            "apiKey": NEWSAPI_KEY,
            "sortBy": "publishedAt",
            "pageSize": min(limit, 100),  # Max 100 per request
            "language": "en"
        }
        
        try:
            async with session.get(
                NEWSAPI_URL,
                params=params,
                timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    articles = data.get("articles", [])
                    
                    # Normalize format
                    normalized = []
                    for article in articles:
                        normalized.append({
                            "source": "newsapi",
                            "headline": article.get("title", ""),
                            "summary": article.get("description", ""),
                            "url": article.get("url", ""),
                            "timestamp": article.get("publishedAt", datetime.utcnow().isoformat()),
                            "symbols": [],  # Must infer from content
                            "author": article.get("author", "")
                        })
                    
                    logger.debug(f"✅ NewsAPI: fetched {len(normalized)} articles")
                    return normalized
                else:
                    logger.warning(f"NewsAPI error: {response.status}")
                    return []
        
        except asyncio.TimeoutError:
            logger.warning("NewsAPI timeout")
            return []
        except Exception as e:
            logger.error(f"NewsAPI fetch error: {e}")
            return []
    
    # --------------------------------------------------------
    # UNIFIED FETCH (PARALLEL)
    # --------------------------------------------------------
    
    async def fetch_all_sources(self) -> List[Dict]:
        """
        Fetch from all enabled sources concurrently.
        Optimized for low-latency with async I/O.
        """
        async with aiohttp.ClientSession() as session:
            tasks = []
            
            if self.alpaca_enabled:
                tasks.append(self.fetch_alpaca_news(session))
            
            if self.finnhub_enabled:
                tasks.append(self.fetch_finnhub_news(session))
            
            if self.newsapi_enabled:
                tasks.append(self.fetch_newsapi(session))
            
            if not tasks:
                logger.warning("No news sources enabled - add API keys to .env")
                return []
            
            # Execute all API calls in parallel
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Flatten and deduplicate
            all_headlines = []
            seen_headlines = set()
            
            for result in results:
                if isinstance(result, Exception):
                    logger.error(f"Fetch error: {result}")
                    continue
                
                for article in result:
                    headline = article["headline"]
                    if headline and headline not in seen_headlines:
                        seen_headlines.add(headline)
                        all_headlines.append(article)
            
            logger.info(f"📰 Fetched {len(all_headlines)} unique headlines from {len(tasks)} sources")
            return all_headlines
    
    # --------------------------------------------------------
    # SYNCHRONOUS WRAPPER
    # --------------------------------------------------------
    
    def get_recent_headlines(
        self,
        symbols: List[str] = None,
        hours: int = 24
    ) -> pd.DataFrame:
        """
        Synchronous method to fetch recent headlines.
        Use this for backtesting or one-off queries.
        """
        # Run async fetch in sync context
        headlines = asyncio.run(self.fetch_all_sources())
        
        if not headlines:
            return pd.DataFrame()
        
        # Filter by time window — use timezone-aware cutoff to match parsed timestamps
        from datetime import timezone
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        filtered = []
        for h in headlines:
            try:
                ts = datetime.fromisoformat(h["timestamp"].replace("Z", "+00:00"))
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                if ts > cutoff:
                    filtered.append(h)
            except Exception:
                filtered.append(h)  # include if timestamp unparseable
        
        return pd.DataFrame(filtered)
    
    # --------------------------------------------------------
    # MEMORY-EFFICIENT BUFFER ACCESS
    # --------------------------------------------------------
    
    def get_buffer_snapshot(self) -> List[Dict]:
        """
        Get current headline buffer (up to 10,000 most recent).
        Leverages 7800X3D's massive L3 cache for instant access.
        """
        return list(self.headline_buffer)
    
    def clear_buffer(self):
        """Clear headline buffer."""
        self.headline_buffer.clear()
        logger.info("Headline buffer cleared")
