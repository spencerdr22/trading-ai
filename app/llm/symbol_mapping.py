"""
Symbol mapping for futures to get better news relevance.
"""

# Futures symbols don't map directly to Alpaca tickers
# Use related ETFs and index symbols for better news coverage
FUTURES_TO_NEWS_SYMBOLS = {
    "MES": ["SPY", "SPX", "^GSPC"],  # Micro E-mini S&P 500 → same underlying as ES
    "ES":  ["SPY", "SPX", "^GSPC"],  # E-mini S&P 500 futures → SPY ETF + index
    "MNQ": ["QQQ", "NDX", "^IXIC"],  # Micro E-mini Nasdaq
    "NQ":  ["QQQ", "NDX", "^IXIC"],  # Nasdaq futures → QQQ ETF + index
    "MYM": ["DIA", "DJI", "^DJI"],   # Micro E-mini Dow
    "YM":  ["DIA", "DJI", "^DJI"],   # Dow futures → DIA ETF + index
    "M2K": ["IWM", "RUT", "^RUT"],   # Micro E-mini Russell 2000
    "RTY": ["IWM", "RUT", "^RUT"]    # Russell 2000 → IWM ETF + index
}

def get_news_symbols(trading_symbol: str) -> list:
    """
    Convert trading symbol to best news symbols.
    
    Args:
        trading_symbol: Your trading symbol (ES, NQ, etc.)
    
    Returns:
        List of symbols to use for news queries
    """
    return FUTURES_TO_NEWS_SYMBOLS.get(trading_symbol, [trading_symbol])
