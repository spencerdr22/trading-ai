"""
LLM prompt templates for financial analysis.
"""

SENTIMENT_ANALYSIS_PROMPT = """Analyze this headline for trading: {headline}

Respond with ONLY this JSON format:
{{
  "sentiment": "bullish",
  "confidence": 0.8,
  "relevance": 0.9,
  "urgency": "high",
  "summary": "Brief explanation"
}}

sentiment: bullish/bearish/neutral
confidence: 0.0 to 1.0
relevance: 0.0 to 1.0 (how much it affects {symbol})
urgency: low/medium/high"""


REGIME_DETECTION_PROMPT = """You are a market regime analyst. Based on recent news sentiment data:

Bullish Headlines: {bullish_count}
Bearish Headlines: {bearish_count}
Neutral Headlines: {neutral_count}

High Urgency Events: {high_urgency_count}
Key Categories: {top_categories}

Recent Major Headlines:
{headline_list}

Determine the current market regime. Respond with ONLY valid JSON:
{{
  "regime": "risk_on|risk_off|consolidation|crisis|transition",
  "confidence": 0.0-1.0,
  "volatility_expectation": "low|medium|high|extreme",
  "recommended_position_sizing": 0.0-1.0,
  "reasoning": "2-3 sentence explanation"
}}

Definitions:
- risk_on: bullish sentiment, low urgency, growth focus
- risk_off: bearish sentiment, high urgency, defensive positioning
- consolidation: neutral sentiment, low urgency, range-bound
- crisis: extreme bearish, high urgency, market dislocations
- transition: mixed signals, shifting dynamics"""


TRADE_EXPLANATION_PROMPT = """Explain why this trade was executed:

Trade Details:
- Side: {side}
- Entry: ${entry_price}
- Size: {size} contracts
- Timestamp: {timestamp}

Model Signals:
- RF Probability (buy): {rf_prob:.2%}
- LSTM Probability (buy): {lstm_prob:.2%}
- RL Policy Action: {rl_action}

Market Context:
- RSI: {rsi}
- EMA Signal: {ema_signal}
- ATR: {atr}
- Recent Sentiment: {sentiment}

Write a 2-sentence technical explanation suitable for an audit log."""
