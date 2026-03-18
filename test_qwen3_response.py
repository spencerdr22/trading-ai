"""
Quick test to see what Qwen3 actually returns.
"""

import ollama

# Test with a simple headline
headline = "Fed signals rate cuts coming in Q2"

prompt = f"""You are a quantitative trading analyst. Analyze this financial news headline.

Headline: {headline}
Symbol Context: SPY
Timestamp: 2026-03-15T20:00:00

You MUST respond with ONLY a valid JSON object. Do not include any markdown, explanations, or extra text. Just the JSON object:
{{
  "sentiment": "bullish|bearish|neutral",
  "confidence": 0.0-1.0,
  "relevance": 0.0-1.0,
  "affected_symbols": ["SPY", "QQQ"],
  "urgency": "low|medium|high",
  "category": "earnings|macro|geopolitical|technical|other",
  "key_entities": ["Fed", "NVIDIA"],
  "summary": "one concise sentence"
}}

Rules:
- confidence: how certain you are about the sentiment
- relevance: how much this affects SPY specifically
- urgency: timeframe of expected price impact
- Be conservative: default to neutral if unclear"""

print("Sending prompt to Qwen3...")
print("="*60)

response = ollama.chat(
    model="qwen3:30b-a3b-q4_K_M",
    messages=[{"role": "user", "content": prompt}],
    options={"temperature": 0.3, "num_predict": 256}
)

content = response["message"]["content"]

print("\nRAW RESPONSE FROM QWEN3:")
print("="*60)
print(content)
print("="*60)
print(f"\nResponse length: {len(content)} characters")
print(f"Starts with: '{content[:50]}'")
print(f"Ends with: '{content[-50:]}'")
