"""
Test Qwen3 with progressively simpler prompts.
"""

import ollama

print("="*60)
print("TESTING QWEN3 WITH DIFFERENT PROMPTS")
print("="*60)

# Test 1: Ultra simple
print("\n[TEST 1] Ultra simple prompt...")
response1 = ollama.chat(
    model="qwen3:30b-a3b-q4_K_M",
    messages=[{"role": "user", "content": "Hello"}]
)
print(f"Response: '{response1['message']['content']}'")
print(f"Length: {len(response1['message']['content'])}")

# Test 2: Simple sentiment
print("\n[TEST 2] Simple sentiment prompt...")
response2 = ollama.chat(
    model="qwen3:30b-a3b-q4_K_M",
    messages=[{"role": "user", "content": "Is this headline bullish or bearish: Fed cuts rates"}]
)
print(f"Response: '{response2['message']['content']}'")
print(f"Length: {len(response2['message']['content'])}")

# Test 3: JSON request (simple)
print("\n[TEST 3] Simple JSON request...")
response3 = ollama.chat(
    model="qwen3:30b-a3b-q4_K_M",
    messages=[{"role": "user", "content": "Respond with just this JSON: {\"test\": \"hello\"}"}]
)
print(f"Response: '{response3['message']['content']}'")
print(f"Length: {len(response3['message']['content'])}")

print("\n" + "="*60)
print("TESTING COMPLETE")
print("="*60)
