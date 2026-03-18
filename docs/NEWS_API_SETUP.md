# 🌐 News API Setup Guide

## Step-by-Step Instructions

### ✅ Step 1: Get Your API Keys (15 minutes)

#### **Alpaca** (Best for US futures/equities)
1. Visit: https://alpaca.markets/
2. Click **"Get Started"** → **"Sign Up"**
3. Choose **"Paper Trading"** (100% free)
4. After signup, go to dashboard: https://app.alpaca.markets/paper/dashboard/overview
5. Click **"Generate API Keys"** or **"View"** if already generated
6. Copy both:
   - `API Key ID` (looks like: `PKXXXXXXXXXXXXXXXX`)
   - `Secret Key` (looks like: `xxxxxxxxxxxxxxxxxxxxxxxxx`)

#### **Finnhub** (Good for macro news)
1. Visit: https://finnhub.io/
2. Click **"Get free API key"**
3. Sign up with email
4. Dashboard shows your key immediately
5. Copy the API key (looks like: `csh4k2hr01abcdefg`)

#### **NewsAPI** (Backup source)
1. Visit: https://newsapi.org/
2. Click **"Get API Key"**
3. Fill form (use "Personal" or "Student" for free tier)
4. Copy your key (looks like: `a1b2c3d4e5f6g7h8i9j0`)

---

### ✅ Step 2: Add Keys to .env File

Open `C:\Users\spenc\Documents\trading-ai\.env` in a text editor and replace the placeholders:

```bash
# Replace these lines with your actual keys:
ALPACA_API_KEY=YOUR_ALPACA_KEY_HERE
ALPACA_SECRET_KEY=YOUR_ALPACA_SECRET_HERE
FINNHUB_API_KEY=YOUR_FINNHUB_KEY_HERE
NEWSAPI_KEY=YOUR_NEWSAPI_KEY_HERE
```

**Example (with fake keys):**
```bash
ALPACA_API_KEY=PK1234ABCD5678EFGH
ALPACA_SECRET_KEY=abc123def456ghi789jkl012mno345
FINNHUB_API_KEY=csh4k2hr01qv1gg2pdag
NEWSAPI_KEY=a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6
```

**⚠️ Important:** 
- No quotes around the values
- No spaces after `=`
- Keep the `.env` file secret (it's in `.gitignore`)

---

### ✅ Step 3: Test API Connections

```powershell
cd C:\Users\spenc\Documents\trading-ai
.venv\Scripts\activate
python -m app.llm.test_news_feeds api
```

**Expected output:**
```
Testing News API Connections
============================================================

API Status:
  Alpaca:  ✅ Enabled
  Finnhub: ✅ Enabled
  NewsAPI: ✅ Enabled

Fetching headlines from all sources...

✅ Successfully fetched 45 headlines

Sample Headlines:
[ALPACA] - 15 total
1. Fed signals potential rate pause amid inflation concerns...
   Time: 2025-01-20 14:32:15
   Symbols: ES, NQ

[FINNHUB] - 18 total
1. Tech sector leads market gains on earnings optimism...
   Time: 2025-01-20 13:45:22

[NEWSAPI] - 12 total
1. Crude oil prices surge affecting market sentiment...
   Time: 2025-01-20 12:18:09
```

---

### ✅ Step 4: Test Full Pipeline (News + Sentiment)

```powershell
python -m app.llm.test_news_feeds full
```

**This will:**
1. Fetch headlines from all 3 APIs
2. Analyze sentiment with Qwen3 LLM
3. Show aggregated market sentiment

**Expected time:** 30-60 seconds (GPU inference)

---

## 🔧 Troubleshooting

### Problem: "No APIs configured"
**Solution:** Check your `.env` file has the keys with no quotes/spaces

### Problem: "Alpaca API error: 401"
**Possible causes:**
- Wrong API keys
- Keys are for live trading (need paper trading keys)
- Keys expired

**Solution:**
1. Go to Alpaca dashboard
2. Regenerate API keys
3. Update `.env` file

### Problem: "Finnhub API error: 403"
**Cause:** API key invalid or rate limit exceeded

**Solution:**
- Verify key on Finnhub dashboard
- Free tier: 60 requests/minute limit

### Problem: "NewsAPI error: 426"
**Cause:** Using free tier with HTTPS (premium feature)

**Solution:** Already handled in code (uses correct endpoint)

### Problem: "No headlines fetched"
**Check:**
1. Internet connection
2. API keys are correct in `.env`
3. No firewall blocking requests
4. Run: `python -m app.llm.test_news_feeds api` for detailed errors

---

## 📊 Rate Limits (Free Tier)

| API | Limit | Notes |
|-----|-------|-------|
| Alpaca | 200/min | Paper trading = unlimited |
| Finnhub | 60/min | Can upgrade for more |
| NewsAPI | 100/day | 1000/day on paid plan |

**System behavior:**
- Fetches from all APIs in parallel
- Deduplicates headlines
- Respects rate limits automatically

---

## 🎯 Next Steps After Setup

Once all APIs are working:

1. **Add to startup script:**
   ```powershell
   .\startup.bat
   # Choose option 5 to verify news feeds
   ```

2. **Test streaming (coming next):**
   - Real-time news monitoring
   - Automatic sentiment analysis
   - Integration with trading signals

3. **Add sentiment features to ML models:**
   - Sentiment becomes an input feature
   - Improves trading decisions

---

## 💡 Pro Tips

- **Alpaca** is best for ES/NQ/YM futures news
- **Finnhub** is great for macro events (Fed, earnings)
- **NewsAPI** is good as a backup/validation source
- You only need **1 API minimum** to get started
- All 3 together give best coverage

---

## 📝 Quick Reference

```powershell
# Test APIs
python -m app.llm.test_news_feeds api

# Test full pipeline
python -m app.llm.test_news_feeds full

# Check .env file
type .env | findstr "API"
```

---

**Questions?** Run the test scripts above - they'll show detailed error messages if something's wrong!
