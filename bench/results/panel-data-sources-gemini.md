# Data Sources Panel: Gemini 2.5 Pro
*Generated: 2026-04-19*

## Core Thesis
"Alpha doesn't live in public price data. It lives in flows, frictions, and forward-looking information that hasn't been fully priced in yet."

"Stop fighting for scraps in the price chart. Start listening to the whispers in the options market."

## Top 5 Signals (Ranked)

### 1. Options Market Implied Volatility & Dealer Positioning (GEX/VEX)
**"This is the single most important one. Full stop."**
- The options market is the insurance market for the stock market
- Price is what happened; IV is what the market thinks is about to happen
- Dealer positioning (GEX) tells about market fragility and feedback loops
- **Pro:** ORATS, OptionMetrics (>$50k/yr)
- **Prosumer:** Polygon.io ($200/mo — top pick), Intrinio, QuantConnect
- **Free:** CBOE daily summary data
- **SNR:** High, but requires feature engineering (IV term structure, skew, GEX derivation)

### 2. Analyst Earnings Estimate Revisions
- Not ratings ("Buy/Hold") which are worthless — the CHANGE in estimates
- Cascade of upward revisions = powerful leading indicator
- Direct proxy for fundamental momentum before price reflects it
- **Prosumer:** Finnhub.io (~$50/mo)
- **Free:** Difficult to get clean PIT data; Yahoo Finance scraping is messy
- **SNR:** Medium. Filter for reputable analysts; look for consensus shifts

### 3. High-Quality, Low-Latency News & Event Sentiment
- Structured data on SEC filings (8-Ks), M&A, drug trials, etc.
- Quantify "surprise" relative to expectations
- **Pro:** RavenPack, Bloomberg Event-Driven
- **Prosumer:** Alpaca News API (included with trading account)
- **DIY:** SEC EDGAR scraper (free but work required)
- **SNR:** Low to Medium. 99% of news is noise

### 4. Insider Transactions (Filtered)
- Raw feed is almost entirely noise
- "Opportunistic" buys = strong but infrequent
- Cluster buys (multiple insiders buying simultaneously) = gold standard
- **Prosumer:** Finnhub.io (included in $50/mo)
- **Free:** SEC EDGAR
- **SNR:** Low. Must filter aggressively

### 5. Short Interest & Securities Lending Data
- Not just bearish signal — measures institutional conviction + squeeze fuel
- Cost to borrow = real-time demand from shorters (higher signal)
- **Pro:** S3 Partners, Markit
- **Prosumer:** IBKR (free with account), Quandl ($50-100/mo)
- **Free:** FINRA bi-monthly (too lagged for 1-5d)
- **SNR:** Medium

## Le-WM Integration — The Key Insight
Current embeddings capture: "Risk-On (Low Vol, Steep Curve)" vs "Risk-Off (High VIX, Flat Curve)"

With options data, embeddings learn:
- **"Complacent Rally"**: Price rising but put/call skew increasing (smart money buying protection). Fragile state a linear model would miss
- **"Panicked Selling"**: Price falling + short-term IV exploding relative to long-term (backwardation). Powerful bounce predictor
- **"Gamma Squeeze Imminent"**: GEX high/positive + stock approaching major strike. Market fragility state

With analyst revisions, embeddings differentiate:
- **"Junk Rally"** (price up, estimates flat) vs **"Fundamental Rally"** (price up, estimates up)

"Your Le-WM is a powerful engine waiting for high-octane fuel."

## $200/mo Stack (Beats 99% of Retail, 70% of "Pro" Retail)
1. **Polygon.io ($200/mo)** — Real-time equities AND US options data. Raw material for IV term structure, skew, GEX proxies, unusual options activity. **"This one subscription is the 10x improvement."**
2. **Finnhub.io ($50/mo, or start with just Polygon)** — Analyst revisions + insider transactions. Budget hack: scrape SEC EDGAR for free instead

**Total: ~$200/mo**

## Remove From Current Set
- **RSI-14, MACD, Bollinger %B:** "Get rid of them. Now." Path-dependent momentum repackaged noisily. XGBoost finds momentum from returns/vol already
- **Day of Week:** "This alpha decayed in the 1990s"
- **52-week high distance:** Redundant with cross-sectional rank features

## Retail Obsessions with Zero Value
- **Chart patterns** (H&S, cup/handle): "Astrology for men"
- **Fibonacci retracements:** "Numerology"
- **Basic social media sentiment:** Easily manipulated, reflects past
- **Most news:** "99% is narrative-driven entertainment"

## Information Cascade (5%+ Move)

### 1. The Whisper (T-5 to T-2)
- First traces in options: surge in OTM call volume at specific strikes
- IV skew changes (calls getting expensive vs puts)
- **Your entry point**

### 2. The Echo (T-2 to T-1)
- Market makers delta-hedge → buying pressure lifts stock
- Other sophisticated players detect options activity
- Slight volume uptick, price drifts for "no reason"
- **The sweet spot**

### 3. The News (T=0)
- 8-K filed, press release, deal announced
- Price gaps, options repriced
- **Too late** — alpha is gone

### 4. The Public (T+1 to T+5)
- CNBC coverage, retail piles in, analysts upgrade
- "Dumb money" phase — you should already be out

"Your system should be positioned firmly in Stage 1 and 2."

## Point-in-Time Reality
- **Options (Polygon):** Real-time feed; snapshot at 3:55 PM EST; no look-ahead if disciplined
- **Analyst revisions:** Use "published" timestamp, NOT "effective date"
- **News/filings:** Every release has precise timestamp
- **Insider Form 4:** Up to T+2 legal lag; use filing date NOT transaction date
- **Short interest:** Free FINRA bi-monthly too lagged; vendor data T+1 or T+2

## Verdict
"Yes, there is a 10x improvement available. The options data from Polygon.io is that one data source."

"Your current system is capturing the 'easy' 20% of retail-accessible alpha. The other 80% is locked away in the data sources we've discussed."
