# Data Sources Panel: o4-mini
*Generated: 2026-04-19*

## Top 7 Signals (Ranked by 1-5 Day Predictive Power)

### 1. Order-Flow Imbalance (Footprint Data)
- Real money shows footprints in the DOM
- **Vendors:** QuantHouse (~$1k/mo), Exegy (~$1.5k/mo), IEX TOPS (free but coarse)
- **SNR:** 0.15-0.25 (Sharpe boost Δ0.1-0.2)

### 2. Short-Interest Dynamics & Borrow Fees
- Spikes in fee rates reveal squeeze risk
- **Vendors:** DataLend (~$500/mo), Markit (~$300-400/mo)
- **SNR:** 0.10-0.15

### 3. Dark-Pool Prints & Block Trades
- Institutions dump/accumulate off-exchange
- **Vendors:** Nasdaq TotalView (~$500/mo), Bloomberg B-PIPE (expensive), DTCC TAQ (free, lagged)
- **SNR:** 0.08-0.12

### 4. High-Frequency Sentiment (Newswire Tagging)
- Thomson Reuters News Analytics (~$200-400/mo), RavenPack (~$600/mo), or fine-tune HuggingFace transformer on headlines
- **SNR:** 0.05-0.1

### 5. Predictive Option-Flow
- Unusual call/put skew and sweeps
- **Vendors:** CBOE Silexx (~$100/mo), OCC daily reports (free, lagged)
- **SNR:** 0.05-0.08

### 6. US Treasury Real-Time Rates (Swap Spreads)
- Money-market stress ahead of equities
- **Vendors:** CME DataMine (~$200/mo)
- **SNR:** 0.03-0.05

### 7. Cross-Sectional Crowdedness (ETF Holdings Flows)
- Net creation/redemption data
- **Vendors:** Provider websites (free daily CSV), Refinitiv Lipper (~$300/mo)
- **SNR:** 0.02-0.04

## Le-WM Integration
- Raw tick-level order flow, dark-pool prints, and options sweeps are non-Gaussian, heavy-tailed, sparsely-populated signals that linear features destroy
- Feeding raw time-series lets Le-WM spot microstructure "shock events" and contagion failures
- Le-WM captures co-movement patterns (option sweeps → DOM liquidity crashes) that static ranks miss

## $200/mo Stack
- IEX TOPS (free) + ETF provider CSV (free) + OCC option open interest (free) + Refinitiv News via free tier (~$200/mo) + CME 1-mo Treasury futures (Quandl free tier)
- Total: $0-$200/mo

## Remove
- 52-week high, day-of-week, Bollinger %B, MACD histogram — zero marginal SNR once you have flow + sentiment

## Information Cascade
1. Smart-money accumulation: dark pool + block prints + buy-side algo footprints
2. Option-flow surges: unusual sweeps + steep skew moves
3. Broker short-cover: borrow-fee spike, locates fail
4. HFT response: DOM imbalance, quote stuffing
5. Retail momentum: social chatter lags by hours
6. CNBC: public coverage close to apex
→ Realistically tap steps 1-3 with tick data + borrow-fee monitoring

## Point-in-Time
- Order-flow & dark pool: real-time (aggregate intraday buckets for EOD)
- Option sweeps: sub-second
- Borrow-fee: daily T+1 (shift features)
- News sentiment: 1-5 sec latency (aggregate by daily close)
- ETF flows: published daily after close (T+0 to T+1)
- Treasury rates: real-time futures

## Verdict
"Your existing public-macro technicals likely capture ~60-80% of what retail can see. Adding a low-latency flow/sentiment stack can boost Sharpe by ~0.1-0.2 in walk-forward. None of these sources deliver a 10x improvement — true edge is incremental."

(Note: o4-mini was more conservative than other panelists on the improvement potential)
