# Data Sources Panel: GPT-5
*Generated: 2026-04-19*

## Top 7 Signals (Ranked by 1-5 Day Predictive Power)

### 1. Options Surface Dynamics & Dealer Positioning
- Changes in IV (level/term/skew), charm/vanna exposures, gamma regime flips, 0DTE flow spillover, put/call demand shocks, single-name skew kinks
- Dealers hedge mechanically; negative gamma = volatility amplification; skew/vanna shifts front-run spot
- **Vendors:** SpotGamma ($75-149/mo), SqueezeMetrics GEX ($79/mo), ORATS ($79-199/mo), Polygon/Intrinio (cost escalates)

### 2. Event-Driven Earnings/Guidance Pipeline
- Estimate revisions (direction/magnitude/breadth), pre-announce probability, guidance tone, transcript sentiment, earnings drift, blackout windows
- Real cash-flow information; desks trade whispers before headlines; drift around events is persistent
- **Vendors:** Estimize (contact), Zacks (>$200/mo), FinancialModelingPrep ($20-60/mo), Seeking Alpha transcripts (free scrape)

### 3. Microstructure/Flow: Off-Exchange & Dark Activity + Auction Imbalance
- TRF off-exchange share, ATS venue concentration, large late prints, closing auction imbalance, intraday net buying pressure
- Informed liquidity routes off-exchange/ATS; auction imbalances leak demand; predicts next-day drift
- **Vendors:** FINRA TRF (free T+1), FINRA ATS weekly (free), IEX Cloud ($10-50/mo), Polygon ($49-199+)

### 4. Corporate Action & Boardroom Tape
- Buyback authorizations + execution pace, Form 4 insider buys (clustered), 13D/G amendments, 8-K material agreements, shelf registrations
- Direct capital flow signals; buybacks create repeatable bid; activists change risk premia
- **Vendors:** SEC EDGAR (free), BamSEC ($33/mo), QuiverQuant ($15-25/mo)

### 5. Securities Lending/Short Squeeze Setup
- Borrow fees/utilization acceleration, on-loan % changes, short volume spikes vs float, fails-to-deliver
- Funding stress + crowded shorts = convex upside when combined with catalysts
- **Vendors:** ORTEX ($59-99/mo), IBKR (free with account), FINRA short volume (free T+1)

### 6. ETF Primary Market Flows & Mechanical Rebalances
- Creations/redemptions, index add/deletes, quarterly rebalance dates, sector ETF flow shocks
- Predictable supply/demand at constituent level; spillovers via SPY/QQQ gamma/liquidity
- **Vendors:** Issuer sites (free), ETF.com (free), calendar providers (free)

### 7. Vol-of-Vol & Skew State (Systemic)
- VVIX, VIX1D term structure, skew indices, SPX/SPY gamma, dealer exposure around OpEx
- Controls shock transmission; identifies fragile vs mean-reverting tape at 1-3d
- **Vendors:** Cboe VIX/VIX1D (free), Nasdaq Data Link ($50-150/mo)

## Le-WM Integration
- Nonlinear phase shifts: SPX dealer gamma crossing zero + rising VVIX + negative term structure → regime flip next 1-3d
- Path dependence: charm/vanna decay into big expiries sets up predictable spot drift
- Cross-tape synchronization: ETF flows + off-exchange surges + skew steepening = "fragile liquidity"
- Event hazard: days-to-earnings + blackout windows + IV term premium = asymmetric outcomes

## $200/mo Stack
- SpotGamma Core (~$75-149/mo) OR SqueezeMetrics GEX (~$79/mo)
- FinancialModelingPrep ($20-60/mo) for earnings calendars + transcripts
- SEC EDGAR (free) + BamSEC ($33/mo) or QuiverQuant ($15-25/mo)
- FINRA TRF + ATS (free)
- IEX Cloud ($10-50/mo)
- Total: ~$160-190/mo

## Remove from Current Set
- RSI-14 + MACD + Bollinger %B → keep at most one normalized mean-reversion factor
- Day-of-week: decayed/arb'd
- Too many commodity z-scores for mega-caps
- 52-week high distance: weak at 1-5d
- HMM may be redundant with Le-WM embeddings

## Information Cascade
1. T-X days: Insiders aware; borrow demand ticks up; options makers see unusual RFQs → skew/IV bid
2. T-1 to T0: Routing shifts off-exchange; IV term structure kinks; gamma may flip negative
3. T0: 8-K/press release drops; price gaps; media picks up
4. T+1: Post-event drift; buybacks restart; analysts revise
→ Position at Stage 1-2 via options surface + EDGAR + off-exchange monitoring

## Point-in-Time Lags
- Options surface: real-time; snapshot at 4:15pm ET
- SpotGamma/SqueezeMetrics: same-day post-close (minutes to hour)
- EDGAR: public within minutes of filing; use filings before 16:00 ET only
- FINRA TRF: T+1 evening
- FINRA ATS: weekly (Thursdays)
- Form 4: up to T+2 legal lag
- ETF shares outstanding: T+1
- VVIX/skew: depends on vendor

## Headroom Assessment
- Current stack captures 60-75% of retail-accessible alpha
- Options/dealer positioning data is closest to 10x improvement — mechanistically governs near-term flow
- Second: near-real-time EDGAR parsing for buybacks/guidance
- Best free signal: FINRA TRF off-exchange share for next-day drift
