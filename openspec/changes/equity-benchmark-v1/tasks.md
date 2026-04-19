# Tasks: Equity Benchmark v1

## Phase 1 — Data Pipeline (Python)
_No dependencies. Can start immediately._

- [ ] **1.1** Create `bench/equity/` directory structure and `requirements.txt` (torch, xgboost, pandas, numpy, scikit-learn, yfinance, pyyaml, ta-lib or manual indicators)
- [ ] **1.2** Create `config.yaml` with all hyperparameters from design doc
- [ ] **1.3** Implement `universe.py` — Ticker universe definition
  - Default 50-ticker universe (mega-cap, sector ETFs, broad ETFs, cross-asset, mid-cap growth)
  - Configurable via YAML override
  - Function: `get_universe(config) -> list[str]`
- [ ] **1.4** Implement `pipeline.py` — Market data download
  - Fetch daily OHLCV via `yfinance` for full universe
  - 0.5s delay between tickers to avoid throttling
  - Cache to Parquet: `data/{ticker}_raw.parquet`
  - Handle missing data, delistings, short histories gracefully
  - Validate: date range, gap detection, adjusted close continuity
- [ ] **1.5** Implement `pipeline.py` — Feature engineering
  - Daily/log returns, intraday range, volume ratio
  - Rolling returns and volatility (5d, 10d, 20d)
  - RSI(14), MACD histogram (12,26,9), Bollinger %B (20,2)
  - 52-week high proximity, day-of-week
  - 60-day OHLCV price history array for Le-WM
  - Output: `data/{ticker}_features.parquet`
- [ ] **1.6** Implement `targets.py` — Target computation
  - `direction_5d`: binary, 1 if 5-day forward return > +1%
  - `regime`: categorical (trending_up/trending_down/mean_reverting/quiet)
  - Configurable threshold and lookback
  - Compute and append to feature DataFrames
- [ ] **1.7** Implement `pipeline.py` — Train/val/test splitting
  - Strict temporal split per ticker (70/15/15)
  - Class balance reporting
  - Summary statistics (windows per ticker, date ranges, target distribution)
- [ ] **1.8** Implement `xgboost_baseline.py`
  - Train XGBoost on 15 raw features
  - Per-ticker and pooled (all tickers) training modes
  - Evaluate AUC, Brier, accuracy, precision@90recall
  - Support threshold sweep (subset training data)
  - Save results to JSON

## Phase 2 — Le-WM Adaptation (Python)
_Depends on: Phase 1 (needs feature data)_

- [ ] **2.1** Implement `lewm_adapter.py` — TemporalEncoder
  - 1D conv stack: (B, 5, 60) → (B, 64)
  - JEPA predictor: predict next-day embedding from current + context
  - Gaussian regularisation loss (KL to N(0,1))
  - Full training loop with early stopping on val loss
- [ ] **2.2** Implement `lewm_adapter.py` — Training harness
  - DataLoader for windowed OHLCV histories
  - MPS device support with CPU fallback
  - Checkpoint saving (best val loss)
  - Training can be per-ticker or pooled across universe
  - Log training curves to stdout (no WandB dependency)
- [ ] **2.3** Implement `lewm_predict.py` — Embedding extraction
  - Load trained checkpoint, freeze encoder
  - `encode(window) -> np.ndarray` function (single and batch)
  - Export to NDJSON: `{"date", "ticker", "embedding", "features"}`
  - Save full embedding matrix: `data/{ticker}_embeddings.npy`
- [ ] **2.4** Implement `lewm_predict.py` — Standalone classification
  - Linear head on frozen embeddings → P(direction_5d)
  - Train on training set, evaluate on test set
  - Report AUC, Brier, accuracy, precision@90recall
  - Support threshold sweep
  - Save results to JSON

## Phase 3 — AWM Equity Store (TypeScript)
_No dependencies on Phase 1-2. Can run in parallel._

- [ ] **3.1** Create `packages/equity-store/` package
  - `package.json` with deps: `better-sqlite3`, `@types/better-sqlite3`
  - `tsconfig.json` extending root config
- [ ] **3.2** Implement `regime.ts` — Market regime classifier
  - `classifyRegime(ret20d, vol20d, medianVol)` → regime string
  - Unit tests for boundary conditions
- [ ] **3.3** Implement `session.ts` — Profile/ticker mapping
  - Maps ticker to profileSlug (ticker symbol directly)
  - Optional sector grouping for cross-ticker beliefs
- [ ] **3.4** Implement `equity-store.ts` — SQLite AWMStore
  - Schema: traces, beliefs, arms, patterns tables
  - All AWMStore interface methods
  - Synchronous writes for determinism
  - `reset()` method for clean sweep between threshold runs
  - Separate DB file per run
- [ ] **3.5** Implement `examples/equity-predict.ts` — CLI predictor
  - Read NDJSON from stdin: `{features, embedding?, ticker, regime}`
  - Create Oracle with equity store
  - Output predictions as NDJSON: `{ticker, date, prediction, confidence, regime}`
  - `--record` flag to also record outcomes for belief updating
  - `--reset` flag to clear beliefs
  - `--db <path>` flag to specify SQLite file
- [ ] **3.6** Write integration tests
  - Test store CRUD operations
  - Test regime classification
  - Test Oracle predict→record→predict cycle with equity data shapes

## Phase 4 — Integration & Benchmark (Python + TypeScript)
_Depends on: Phases 1, 2, 3 all complete_

- [ ] **4.1** Implement `awm_bridge.py` — Python→TypeScript bridge
  - `awm_predict(features, embedding?, ticker, regime)` via subprocess
  - `awm_batch_predict(batch)` via NDJSON pipe
  - `awm_reset(db_path)` to clear beliefs between threshold sweeps
  - `awm_record(trace)` to feed outcomes
  - Error handling: timeout (30s), JSON parse errors, process crashes
  - Health check: verify TypeScript CLI works before starting benchmark
- [ ] **4.2** Implement `backtest.py` — Simple backtest simulator
  - Input: array of (date, ticker, predicted_prob, actual_5d_return)
  - Strategy: long when P(up) > threshold, flat otherwise
  - Position sizing: proportional to confidence (P - 0.5) * 2
  - Transaction costs: configurable basis points
  - Output: Sharpe, max drawdown, total return, win rate, profit factor, trade count
  - No overlapping positions per ticker (5-day hold then reassess)
- [ ] **4.3** Implement `benchmark.py` — Main harness
  - Load config from `config.yaml`
  - Run data pipeline if features don't exist
  - For each threshold in sweep:
    - Subset training data (first N days per ticker)
    - Train/evaluate XGBoost baseline
    - Train Le-WM + extract embeddings
    - Run AWM-alone (sequential with belief updates, per-ticker)
    - Run Fusion (Le-WM embeddings → AWM)
    - Run backtest on all four model outputs
    - Collect all metrics
  - Save full results JSON + TSV summary
  - Print progress to stdout
- [ ] **4.4** Implement sequential AWM evaluation
  - Feed windows in temporal order per ticker
  - AWM updates beliefs after each prediction + outcome reveal
  - Evaluate on test set after training-set belief accumulation
  - Track belief evolution over time (for analysis)
- [ ] **4.5** Implement fusion evaluation
  - Le-WM embeddings + raw features → AWM Oracle
  - Embedding-derived regime as stepType (cluster embeddings → regime labels)
  - Embedding hash as inputFingerprint for similarity detection
  - Same online learning protocol as AWM-alone
- [ ] **4.6** Implement `report.py` — Report generator
  - Load results JSON
  - Markdown report with: data summary, results tables (AUC + Sharpe by model × threshold), per-ticker breakdown for best model, key findings
  - Answer three core questions: (1) fusion beats individuals? (2) at what N? (3) tradeable edge (Sharpe > 1.0)?
  - Highlight crossover points and minimum viable data thresholds
  - ASCII chart of AUC by threshold for each model

## Phase 5 — Validation & Analysis
_Depends on: Phase 4_

- [ ] **5.1** Run full benchmark end-to-end
  - All 4 models × 6 thresholds × ~50 tickers
  - Verify reproducibility (two runs → identical results)
  - Profile runtime (estimate: 2-4 hours on M-series Mac)
- [ ] **5.2** Analyse results and interpret findings
  - Crossover points: where does neural beat statistical?
  - Per-regime analysis: does fusion shine in volatile regimes?
  - Per-sector analysis: does it work better on tech vs utilities?
  - Minimum viable N for useful predictions
  - Honest assessment: is there a tradeable edge or not?
- [ ] **5.3** Document results
  - Update AWM README with equity benchmark section
  - Commit results to `bench/results/`
  - If edge exists: proceed to Phase 6 (live trading pipeline)
  - If no edge: document what we learned and where the thesis breaks down

## Phase 6 — Live Trading Pipeline (QuantConnect + Interactive Brokers)
_Depends on: Phase 5 confirming tradeable edge (Sharpe > 1.0 on backtest)_
_Prerequisites: IBKR account with API access, QuantConnect account_

### 6A — QuantConnect Rigorous Backtest
- [ ] **6A.1** Set up QuantConnect LEAN project
  - Create algorithm in Python (QuantConnect's native Python support)
  - Configure IBKR brokerage model (commissions, slippage, margin)
  - Set universe to winning tickers from Phase 5 (or full 40)
- [ ] **6A.2** Port signal generation to LEAN
  - Daily scheduled event: download latest OHLCV → compute features → run Le-WM encoder → AWM predict
  - Le-WM inference: export frozen ONNX model or embed PyTorch in algo
  - AWM beliefs: SQLite store persisted between runs (or port Bayesian logic to pure Python)
  - Signal: P(up) > entry_threshold → SetHoldings(ticker, confidence_weight)
- [ ] **6A.3** Implement realistic execution model
  - Slippage: QuantConnect's VolumeShareSlippage (can't fill more than 1% of avg volume)
  - Fill model: ImmediateFillModel for liquid ETFs, PartialFillModel for mid-caps
  - Transaction costs: IBKR fee schedule (tiered or fixed, ~$0.005/share for US equities)
  - Market hours only (9:30 AM–4:00 PM ET), no after-hours fills
- [ ] **6A.4** Walk-forward optimization
  - Expanding window retraining: retrain Le-WM every 60 days on growing dataset
  - AWM belief reset frequency: test never-reset vs quarterly-reset vs yearly-reset
  - Entry threshold sweep: 0.52, 0.55, 0.58, 0.60
  - Position size cap: max 20% per ticker, max 5 concurrent positions
- [ ] **6A.5** Validate backtest results
  - Compare QuantConnect Sharpe vs our simple backtest.py Sharpe
  - Verify no look-ahead bias (features only use data available at signal time)
  - Check: max drawdown < 15%, win rate > 52%, avg trade return > 10bps after costs
  - Run on 2 separate time periods to check stability
  - Gate: Sharpe > 0.8 after realistic costs to proceed

### 6B — Paper Trading (IBKR)
- [ ] **6B.1** Set up IBKR paper trading account
  - Enable API access (TWS or Client Portal Gateway)
  - Configure paper trading with realistic starting capital matching TFSA balance
  - Set up IB Gateway for headless operation (no TWS GUI needed)
- [ ] **6B.2** Build signal pipeline service
  - Cron job: run daily at 9:00 AM ET (pre-market)
  - Fetch latest data → compute features → Le-WM encode → AWM predict
  - Output: list of (ticker, direction, confidence, position_size)
  - Log all signals to Engram for audit trail
- [ ] **6B.3** Implement IBKR order execution
  - Use `ib_insync` Python library (async, well-maintained)
  - Order types: Limit orders at mid-price, 30-second timeout, cancel if unfilled
  - Position management: track open positions, enforce 5-day hold, auto-close
  - Risk controls: max position size, max daily loss ($X), kill switch
  - Error handling: connection drops, order rejections, partial fills
- [ ] **6B.4** Build monitoring dashboard
  - Telegram notifications: daily P&L summary, trade alerts, error alerts
  - Track: cumulative return, Sharpe (rolling 30d), max drawdown, trade count
  - Alert thresholds: drawdown > 5% → warning, > 10% → pause trading
  - Weekly summary report to Beaux via Telegram
- [ ] **6B.5** Run paper trading for 4-6 weeks
  - Minimum 20 trade signals to assess statistical significance
  - Compare paper results to backtest expectations
  - Document any execution issues (latency, fills, data quality)
  - Gate: paper Sharpe within 20% of backtest Sharpe to proceed

### 6C — Live Trading (IBKR TFSA)
- [ ] **6C.1** Transition from paper to live
  - Switch IBKR connection from paper to live TFSA account
  - Start with 25% of intended capital allocation
  - Same risk controls as paper, with tighter limits initially
- [ ] **6C.2** Scale-up protocol
  - Week 1-2: 25% capital, max 2 positions
  - Week 3-4: 50% capital, max 3 positions (if Sharpe > 0 and drawdown < 5%)
  - Week 5-8: 75% capital, max 4 positions (if cumulative return > 0)
  - Week 9+: full allocation, max 5 positions (if all metrics hold)
- [ ] **6C.3** Ongoing monitoring and adaptation
  - AWM beliefs update automatically (online learning)
  - Le-WM retrained monthly on expanding window
  - Monthly performance review: is the edge persisting or decaying?
  - Quarterly: re-run full benchmark with latest data to check for regime shifts
  - Kill criteria: 3 consecutive losing weeks, drawdown > 10%, Sharpe < 0.3 over 30d
- [ ] **6C.4** Tax and compliance
  - Verify all trades are TFSA-eligible (no day trading pattern that CRA could flag)
  - Hold periods > 1 day satisfies TFSA rules (our 5-day hold is fine)
  - Track ACB (adjusted cost base) even though TFSA gains are tax-free
  - Document strategy for CRA audit protection (systematic, rules-based, not frequent)
