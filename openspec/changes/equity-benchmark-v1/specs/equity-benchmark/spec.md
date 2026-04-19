# Equity Prediction Benchmark

## Purpose
Validate the AWM + Le-WM fusion thesis using publicly available equity market data. The core question: does combining Le-WM's learned latent representations with AWM's Bayesian decision engine outperform either system alone — and how much data do we need before it works?

This is both a research experiment and a direct monetisation path. If the fusion system produces a measurable edge, it can be deployed immediately in a Canadian TFSA (tax-free compounding, no PDT rule, no wash sale rule). No sales cycle, no enterprise customers — the builder is the user.

## ADDED Requirements

### Requirement: Market data pipeline
The system SHALL ingest historical equity OHLCV data and produce clean, windowed time-series suitable for both Le-WM training and AWM prediction.

#### Scenario: Historical data ingestion
- GIVEN Yahoo Finance provides free daily OHLCV data for US-listed equities
- WHEN the pipeline fetches data for a configurable universe of tickers (default: 50 liquid stocks + ETFs)
- THEN it downloads at least 5 years of contiguous daily history per ticker
- AND each record contains: date, open, high, low, close, adjusted_close, volume, ticker
- AND splits/dividends are adjusted (using adjusted close)
- AND data gaps (holidays, halts) are documented but not interpolated

#### Scenario: Feature engineering
- GIVEN raw daily OHLCV data for a ticker
- WHEN the pipeline constructs training windows
- THEN each window contains: price history (configurable lookback, default 60 trading days), daily return, log return, intraday range (high-low)/close, volume ratio (volume / 20-day avg volume), rolling return (5d, 10d, 20d), rolling volatility (5d, 10d, 20d), RSI(14), MACD (12,26,9) histogram, Bollinger Band %B(20,2), 52-week high proximity, and day-of-week
- AND windows are generated with stride of 1 trading day
- AND train/val/test splits are strictly temporal (no future leakage): train = first 70%, val = next 15%, test = final 15%

#### Scenario: Multi-ticker universe
- GIVEN a universe of tickers
- WHEN the pipeline processes all tickers
- THEN each ticker is processed independently (no cross-ticker features in v1)
- AND the universe includes a mix of: mega-cap stocks (AAPL, MSFT, NVDA, AMZN, GOOGL), sector ETFs (XLF, XLE, XLK, XLV), broad market ETFs (SPY, QQQ, IWM), volatility-adjacent (GLD, TLT, VIX-adjacent via UVXY), and mid-cap growth names
- AND the universe is configurable via config file

#### Scenario: Data validation
- GIVEN constructed feature windows
- WHEN validation runs
- THEN it reports: total windows per ticker, date range covered, any gaps > 3 trading days, feature distributions, and class balance for the prediction target
- AND the pipeline warns if any ticker has fewer than 1000 trading days of history

### Requirement: Prediction targets
The system SHALL define clear, tradeable prediction targets that map to position-sizing decisions.

#### Scenario: Primary target — next-5-day direction
- GIVEN the pipeline produces windowed features
- WHEN the target is computed
- THEN `direction_5d` is a binary label: 1 if the 5-trading-day forward return exceeds +1%, 0 otherwise
- AND this represents a swing-trade thesis: "Is this stock likely to move up meaningfully this week?"
- AND the threshold (+1%) is configurable

#### Scenario: Secondary target — regime classification
- GIVEN the pipeline produces windowed features
- WHEN the regime target is computed
- THEN `regime` is one of: trending_up (20d return > +5%), trending_down (20d return < -5%), mean_reverting (20d return between -5% and +5% with 20d volatility > median), or quiet (otherwise)
- AND this maps to AWM's stepType for belief tracking

### Requirement: Le-WM time-series adaptation
The system SHALL adapt Le-WM's JEPA architecture from pixel-based control tasks to 1D equity time-series.

#### Scenario: Temporal encoder replaces vision encoder
- GIVEN Le-WM's original architecture uses a ViT encoder on pixel frames
- WHEN adapted for equity data
- THEN the vision encoder is replaced with a temporal encoder: 1D convolutions on windowed OHLCV sequences (60 days × 5 channels)
- AND the predictor network architecture (predict next embedding from current embedding + context) is preserved unchanged
- AND the Gaussian regularisation loss on latent space is preserved
- AND the model trains on the same two-loss objective as the original Le-WM

#### Scenario: Embedding extraction
- GIVEN a trained Le-WM equity model
- WHEN a new price window is presented
- THEN the encoder produces a fixed-dimension latent embedding (configurable, default 64-dim)
- AND embeddings are extractable via a simple Python function: `encode(window) -> np.ndarray`
- AND embeddings can be exported as JSON or NumPy arrays for cross-language consumption by AWM (TypeScript)

#### Scenario: Le-WM standalone prediction
- GIVEN Le-WM embeddings and a classification head
- WHEN predicting direction_5d for the next period
- THEN Le-WM's standalone AUC, Brier score, and accuracy are recorded
- AND this serves as the Le-WM-alone baseline

### Requirement: AWM equity store and prediction
The system SHALL implement an AWM store adapter and prediction pipeline for equity data.

#### Scenario: Equity store adapter
- GIVEN AWM's `AWMStore` interface requires: storeTrace, queryTraces, getBelief, setBelief, getArms, setArm, getPatterns, setPattern
- WHEN implemented for the equity domain
- THEN traces map to: stepType = market_regime (trending_up/trending_down/mean_reverting/quiet), profileSlug = ticker or sector grouping
- AND beliefs track P(direction_5d=up | regime, ticker) via Beta distributions
- AND the store persists to SQLite for reproducibility

#### Scenario: AWM raw-feature prediction
- GIVEN AWM's Oracle with the equity store
- WHEN predicting from raw technical features without Le-WM embeddings
- THEN features are encoded as a PredictionContext with stepType derived from current regime and profileSlug from ticker
- AND AWM's prediction (pass/revise/fail mapped to up/uncertain/down) is compared against the actual 5-day outcome
- AND AUC, Brier score, and accuracy are recorded as the AWM-alone baseline

#### Scenario: AWM consumes Le-WM embeddings
- GIVEN Le-WM produces 64-dim embeddings for each price window
- WHEN AWM's Oracle receives a PredictionContext augmented with Le-WM embeddings
- THEN embeddings are used as additional features for prediction context (embedding-derived regime classification fed to stepType, embedding hash as inputFingerprint)
- AND the Oracle's belief updates incorporate Le-WM's latent signal
- AND this is the fusion system under test

### Requirement: Benchmark harness
The system SHALL run all model variants at multiple data thresholds and produce comparable metrics.

#### Scenario: Model variants
- GIVEN four systems to compare
- WHEN the benchmark runs
- THEN it evaluates: (1) XGBoost baseline on raw features, (2) AWM-alone on raw features, (3) Le-WM-alone with classification head, (4) AWM + Le-WM fusion
- AND all four use identical train/val/test splits
- AND all four predict the same target: direction_5d

#### Scenario: Data threshold sweep
- GIVEN the benchmark harness
- WHEN run with threshold sweep enabled
- THEN it trains and evaluates each model variant at: 100, 250, 500, 1000, 2500, 5000 trading days of history (per ticker)
- AND subsets are drawn from the front of the training set (simulating "we only have N days of data")
- AND validation and test sets remain fixed across all thresholds

#### Scenario: Metrics collection
- GIVEN a completed benchmark run
- WHEN results are collected
- THEN for each (model, threshold) pair, the following are recorded: AUC-ROC, Brier score, accuracy, precision@90recall, Sharpe ratio (simulated: go long when predicted up, flat otherwise), max drawdown (simulated), training time, inference time per window
- AND results are saved as JSON and as a TSV summary table
- AND the output identifies: the minimum N where fusion beats all baselines, the minimum N where any model beats buy-and-hold, and the winner at each threshold

#### Scenario: Reproducibility
- GIVEN benchmark results
- WHEN another researcher runs the same benchmark
- THEN results are deterministic (fixed random seeds: 42 for data, 123 for models)
- AND all hyperparameters are logged in the results JSON
- AND the data date range and ticker universe are recorded

### Requirement: Results reporting
The system SHALL produce human-readable output summarising the experiment.

#### Scenario: Summary report
- GIVEN completed benchmark results
- WHEN the report generator runs
- THEN it produces a markdown report with: experiment description, data summary (date range, universe, class balance), results table (model × threshold → AUC + Sharpe), key findings (minimum viable N, fusion advantage, crossover points), and raw JSON path
- AND the report answers three core questions: (1) Does fusion beat individual systems? (2) At what N? (3) Is the edge tradeable (Sharpe > 1.0 after simulated costs)?

## ADDED Out of Scope
- Real-time trading execution or broker integration (this is offline benchmarking only)
- Intraday data or minute-level granularity (daily only for v1)
- Cross-ticker features, pairs trading, or portfolio optimisation (single-ticker predictions for v1)
- Options or derivatives
- Fundamental data (earnings, revenue, macro indicators — future enhancement)
- Live paper trading (Phase 2 of the project, after benchmark validates)
