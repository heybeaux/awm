# Design: Equity Benchmark v1

## Architecture Overview

```
┌─────────────────────────────────────────────────────┐
│                  Benchmark Harness                   │
│              (Python — orchestrates all)              │
├─────────────┬──────────────┬────────────┬────────────┤
│  XGBoost    │  AWM-alone   │ Le-WM-alone│  Fusion    │
│  baseline   │  (TS→JSON)   │ (Python)   │ (Both)     │
├─────────────┴──────────────┴────────────┴────────────┤
│              Feature Store (shared)                   │
│    Windows, splits, targets, ticker universe          │
├──────────────────────────────────────────────────────┤
│              Market Data Pipeline                     │
│       Download → Clean → Engineer → Split             │
└──────────────────────────────────────────────────────┘
```

## Directory Structure

```
awm/
├── bench/
│   ├── equity/                     # Equity-specific code
│   │   ├── data/                   # Downloaded market data (gitignored)
│   │   ├── pipeline.py             # Yahoo Finance download + feature engineering
│   │   ├── targets.py              # Target computation (direction_5d, regime)
│   │   ├── universe.py             # Ticker universe definition
│   │   ├── lewm_adapter.py         # Le-WM temporal encoder + training
│   │   ├── lewm_predict.py         # Le-WM embedding extraction + classification
│   │   ├── awm_bridge.py           # Python→TypeScript AWM bridge (subprocess + JSON)
│   │   ├── xgboost_baseline.py     # XGBoost on raw features
│   │   ├── backtest.py             # Simple long/flat backtest simulator
│   │   ├── benchmark.py            # Main harness — runs all variants
│   │   ├── report.py               # Markdown report generator
│   │   ├── config.yaml             # Hyperparameters, paths, thresholds
│   │   └── requirements.txt        # Python deps
│   └── results/                    # Output directory
│       ├── equity-v1.json          # Full results
│       ├── equity-v1.tsv           # Summary table
│       └── equity-v1-report.md     # Human-readable report
├── packages/
│   ├── core/                       # Existing AWM core (unchanged)
│   └── equity-store/               # New: SQLite-backed equity store
│       ├── src/
│       │   ├── index.ts
│       │   ├── equity-store.ts     # AWMStore implementation for equities
│       │   ├── regime.ts           # Market regime classifier
│       │   └── session.ts          # Market session mapper
│       ├── package.json
│       └── tsconfig.json
├── examples/
│   └── equity-predict.ts           # CLI: run AWM prediction on equity data
```

## Ticker Universe

Default universe of ~50 instruments, configurable via `universe.py`:

**Mega-cap (high liquidity, well-studied):**
AAPL, MSFT, NVDA, AMZN, GOOGL, META, TSLA, BRK-B, JPM, V

**Sector ETFs (broad market representation):**
XLF (financials), XLE (energy), XLK (tech), XLV (healthcare), XLI (industrials), XLC (comms), XLY (discretionary), XLP (staples), XLU (utilities), XLRE (real estate), XLB (materials)

**Broad market / index ETFs:**
SPY, QQQ, IWM, DIA, VTI

**Cross-asset (regime detection signals):**
GLD (gold), TLT (long bonds), HYG (high yield), UUP (dollar), USO (oil)

**Mid-cap growth (higher volatility, more signal potential):**
CRWD, DDOG, NET, SNOW, PLTR, MDB, ZS, COIN, SQ, SHOP

**Canadian-accessible, commission-friendly brokers:**
All above trade on NYSE/NASDAQ, accessible from Canadian brokerages (Questrade, IBKR, Wealthsimple Trade).

## Data Pipeline (Python)

### Data Source
- **Primary**: `yfinance` Python package (wraps Yahoo Finance API)
- **Frequency**: Daily OHLCV
- **History**: 5+ years per ticker (2021-01-01 to present, or max available)
- **Adjustments**: Adjusted close used for all return calculations (accounts for splits/dividends)
- **Rate limiting**: 0.5s delay between ticker fetches to avoid throttling

### Feature Engineering
Each window produces a feature vector per ticker:

| Feature | Dimensions | Description |
|---------|-----------|-------------|
| `daily_return` | 1 | (close - prev_close) / prev_close |
| `log_return` | 1 | log(close / prev_close) |
| `intraday_range` | 1 | (high - low) / close |
| `volume_ratio` | 1 | volume / SMA(volume, 20) |
| `roll_return_5d` | 1 | 5-day cumulative return |
| `roll_return_10d` | 1 | 10-day cumulative return |
| `roll_return_20d` | 1 | 20-day cumulative return |
| `roll_vol_5d` | 1 | 5-day rolling std of daily returns |
| `roll_vol_10d` | 1 | 10-day rolling std |
| `roll_vol_20d` | 1 | 20-day rolling std |
| `rsi_14` | 1 | Relative Strength Index (14-day) |
| `macd_hist` | 1 | MACD histogram (12,26,9) |
| `bb_pctb` | 1 | Bollinger Band %B (20,2) |
| `high_52w_pct` | 1 | Distance from 52-week high (0-1) |
| `day_of_week` | 1 | 0-4 (Mon-Fri) |
| `price_history` | 60×5 | Raw OHLCV for last 60 days (for Le-WM) |
| **Total raw features** | **15** | For XGBoost/AWM |
| **Total Le-WM input** | **300** | 60 days × 5 channels |

### Targets

**Primary — `direction_5d`** (binary):
```python
fwd_return_5d = close[t+5] / close[t] - 1
direction_5d = 1 if fwd_return_5d > 0.01 else 0  # >1% up = positive
```

**Secondary — `regime`** (categorical, for AWM stepType):
```python
ret_20d = close[t] / close[t-20] - 1
vol_20d = daily_returns[t-20:t].std()
if ret_20d > 0.05:    regime = 'trending_up'
elif ret_20d < -0.05: regime = 'trending_down'
elif vol_20d > median_vol: regime = 'mean_reverting'
else: regime = 'quiet'
```

### Splits
- Strict temporal: train (first 70%), val (next 15%), test (final 15%)
- Applied per-ticker independently
- No shuffling across time boundaries
- Threshold sweep subsets drawn from front of training set per ticker

## Le-WM Adaptation (Python)

### Encoder Swap
Replace `ViT encoder` with `TemporalEncoder`:

```python
class TemporalEncoder(nn.Module):
    """1D temporal encoder for equity OHLCV time-series.
    Input: (B, 60, 5) — batch of 60-day windows, 5 channels (OHLCV)
    Output: (B, 64) — latent embedding
    """
    def __init__(self, input_channels=5, d_model=64):
        # Conv1d stack operating on (B, 5, 60):
        # Conv1d(5, 32, kernel=5, stride=2, padding=2)   → (B, 32, 30)
        # Conv1d(32, 64, kernel=5, stride=2, padding=2)  → (B, 64, 15)
        # Conv1d(64, d_model, kernel=3, stride=2, padding=1) → (B, 64, 8)
        # AdaptiveAvgPool1d(1) → (B, 64, 1) → (B, 64)
        # LayerNorm(d_model)
```

### Predictor
Preserved from Le-WM: given current embedding + "action" (next day's known context — day of week, volume regime), predict next-day embedding.

### Training Objective
1. **Prediction loss**: MSE between predicted next-state embedding and actual next-state embedding
2. **Gaussian regularisation**: KL divergence between latent distribution and N(0,1), weight λ=0.01

### Embedding Extraction
```python
embedding = model.encode(ohlcv_window)  # shape (64,)
```

Export for AWM:
```python
{"date": "2024-01-15", "ticker": "AAPL", "embedding": [0.12, -0.34, ...], "features": {...}}
```

## AWM Equity Store (TypeScript)

### Regime Classification
```typescript
function classifyRegime(
  ret20d: number, vol20d: number, medianVol: number
): 'trending_up' | 'trending_down' | 'mean_reverting' | 'quiet' {
  if (ret20d > 0.05) return 'trending_up';
  if (ret20d < -0.05) return 'trending_down';
  if (vol20d > medianVol) return 'mean_reverting';
  return 'quiet';
}
```

### Profile Mapping
AWM's `profileSlug` maps to ticker symbol. This gives per-stock belief tracking — AWM learns that TSLA behaves differently from SPY.

### Store Implementation
- SQLite via `better-sqlite3` (single file, zero dependencies)
- Tables: `traces`, `beliefs`, `arms`, `patterns`
- All writes synchronous for benchmark determinism
- Separate DB file per benchmark run for clean comparisons

## Cross-Language Bridge

Python calls AWM (TypeScript) via subprocess:
```python
def awm_predict(features: dict, embedding: list | None) -> dict:
    input_json = json.dumps({"features": features, "embedding": embedding})
    result = subprocess.run(
        ["npx", "tsx", "examples/equity-predict.ts"],
        input=input_json, capture_output=True, text=True
    )
    return json.loads(result.stdout)
```

Batch mode for threshold sweeps (single process, NDJSON pipe).

## Backtest Simulator

Simple long/flat strategy to convert predictions into financial metrics:

```python
def backtest(predictions, actuals, prices, cost_bps=10):
    """
    predictions: array of P(up) per day
    actuals: array of actual 5d returns
    prices: array of close prices
    cost_bps: round-trip transaction cost in basis points
    
    Strategy: Go long when P(up) > 0.55, flat otherwise.
    Position size proportional to confidence: size = (P - 0.5) * 2
    Hold for 5 days, no overlapping positions.
    
    Returns: Sharpe ratio, max drawdown, total return, win rate, trade count
    """
```

## Benchmark Matrix

| Model | Input | Language | Notes |
|-------|-------|----------|-------|
| XGBoost | 15 raw features | Python | Gradient boosted trees baseline |
| AWM-alone | Raw features → regime/ticker | TypeScript | Bayesian online learner |
| Le-WM-alone | 60d OHLCV → embedding → linear head | Python | Neural latent + classifier |
| Fusion | 60d OHLCV → Le-WM embedding → AWM Oracle | Both | The thesis under test |

## Metrics

| Metric | What it measures | Why it matters |
|--------|-----------------|----------------|
| AUC-ROC | Discrimination ability | Can it separate winners from losers? |
| Brier score | Calibration | When it says 70% up, does it go up 70% of the time? |
| Accuracy | Overall correctness | Sanity check |
| Precision@90Recall | Precision catching 90% of moves | Trade quality |
| Sharpe ratio | Risk-adjusted return | Is the edge worth the volatility? |
| Max drawdown | Worst peak-to-trough | Can you stomach the losses? |
| Win rate | % of profitable trades | Psychological sustainability |
| Profit factor | Gross profit / gross loss | Overall edge magnitude |

## Hyperparameters (config.yaml)

```yaml
data:
  start_date: "2019-01-01"
  end_date: "2026-04-18"
  lookback_days: 60
  stride_days: 1
  direction_threshold: 0.01     # 1% for positive classification
  train_ratio: 0.70
  val_ratio: 0.15
  test_ratio: 0.15

lewm:
  d_model: 64
  input_channels: 5             # OHLCV
  n_conv_layers: 3
  lr: 1e-3
  epochs: 100
  batch_size: 256
  reg_lambda: 0.01
  patience: 10

awm:
  cost_weight: 0.3
  skip_confidence: 0.85
  routing_confidence: 0.7

xgboost:
  n_estimators: 300
  max_depth: 6
  learning_rate: 0.1
  subsample: 0.8
  colsample_bytree: 0.8

backtest:
  entry_threshold: 0.55         # P(up) > this to enter
  cost_bps: 10                  # round-trip cost estimate
  hold_days: 5
  max_positions: 5              # max concurrent (for portfolio sim)

benchmark:
  thresholds: [100, 250, 500, 1000, 2500, 5000]
  seeds:
    data: 42
    model: 123
  device: mps
```

## Phase 6 — Live Trading Architecture

### Overview

If Phases 4-5 confirm a tradeable edge (Sharpe > 1.0 on simple backtest, > 0.8 after
realistic costs), the system transitions from research to production through three stages:
QuantConnect rigorous backtesting → IBKR paper trading → IBKR live TFSA trading.

### Production Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   Daily Signal Pipeline                  │
│              (Cron: 9:00 AM ET, pre-market)              │
├─────────────────────────────────────────────────────────┤
│  1. Fetch latest OHLCV (yfinance or IBKR real-time)     │
│  2. Compute features (pipeline.py)                       │
│  3. Le-WM encode → 64-dim embeddings (CPU inference)     │
│  4. AWM predict → P(up), confidence, regime              │
│  5. Signal filter: P(up) > entry_threshold?              │
│  6. Position sizing: (P - 0.5) * 2, capped at 20%       │
├─────────────────────────────────────────────────────────┤
│                   Execution Layer                        │
│              (ib_insync → IBKR TWS/Gateway)              │
├─────────────────────────────────────────────────────────┤
│  7. Submit limit orders at mid-price                     │
│  8. Monitor fills (30s timeout, cancel if unfilled)      │
│  9. Track positions, enforce 5-day hold                  │
│  10. Close expired positions at market open              │
├─────────────────────────────────────────────────────────┤
│                   Feedback Loop                          │
│            (Online learning — AWM beliefs update)         │
├─────────────────────────────────────────────────────────┤
│  11. Record trade outcomes in AWM equity store            │
│  12. Beliefs update: Beta(α,β) ← outcome                │
│  13. Le-WM retrained monthly on expanding window          │
│  14. Signals to Engram + Telegram for audit trail         │
└─────────────────────────────────────────────────────────┘
```

### Key Dependencies

| Component | Library/Service | Purpose |
|-----------|----------------|---------|
| Data feed | yfinance (free) or IBKR API | Daily OHLCV |
| Backtest | QuantConnect LEAN (Python) | Realistic execution model |
| Broker | Interactive Brokers TWS API | Order execution |
| Broker lib | `ib_insync` (Python) | Async IBKR wrapper |
| Account | IBKR TFSA (Canadian) | Tax-free gains |
| Monitoring | Telegram (via OpenClaw) | P&L alerts, trade notifications |
| Audit | Engram | Signal + outcome logging |
| Model serving | Local (CPU) | Le-WM inference (~1s for 40 tickers) |

### Risk Controls

| Control | Value | Action |
|---------|-------|--------|
| Max position size | 20% of portfolio | Cap SetHoldings |
| Max concurrent positions | 5 | Queue excess signals |
| Max daily loss | 2% of portfolio | Halt trading for day |
| Max drawdown | 10% | Pause trading, alert Beaux |
| Kill switch drawdown | 15% | Liquidate all, full stop |
| Min hold period | 5 trading days | Prevents CRA day-trading flag |
| Order timeout | 30 seconds | Cancel unfilled limits |
| Max signal age | 4 hours | Discard stale signals |

### Canadian TFSA Considerations

- **Contribution room**: ~$95K+ (accumulated since 2009)
- **Tax treatment**: All gains inside TFSA are 100% tax-free
- **No PDT rule**: Canada has no pattern day trader restriction
- **Superficial loss rule**: 30-day wash sale equivalent — our 5-day hold + monthly rebalance avoids this
- **CRA risk**: Frequent day trading in TFSA can be deemed "carrying on business" — our systematic, rules-based, multi-day-hold approach is well within safe territory
- **Documentation**: Keep algorithm specs + trade logs as evidence of systematic (not speculative) trading

### QuantConnect Integration

QuantConnect's LEAN engine runs the same signal logic with institutional-grade execution simulation:

```python
class AWMFusionAlgorithm(QCAlgorithm):
    def Initialize(self):
        self.SetStartDate(2022, 1, 1)
        self.SetCash(100000)
        self.SetBrokerageModel(InteractiveBrokersBrokerageModel())
        
        # Load Le-WM encoder (ONNX export)
        self.lewm_model = self.ObjectStore.ReadBytes("lewm_encoder.onnx")
        
        # AWM belief engine (pure Python port or subprocess)
        self.awm = AWMBeliefEngine()
        
        # Universe
        for ticker in UNIVERSE:
            self.AddEquity(ticker, Resolution.Daily)
        
        # Daily signal generation at market open
        self.Schedule.On(
            self.DateRules.EveryDay(),
            self.TimeRules.AfterMarketOpen("SPY", 30),
            self.GenerateSignals
        )
    
    def GenerateSignals(self):
        for ticker in self.ActiveSecurities:
            features = self.compute_features(ticker)
            embedding = self.lewm_encode(features)
            signal = self.awm.predict(ticker, embedding)
            
            if signal.p_up > self.entry_threshold:
                weight = (signal.p_up - 0.5) * 2
                self.SetHoldings(ticker, min(weight, 0.20))
```

The LEAN backtest validates: realistic fills, commission impact, capital constraints, and concurrent position limits — things our `backtest.py` cannot model.
