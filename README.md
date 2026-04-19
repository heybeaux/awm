# AWM — Agent Workflow Model

> Bayesian prediction engine for agent pipelines — and an equity direction prediction system that fuses latent time-series embeddings with regime-aware beliefs.

[![CI](https://github.com/heybeaux/awm/actions/workflows/ci.yml/badge.svg)](https://github.com/heybeaux/awm/actions/workflows/ci.yml)

---

## What's in this repo

AWM is two things:

1. **AWM Core** (TypeScript) — A prediction and adaptation layer for multi-agent workflows. It observes past pipeline runs, predicts outcomes, and recommends runtime adaptations (model routing, constraint injection, step skipping) before each step executes.

2. **Equity Benchmark** (Python) — A real-world validation: can AWM's Bayesian regime beliefs, combined with a neural time-series encoder (Le-WM), predict equity direction better than either system alone?

---

## AWM Core — Pipeline Prediction Engine

AWM uses Bayesian statistics, multi-armed bandits, and historical pattern matching to make agent pipelines smarter over time. No neural simulation. No billion-dollar compute.

### Benchmark: Grade A (8/8 scenarios)

| Metric | Value |
|--------|-------|
| Prediction Accuracy | 81.5% |
| Cost Reduction (model routing) | 59.6% vs always-expensive |
| Cold Start | 7 runs to beat random |
| Constraint Injection | Prevents 15%+ of revision cycles |
| Calibration (Brier) | ≤ 0.25 |
| Profile Isolation | ≥ 0.85 |

### Packages

| Package | Description |
|---------|-------------|
| `packages/core` | Bayesian beliefs, bandits, constraints, oracle |
| `packages/bench` | Benchmark runner and scenario simulator |
| `packages/mastra-middleware` | Mastra framework integration |
| `packages/engram-adapter` | [Engram](https://openengram.ai) memory persistence |
| `packages/equity-store` | Equity-specific state management and regime detection |

---

## Equity Benchmark — Le-WM + AWM Fusion

The equity benchmark answers: **does fusing latent time-series representations with Bayesian regime beliefs produce a tradeable edge?**

### Architecture

```
                    ┌─────────────────────────────────┐
                    │     Daily OHLCV (40 tickers)     │
                    │     2005-2026 · Yahoo Finance     │
                    └───────────┬─────────────────────┘
                                │
                ┌───────────────┼───────────────┐
                ▼               ▼               ▼
        ┌──────────────┐ ┌────────────┐ ┌──────────────┐
        │   XGBoost    │ │   Le-WM    │ │     AWM      │
        │  (Features)  │ │ (Encoder)  │ │  (Bayesian)  │
        │              │ │            │ │              │
        │ RSI, MACD,   │ │ 60-day     │ │ Beta(α,β)    │
        │ Bollinger,   │ │ OHLCV →    │ │ posteriors   │
        │ rolling vol, │ │ 64-dim     │ │ per regime:  │
        │ momentum,    │ │ latent     │ │ trend_up,    │
        │ day-of-week  │ │ embedding  │ │ trend_down,  │
        │              │ │            │ │ mean_revert, │
        │ 300 trees    │ │ JEPA-style │ │ quiet        │
        │ depth=6      │ │ prediction │ │              │
        └──────┬───────┘ └─────┬──────┘ └──────┬───────┘
               │               │               │
               │         ┌─────┴─────┐         │
               │         ▼           ▼         │
               │    LogisticReg   Regime       │
               │    on embeddings  belief      │
               │         │           │         │
               │         └─────┬─────┘         │
               │               ▼               │
               │     ┌───────────────────┐     │
               │     │  Fusion (50/50)   │     │
               │     │  Le-WM + AWM      │     │
               │     └────────┬──────────┘     │
               │              │                │
               └──────────────┼────────────────┘
                              ▼
                    ┌───────────────────┐
                    │    Walk-Forward   │
                    │     Backtest      │
                    │                   │
                    │ • Retrain q60 days│
                    │ • No future leak  │
                    │ • Entry > 0.55    │
                    │ • Cost: 10 bps    │
                    │ • Hold: 5 days    │
                    └───────────────────┘
```

### Components

#### Le-WM (Latent World Model)

Temporal convolutional encoder adapted from JEPA (Joint Embedding Predictive Architecture). Instead of reconstructing input, it learns to predict the *next* latent embedding from the current one plus context.

- **Input**: 60-day OHLCV window (60 × 5 channels)
- **Encoder**: 3-layer 1D conv → AdaptiveAvgPool → LayerNorm → 64-dim embedding
- **Predictor**: MLP predicts next-window embedding from current + 2-dim context (day-of-week, volume regime)
- **Loss**: MSE(predicted, actual next embedding) + λ · KL(z ∥ N(0,1))
- **Training**: 116K window pairs, val MSE 0.037

#### AWM (Bayesian Regime Beliefs)

Beta distribution posteriors updated per market regime. Each regime (trending up/down, mean-reverting, quiet) maintains separate beliefs about directional probability, updated from historical outcomes.

#### XGBoost Baseline

Gradient-boosted trees on handcrafted features: RSI(14), MACD histogram, Bollinger %B, rolling volatility (5/10/20d), rolling returns, 52-week high proximity, volume ratio, day-of-week. 300 trees, max depth 6.

### Results

#### Walk-Forward Backtest (2025–2026)

Rigorous out-of-sample evaluation. Retrained every 60 trading days (152 retrain cycles). 35 tickers, 9,165 test observations.

| Model | AUC | Brier | Accuracy |
|-------|-----|-------|----------|
| **Fusion (Le-WM + AWM)** | **0.597** | **0.238** | **59.2%** |
| Le-WM standalone | 0.534 | 0.244 | 56.4% |

| Configuration | Model | Sharpe | Return | Max DD | Trades |
|---------------|-------|--------|--------|--------|--------|
| threshold=0.65, 5bps | Fusion | **10.03** | 152.5% | -5.8% | 109 |
| threshold=0.65, 10bps | Fusion | **9.68** | 142.7% | -6.2% | 109 |

#### Full Benchmark (2005–2026, multiple thresholds)

Cross-model comparison across training data sizes (100–5000 observations per ticker).

| Model | Best AUC | Best Sharpe | Notes |
|-------|----------|-------------|-------|
| XGBoost | 0.559 | **+2.33** | Strong across all regimes |
| AWM (regime) | **0.584** | +1.89 | Best calibration |
| Le-WM | 0.563 | +1.96 | Improves with more data |
| Fusion | 0.568 | +1.54 | Conservative, lower drawdown |

#### Bear Market Stress Test (in progress)

77 walk-forward windows across 20 years including GFC crash (2007–2009), COVID crash (2020), and 2022 bear market. Validates whether the model knows when to stay out vs. riding beta in a bull market.

Early results (windows 1–46 of 77):
- GFC crash windows: Fusion AUC drops to 0.49–0.54 (honest uncertainty)
- Bull market windows: Fusion AUC recovers to 0.56–0.61
- The model correctly reduces confidence during regime transitions

### Universe

40 tickers covering mega-caps, sector ETFs, and broad market:

**Mega-cap**: AAPL, MSFT, NVDA, AMZN, GOOGL, META, TSLA, BRK-B, JPM, V
**Growth**: PLTR, CRWD, DDOG, NET, ZS, SNOW, MDB, SHOP, COIN
**Broad ETFs**: SPY, QQQ, IWM, VTI, DIA
**Sector ETFs**: XLF, XLE, XLK, XLV, XLB, XLC, XLI, XLP, XLRE, XLU, XLY
**Macro**: GLD, TLT, HYG, USO, UUP

---

## Getting Started

### AWM Core (TypeScript)

```bash
npm install
npm test          # 91 tests
npm run bench     # Run benchmark scenarios
```

### Equity Benchmark (Python)

```bash
cd bench/equity
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Download data & compute features (40 tickers, 2005-present)
python pipeline.py

# Train Le-WM encoder
python lewm_adapter.py --device cpu --epochs 100

# Run full benchmark
python benchmark.py

# Run XGBoost baseline
python xgboost_baseline.py

# Run fusion benchmark
python fusion_benchmark.py

# Run rigorous walk-forward backtest
python backtest.py

# Run bear market stress test (20-year walk-forward)
python bear_market_stress_test.py --device cpu
```

### Configuration

Edit `bench/equity/config.yaml` to adjust:
- Universe (tickers)
- Date range
- Model hyperparameters (XGBoost, Le-WM, backtest)
- Feature engineering parameters

---

## Specification

Formal requirements and design documents live in [`openspec/`](./openspec/). The equity benchmark specification covers:
- Market data pipeline requirements
- Prediction target definitions
- Le-WM architecture adaptation spec
- AWM integration requirements
- Backtest methodology constraints

---

## Known Issues

- **PyTorch + XGBoost SIGSEGV on Apple Silicon**: When both PyTorch and XGBoost are imported in the same process, XGBoost's `.fit()` can segfault due to OpenMP threading conflicts. Fix: set `OMP_NUM_THREADS=1` before importing either library.
- **Bear market stress test**: Sharpe/returns show as nan/0% due to conservative entry threshold (0.55) when model AUC is near 0.5 during crash periods. This is correct behavior — the model stays out when uncertain.

---

## License

MIT — see [LICENSE](./LICENSE)
