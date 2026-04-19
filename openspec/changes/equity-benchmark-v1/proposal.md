# Change Proposal: Equity Benchmark v1

## Summary
Build the AWM + Le-WM fusion benchmark for equity price prediction. Validates whether combining Le-WM's learned latent representations with AWM's Bayesian decision engine outperforms either alone, identifies the minimum data threshold for useful predictions, and establishes whether the system produces a tradeable edge.

## Motivation
- AWM has 81.5% prediction accuracy on synthetic pipeline data but has never been tested on real-world continuous-domain prediction
- Le-WM (JEPA) learns latent state representations from sequential data but has only been validated on pixel-based control tasks
- The fusion thesis — Le-WM answers "what is the state?" and AWM answers "what should we do?" — is untested
- Equity markets provide the fastest validation loop: free data, immediate applicability, no sales cycle
- Canadian regulatory advantages (no PDT rule, TFSA tax-free compounding, no wash sale rule) make this a viable direct monetisation path
- Prior validation (`lewm-validation/`) showed XGBoost AUC of 0.60-0.63 on synthetic ad data — we need real-world comparison
- Even if trading doesn't pan out, the benchmark validates AWM+Le-WM on a challenging real-world domain with abundant public data

## Spec References
- `equity-benchmark` — all 6 requirements, 15 scenarios

## Risks
- Stock prediction is the most competitive ML domain — we're competing against billion-dollar quant firms
- Overfitting on historical data is the #1 failure mode — a model that backtests well can fail live
- Transaction costs, slippage, and spread erode thin edges
- Le-WM adaptation from pixels to 1D time-series is non-trivial
- Cross-language bridge (Python Le-WM → TypeScript AWM) adds complexity
- Survivorship bias in ticker universe (companies that exist today aren't representative of 5 years ago)
- Capital is at risk if we proceed to live trading

## Mitigations
- Strict temporal train/val/test splits (no future leakage)
- Simulated transaction costs in Sharpe/drawdown calculations
- Paper trading phase before any real capital deployment
- AWM confidence scores drive position sizing (low confidence = skip or tiny size)
- Multiple metrics including risk-adjusted (Sharpe, max drawdown) not just accuracy

## Approach
Three-phase implementation across two languages:
1. **Phase 1 (Python)**: Market data pipeline + feature engineering
2. **Phase 2 (Python)**: Le-WM time-series adaptation + standalone baseline
3. **Phase 3 (TypeScript)**: AWM equity store + raw-feature baseline (parallel with Phase 2)
4. **Phase 4 (Both)**: Integration bridge + benchmark harness + report generation
5. **Phase 5**: Run benchmark, validate results, analyse findings
