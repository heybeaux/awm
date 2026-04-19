"""Ticker universe for AWM equity benchmark.

Default 50-instrument universe spanning mega-cap, sector ETFs, broad ETFs,
cross-asset, and mid-cap growth. All tickers trade on NYSE/NASDAQ.
"""
from __future__ import annotations

DEFAULT_UNIVERSE: list[str] = [
    # Mega-cap (10)
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL",
    "META", "TSLA", "BRK-B", "JPM", "V",
    # Sector ETFs (11)
    "XLF", "XLE", "XLK", "XLV", "XLI",
    "XLC", "XLY", "XLP", "XLU", "XLRE", "XLB",
    # Broad market / index ETFs (5)
    "SPY", "QQQ", "IWM", "DIA", "VTI",
    # Cross-asset (5)
    "GLD", "TLT", "HYG", "UUP", "USO",
    # Mid-cap growth (10)
    "CRWD", "DDOG", "NET", "SNOW", "PLTR",
    "MDB", "ZS", "COIN", "SQ", "SHOP",
]


def get_universe(config: dict | None = None) -> list[str]:
    """Return the ticker universe.

    If `config['universe']['tickers']` is a non-empty list, it overrides
    the default. Otherwise returns `DEFAULT_UNIVERSE`.
    """
    if config:
        override = (config.get("universe") or {}).get("tickers")
        if override:
            return list(override)
    return list(DEFAULT_UNIVERSE)


if __name__ == "__main__":
    u = get_universe()
    print(f"Default universe: {len(u)} tickers")
    print(", ".join(u))
