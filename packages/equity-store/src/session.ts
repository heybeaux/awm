/**
 * Session / Profile Mapping
 *
 * AWM's `profileSlug` partitions belief and bandit state. For the equity
 * benchmark we use the ticker symbol directly so AWM learns that TSLA
 * behaves differently from SPY.
 *
 * `tickerToSector` is offered for callers who want to also accumulate
 * cross-ticker beliefs at the sector level (not used by the default flow).
 */

export function tickerToProfile(ticker: string): string {
  return ticker.trim().toUpperCase();
}

const SECTOR_MAP: Record<string, string> = {
  // Mega-cap tech / consumer
  AAPL: 'tech', MSFT: 'tech', NVDA: 'tech', GOOGL: 'tech', META: 'tech',
  AMZN: 'consumer', TSLA: 'consumer',
  // Financials
  'BRK-B': 'financials', JPM: 'financials', V: 'financials',
  // Sector ETFs
  XLF: 'financials', XLE: 'energy', XLK: 'tech', XLV: 'healthcare',
  XLI: 'industrials', XLC: 'comms', XLY: 'consumer', XLP: 'consumer',
  XLU: 'utilities', XLRE: 'real-estate', XLB: 'materials',
  // Broad market
  SPY: 'broad', QQQ: 'broad', IWM: 'broad', DIA: 'broad', VTI: 'broad',
  // Cross-asset
  GLD: 'commodities', TLT: 'bonds', HYG: 'bonds', UUP: 'fx', USO: 'commodities',
  // Mid-cap growth
  CRWD: 'tech', DDOG: 'tech', NET: 'tech', SNOW: 'tech', PLTR: 'tech',
  MDB: 'tech', ZS: 'tech', COIN: 'financials', SQ: 'financials', SHOP: 'tech',
};

export function tickerToSector(ticker: string): string | undefined {
  return SECTOR_MAP[tickerToProfile(ticker)];
}
