# region imports
from AlgorithmImports import *
import csv
import os
from collections import defaultdict
from io import StringIO
# endregion


class AWMFusion(QCAlgorithm):
    """AWM + Le-WM Fusion Strategy — QuantConnect Backtest.

    Pre-computed fusion signals are loaded from signals.csv.
    Strategy: go long when fusion_p > entry_threshold, flat otherwise.
    Position size proportional to confidence: size = (fusion_p - 0.5) * 2.
    Hold for 5 trading days then re-evaluate.

    Walk-forward: only uses test-split signals (out-of-sample).
    IBKR brokerage model for realistic commissions + slippage.
    """

    def initialize(self):
        # ── Date range (test split starts ~mid 2024) ──
        self.set_start_date(2024, 6, 1)
        self.set_end_date(2026, 4, 17)

        # ── Capital ──
        self.set_cash(95000)  # ~TFSA room

        # ── Brokerage: Interactive Brokers ──
        self.set_brokerage_model(BrokerageName.INTERACTIVE_BROKERS_BROKERAGE, AccountType.MARGIN)

        # ── Parameters (sweepable) ──
        self.entry_threshold = float(self.get_parameter("entry_threshold", 0.55))
        self.max_position_pct = float(self.get_parameter("max_position_pct", 0.20))
        self.hold_days = int(self.get_parameter("hold_days", 5))
        self.max_positions = int(self.get_parameter("max_positions", 8))

        # ── Universe ──
        # Exclude bond/commodity ETFs (trivial predictions)
        self.excluded = {"HYG", "TLT", "GLD", "UUP", "USO"}

        # ── Load signals ──
        self.signals = defaultdict(dict)  # {date_str: {ticker: {fusion_p, regime, ...}}}
        self._load_signals()

        # ── Add equities ──
        self.symbol_map = {}
        for ticker in self.all_tickers:
            if ticker in self.excluded:
                continue
            try:
                sym = self.add_equity(ticker, Resolution.DAILY,
                                      market=Market.USA,
                                      fill_forward=True,
                                      leverage=1.0).symbol
                self.symbol_map[ticker] = sym
            except Exception as e:
                self.debug(f"Could not add {ticker}: {e}")

        # ── Position tracking ──
        self.entry_dates = {}  # symbol -> entry date
        self.trade_log = []

        # ── Risk management ──
        self.drawdown_pause = False
        self.peak_equity = self.portfolio.total_portfolio_value
        self.max_drawdown_pct = 0.10  # 10% drawdown → pause
        self.kill_switch_pct = 0.15   # 15% drawdown → kill

        # ── Schedule daily signal check at market open + 30 min ──
        self.schedule.on(
            self.date_rules.every_day(),
            self.time_rules.after_market_open("SPY", 30),
            self.check_signals
        )

        # ── Warm up ──
        self.set_warm_up(timedelta(days=5))

    def _load_signals(self):
        """Load pre-computed fusion signals from CSV stored in ObjectStore or local file."""
        # Try ObjectStore first, then local file
        signals_path = os.path.join(os.path.dirname(__file__), "..", "quantconnect", "signals.csv")

        if not os.path.exists(signals_path):
            # Fall back to algo directory
            signals_path = os.path.join(os.path.dirname(__file__), "signals.csv")

        tickers_seen = set()
        count = 0

        try:
            with open(signals_path, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row["split"] != "test":
                        continue
                    if row["fusion_p"] in (None, "", "None"):
                        continue

                    date_str = row["date"]
                    ticker = row["ticker"]
                    tickers_seen.add(ticker)

                    self.signals[date_str][ticker] = {
                        "fusion_p": float(row["fusion_p"]),
                        "lewm_p": float(row["lewm_p"]),
                        "awm_p": float(row["awm_p"]) if row.get("awm_p") not in (None, "", "None") else 0.5,
                        "regime": row["regime"],
                        "direction_5d": float(row["direction_5d"]) if row.get("direction_5d") not in (None, "", "None") else None,
                    }
                    count += 1
        except FileNotFoundError:
            self.debug(f"WARNING: signals.csv not found at {signals_path}")
            self.debug("Strategy will not generate any trades.")

        self.all_tickers = tickers_seen
        self.debug(f"Loaded {count} test signals for {len(tickers_seen)} tickers, "
                   f"{len(self.signals)} unique dates")

    def check_signals(self):
        """Daily signal evaluation — the core trading logic."""
        if self.is_warming_up:
            return

        # ── Risk checks ──
        equity = self.portfolio.total_portfolio_value
        self.peak_equity = max(self.peak_equity, equity)
        drawdown = (self.peak_equity - equity) / self.peak_equity

        if drawdown >= self.kill_switch_pct:
            self.debug(f"KILL SWITCH: {drawdown:.1%} drawdown. Liquidating all.")
            self.liquidate()
            self.drawdown_pause = True
            return

        if drawdown >= self.max_drawdown_pct:
            if not self.drawdown_pause:
                self.debug(f"DRAWDOWN PAUSE: {drawdown:.1%}. No new entries.")
                self.drawdown_pause = True
        elif self.drawdown_pause and drawdown < self.max_drawdown_pct * 0.5:
            self.debug(f"Drawdown recovered to {drawdown:.1%}. Resuming.")
            self.drawdown_pause = False

        # ── Exit stale positions ──
        for ticker, sym in list(self.symbol_map.items()):
            if self.portfolio[sym].invested and ticker in self.entry_dates:
                days_held = (self.time - self.entry_dates[ticker]).days
                if days_held >= self.hold_days:
                    self.liquidate(sym, tag=f"hold_exit_{days_held}d")
                    del self.entry_dates[ticker]

        # ── Check today's signals ──
        date_str = self.time.strftime("%Y-%m-%d")
        today_signals = self.signals.get(date_str, {})

        if not today_signals:
            return

        # ── Rank candidates by fusion_p ──
        candidates = []
        for ticker, sig in today_signals.items():
            if ticker in self.excluded:
                continue
            if ticker not in self.symbol_map:
                continue
            if sig["fusion_p"] <= self.entry_threshold:
                continue
            if self.portfolio[self.symbol_map[ticker]].invested:
                continue
            candidates.append((ticker, sig))

        # Sort by fusion_p descending — best signals first
        candidates.sort(key=lambda x: x[1]["fusion_p"], reverse=True)

        # ── Count current positions ──
        current_positions = sum(
            1 for sym in self.symbol_map.values()
            if self.portfolio[sym].invested
        )

        # ── Enter new positions ──
        slots_available = self.max_positions - current_positions
        if self.drawdown_pause:
            slots_available = 0

        for ticker, sig in candidates[:slots_available]:
            sym = self.symbol_map[ticker]
            if not self.securities[sym].has_data:
                continue

            # Position size: proportional to confidence, capped at max_position_pct
            confidence = (sig["fusion_p"] - 0.5) * 2.0  # 0 to 1
            position_pct = min(confidence * self.max_position_pct, self.max_position_pct)

            # Use limit order at current price + 0.1% (avoid chasing)
            price = self.securities[sym].price
            if price <= 0:
                continue

            limit_price = round(price * 1.001, 2)
            qty = int((equity * position_pct) / price)
            if qty <= 0:
                continue

            ticket = self.limit_order(sym, qty, limit_price,
                                      tag=f"fusion={sig['fusion_p']:.3f}|regime={sig['regime']}")

            self.entry_dates[ticker] = self.time
            self.trade_log.append({
                "date": date_str,
                "ticker": ticker,
                "fusion_p": sig["fusion_p"],
                "regime": sig["regime"],
                "qty": qty,
                "price": price,
                "direction_5d": sig.get("direction_5d"),
            })

    def on_order_event(self, order_event: OrderEvent):
        """Track fills for analysis."""
        if order_event.status == OrderStatus.FILLED:
            self.debug(f"FILLED: {order_event.symbol} qty={order_event.fill_quantity} "
                       f"@ {order_event.fill_price:.2f}")
        elif order_event.status == OrderStatus.CANCELED:
            # Limit order expired — that's fine, signal wasn't actionable
            pass

    def on_end_of_algorithm(self):
        """Final stats."""
        equity = self.portfolio.total_portfolio_value
        total_return = (equity / 95000) - 1
        self.debug(f"\n{'='*50}")
        self.debug(f"FINAL EQUITY: ${equity:,.2f}")
        self.debug(f"TOTAL RETURN: {total_return:.2%}")
        self.debug(f"TOTAL TRADES: {len(self.trade_log)}")
        self.debug(f"ENTRY THRESHOLD: {self.entry_threshold}")
        self.debug(f"MAX POSITION: {self.max_position_pct:.0%}")
        self.debug(f"MAX POSITIONS: {self.max_positions}")
        self.debug(f"HOLD DAYS: {self.hold_days}")

        if self.trade_log:
            regimes = defaultdict(int)
            for t in self.trade_log:
                regimes[t["regime"]] += 1
            self.debug(f"TRADES BY REGIME: {dict(regimes)}")

        self.debug(f"{'='*50}")
