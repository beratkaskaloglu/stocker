"""
backtest/engine.py
Walk-forward backtest motoru.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Generator, Protocol

import numpy as np
import pandas as pd

from backtest.metrics import (
    annualized_return,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
    win_rate,
)


@dataclass
class BacktestResult:
    market: str
    start: str
    end: str
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    win_rate: float
    n_trades: int
    equity_curve: pd.Series
    trades: pd.DataFrame = field(default_factory=lambda: pd.DataFrame())


@dataclass
class Position:
    symbol: str
    direction: int          # +1 long, -1 short
    entry_price: float
    size: float             # number of shares/units
    entry_date: pd.Timestamp
    stop_loss: float | None = None
    take_profit: float | None = None


@dataclass
class Portfolio:
    cash: float
    positions: dict[str, Position] = field(default_factory=dict)

    @property
    def value(self) -> float:
        return self.cash

    def total_value(self, prices: dict[str, float]) -> float:
        val = self.cash
        for sym, pos in self.positions.items():
            price = prices.get(sym, pos.entry_price)
            val += pos.size * price * pos.direction
        return val


class Strategy(Protocol):
    """Callback protocol the engine uses for predictions.

    Implementations wrap the full pipeline:
    FeatureAggregator → MetaLearner → SignalGenerator (→ optional RL sizing).
    """

    def on_train(self, train_data: pd.DataFrame) -> None:
        """Called at the start of each walk-forward fold with training data."""
        ...

    def predict(self, row: pd.Series) -> dict:
        """Return prediction dict for a single bar.

        Expected keys:
            symbol:     str
            direction:  int   (-1, 0, +1)
            confidence: float (0-1)
            size:       float (fraction of capital, 0-1)
            stop_loss:  float | None  (absolute price)
            take_profit: float | None (absolute price)
        """
        ...


# ---------------------------------------------------------------------------
# Walk-forward split generator
# ---------------------------------------------------------------------------

WFSplit = tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]


def walk_forward_splits(
    dates: pd.DatetimeIndex,
    window: int = 252,
    step: int = 21,
    min_folds: int = 5,
) -> Generator[WFSplit, None, None]:
    """Yield (train_start, train_end, val_start, val_end) tuples.

    Args:
        dates:  Sorted DatetimeIndex of available trading days.
        window: Number of trading days in the training window.
        step:   Number of trading days in the validation (out-of-sample) window.
        min_folds: Minimum folds required; raises if not enough data.
    """
    n = len(dates)
    if n < window + step:
        raise ValueError(
            f"Not enough data for walk-forward: need {window + step} bars, got {n}"
        )

    folds: list[WFSplit] = []
    start = 0
    while start + window + step <= n:
        train_start = dates[start]
        train_end = dates[start + window - 1]
        val_start = dates[start + window]
        val_end = dates[min(start + window + step - 1, n - 1)]
        folds.append((train_start, train_end, val_start, val_end))
        start += step

    if len(folds) < min_folds:
        raise ValueError(
            f"Only {len(folds)} folds available, need at least {min_folds}. "
            f"Provide more data or reduce window/step."
        )

    yield from folds


# ---------------------------------------------------------------------------
# Backtest Engine
# ---------------------------------------------------------------------------

class BacktestEngine:
    """Walk-forward backtest engine with realistic trade execution."""

    def __init__(
        self,
        initial_capital: float = 100_000,
        trading_cost: float = 0.001,
        stop_loss_pct: float = 0.02,
        take_profit_pct: float = 0.04,
        max_position_pct: float = 0.10,
    ):
        self.initial_capital = initial_capital
        self.trading_cost = trading_cost
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.max_position_pct = max_position_pct

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(
        self,
        market: str,
        ohlcv: pd.DataFrame,
        strategy: Strategy,
        window: int = 252,
        step: int = 21,
    ) -> BacktestResult:
        """Run a walk-forward backtest.

        Args:
            market:   'US' or 'BIST'.
            ohlcv:    DataFrame with DatetimeIndex and columns
                      [symbol, open, high, low, close, volume].
                      Multi-symbol data supported.
            strategy: Object implementing the Strategy protocol.
            window:   Training window in trading days.
            step:     Validation window in trading days.
        """
        ohlcv = ohlcv.sort_index()
        dates = ohlcv.index.unique().sort_values()

        portfolio = Portfolio(cash=self.initial_capital)
        equity_records: list[tuple[pd.Timestamp, float]] = []
        trade_records: list[dict] = []

        for train_start, train_end, val_start, val_end in walk_forward_splits(
            dates, window=window, step=step
        ):
            # --- training phase ---
            train_mask = (ohlcv.index >= train_start) & (ohlcv.index <= train_end)
            strategy.on_train(ohlcv.loc[train_mask])

            # --- validation (out-of-sample) phase ---
            val_mask = (ohlcv.index >= val_start) & (ohlcv.index <= val_end)
            val_data = ohlcv.loc[val_mask]

            for dt, group in val_data.groupby(val_data.index):
                current_prices: dict[str, float] = {}
                highs: dict[str, float] = {}
                lows: dict[str, float] = {}

                for _, row in group.iterrows():
                    sym = row["symbol"]
                    current_prices[sym] = row["close"]
                    highs[sym] = row["high"]
                    lows[sym] = row["low"]

                # 1. Check stop-loss / take-profit on existing positions
                closed = self._check_exits(portfolio, highs, lows, dt, trade_records)

                # 2. Get predictions for each symbol in this bar
                for _, row in group.iterrows():
                    signal = strategy.predict(row)
                    if signal["direction"] == 0:
                        continue
                    self._apply_trade(
                        portfolio,
                        signal,
                        current_prices,
                        dt,
                        trade_records,
                    )

                # 3. Record equity
                equity_records.append(
                    (dt, portfolio.total_value(current_prices))
                )

        # --- build results ---
        if not equity_records:
            raise ValueError("No equity records produced — check data coverage.")

        equity = pd.Series(
            {dt: val for dt, val in equity_records},
            name="equity",
        ).sort_index()

        daily_returns = equity.pct_change().dropna()
        trades_df = pd.DataFrame(trade_records) if trade_records else pd.DataFrame()

        total_ret = (equity.iloc[-1] / self.initial_capital) - 1.0
        n_days = max((equity.index[-1] - equity.index[0]).days, 1)

        return BacktestResult(
            market=market,
            start=str(equity.index[0].date()),
            end=str(equity.index[-1].date()),
            total_return=total_ret,
            annualized_return=annualized_return(total_ret, n_days),
            sharpe_ratio=sharpe_ratio(daily_returns),
            sortino_ratio=sortino_ratio(daily_returns),
            max_drawdown=max_drawdown(equity),
            win_rate=win_rate(trades_df) if len(trades_df) > 0 else 0.0,
            n_trades=len(trades_df),
            equity_curve=equity,
            trades=trades_df,
        )

    # ------------------------------------------------------------------
    # Trade execution
    # ------------------------------------------------------------------

    def _apply_trade(
        self,
        portfolio: Portfolio,
        signal: dict,
        prices: dict[str, float],
        dt: pd.Timestamp,
        trade_records: list[dict],
    ) -> None:
        """Open, close, or flip a position based on signal."""
        sym = signal["symbol"]
        direction = signal["direction"]
        price = prices[sym]
        confidence = signal.get("confidence", 1.0)
        size_frac = signal.get("size", self.max_position_pct)

        # Close existing position if direction conflicts
        if sym in portfolio.positions:
            pos = portfolio.positions[sym]
            if pos.direction != direction:
                self._close_position(portfolio, sym, price, dt, trade_records, "flip")
            else:
                return  # already aligned, skip

        # Open new position
        alloc = portfolio.cash * min(size_frac, self.max_position_pct) * confidence
        if alloc <= 0 or price <= 0:
            return

        cost = alloc * self.trading_cost
        net_alloc = alloc - cost
        n_shares = net_alloc / price

        sl = price * (1 - self.stop_loss_pct) if direction == 1 else price * (1 + self.stop_loss_pct)
        tp = price * (1 + self.take_profit_pct) if direction == 1 else price * (1 - self.take_profit_pct)

        stop_loss = signal.get("stop_loss") or sl
        take_profit = signal.get("take_profit") or tp

        portfolio.cash -= alloc
        portfolio.positions[sym] = Position(
            symbol=sym,
            direction=direction,
            entry_price=price,
            size=n_shares,
            entry_date=dt,
            stop_loss=stop_loss,
            take_profit=take_profit,
        )

    def _close_position(
        self,
        portfolio: Portfolio,
        symbol: str,
        exit_price: float,
        dt: pd.Timestamp,
        trade_records: list[dict],
        reason: str = "signal",
    ) -> float:
        """Close a position and record the trade. Returns PnL."""
        pos = portfolio.positions.pop(symbol)
        exit_cost = pos.size * exit_price * self.trading_cost
        entry_cost = pos.size * pos.entry_price * self.trading_cost

        # PnL: price diff * shares * direction - round-trip commissions
        pnl = (
            (exit_price - pos.entry_price) * pos.size * pos.direction
            - entry_cost
            - exit_cost
        )

        # Settle cash
        if pos.direction == 1:  # long: sell shares
            portfolio.cash += pos.size * exit_price - exit_cost
        else:  # short: return borrowed shares, settle difference
            # At entry we received entry_price * size (margin).
            # At exit we buy back at exit_price * size.
            portfolio.cash += pos.size * (2 * pos.entry_price - exit_price) - exit_cost

        trade_records.append(
            {
                "symbol": symbol,
                "direction": pos.direction,
                "entry_price": pos.entry_price,
                "exit_price": exit_price,
                "entry_date": pos.entry_date,
                "exit_date": dt,
                "size": pos.size,
                "pnl": pnl,
                "reason": reason,
            }
        )
        return pnl

    def _check_exits(
        self,
        portfolio: Portfolio,
        highs: dict[str, float],
        lows: dict[str, float],
        dt: pd.Timestamp,
        trade_records: list[dict],
    ) -> list[str]:
        """Check stop-loss and take-profit for all open positions."""
        to_close: list[tuple[str, float, str]] = []

        for sym, pos in list(portfolio.positions.items()):
            high = highs.get(sym)
            low = lows.get(sym)
            if high is None or low is None:
                continue

            if pos.direction == 1:  # long
                if pos.stop_loss is not None and low <= pos.stop_loss:
                    to_close.append((sym, pos.stop_loss, "stop_loss"))
                elif pos.take_profit is not None and high >= pos.take_profit:
                    to_close.append((sym, pos.take_profit, "take_profit"))
            else:  # short
                if pos.stop_loss is not None and high >= pos.stop_loss:
                    to_close.append((sym, pos.stop_loss, "stop_loss"))
                elif pos.take_profit is not None and low <= pos.take_profit:
                    to_close.append((sym, pos.take_profit, "take_profit"))

        closed_syms = []
        for sym, exit_price, reason in to_close:
            self._close_position(portfolio, sym, exit_price, dt, trade_records, reason)
            closed_syms.append(sym)

        return closed_syms

    # ------------------------------------------------------------------
    # Benchmark comparison
    # ------------------------------------------------------------------

    @staticmethod
    def compare_benchmark(
        result: BacktestResult,
        benchmark_returns: pd.Series,
    ) -> dict:
        """Compare backtest result against a benchmark.

        Args:
            result:            BacktestResult from run().
            benchmark_returns: Daily returns of benchmark (SPY / XU100).

        Returns:
            dict with alpha, beta, information_ratio, tracking_error,
            benchmark_sharpe, strategy_sharpe.
        """
        strategy_returns = result.equity_curve.pct_change().dropna()

        # Align dates
        common = strategy_returns.index.intersection(benchmark_returns.index)
        if len(common) < 30:
            return {"error": "Not enough overlapping dates for comparison"}

        sr = strategy_returns.loc[common]
        br = benchmark_returns.loc[common]

        # Beta & Alpha (CAPM)
        cov = np.cov(sr.values, br.values)
        beta = cov[0, 1] / cov[1, 1] if cov[1, 1] > 0 else 0.0
        alpha = (sr.mean() - beta * br.mean()) * 252

        # Tracking error & information ratio
        excess = sr - br
        tracking_error = excess.std() * np.sqrt(252)
        information_ratio = (
            (excess.mean() * 252) / tracking_error if tracking_error > 0 else 0.0
        )

        return {
            "alpha": alpha,
            "beta": beta,
            "information_ratio": information_ratio,
            "tracking_error": tracking_error,
            "benchmark_sharpe": sharpe_ratio(br),
            "strategy_sharpe": sharpe_ratio(sr),
        }
