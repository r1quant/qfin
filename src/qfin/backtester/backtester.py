"""
Backtester class to manage and analyze trading strategies.

It provides a way to run backtesting on historical data, perform trades,
and calculate profit/loss.
"""

from __future__ import annotations

from typing import Generator, List

import numpy as np
import pandas as pd

from .plot import plot_basic, plot_thumbnail
from .stats import stats


class Trade:
    """Represents a single trade with entry and exit prices, commissions, and P&L."""

    def __init__(self, state: BrokerState | None = None):
        self.state: BrokerState = state
        self.is_long: bool | None = None
        self.leverage: float | None = None
        self.entry_value: float | None = None
        self.entry_price: float | None = None
        self.entry_bar: int | None = None
        self.entry_time: str | None = None
        self.entry_commission: float = 0.0
        self.exit_value: float | None = None
        self.exit_price: float | None = None
        self.exit_bar: int | None = None
        self.exit_time: str | None = None
        self.exit_commission: float = 0.0
        self.trailing_high: float | None = None
        self.trailing_low: float | None = None
        self.trailing_stop_price: float | None = None
        self.breakeven_triggered: bool = False
        self.exit_reason: str = "manual"

    @property
    def pl_value(self) -> float:
        """Trade profit (positive) or loss (negative) in cash units."""
        price = self.exit_price or self.state.last_price

        if self.is_long:
            perc = price / self.entry_price
        else:
            perc = self.entry_price / price

        return self.entry_value * (perc - 1)

    @property
    def pl_pct(self) -> float:
        """Trade profit (positive) or loss (negative) as a decimal fraction."""
        price = self.exit_price or self.state.last_price
        if self.is_long:
            perc = price / self.entry_price
        else:
            perc = self.entry_price / price

        return perc - 1

    @property
    def commissions(self) -> float:
        """Total commissions (entry + exit) spent on the trade."""
        return self.entry_commission + self.exit_commission


class Params:
    """Configuration parameters for the backtester.

    Attributes:
        dataset: Historical price data with a 'close' column and DatetimeIndex.
        initial_balance: Starting cash balance.
        commission: Commission rate as a decimal (e.g., 0.001 = 0.1%).
        default_entry_value: If <= 1, treated as percentage of balance; if > 1, treated as cash amount.
        default_entry_value_max: Maximum position size in cash units.
        close_column: Name of the column containing close prices.
    """

    def __init__(
        self,
        dataset: pd.DataFrame,
        initial_balance: float = 10000,
        commission: float = 0.01,
        default_entry_value: float = 1,
        default_entry_value_max: float = 1000000.0,
        trailing_enabled: bool = False,
        trailing_distance_pct: float = 0,
        trailing_activation_pct: float = 0,
        trailing_min_step_pct: float = 0,
        takeprofit_pct: float = 0,
        stoploss_pct: float = 0,
        breakeven_pct: float = 0,
    ) -> None:
        for name, val in [
            ("trailing_distance_pct", trailing_distance_pct),
            ("trailing_activation_pct", trailing_activation_pct),
            ("trailing_min_step_pct", trailing_min_step_pct),
            ("takeprofit_pct", takeprofit_pct),
            ("stoploss_pct", stoploss_pct),
            ("breakeven_pct", breakeven_pct),
        ]:
            if not (0 <= val <= 100):
                raise ValueError(f"{name} must be between 0 and 100, got {val}")

        self.dataset = dataset.copy()
        self.initial_balance = initial_balance
        self.commission = commission
        self.default_entry_value = default_entry_value
        self.default_entry_value_max = default_entry_value_max
        self.close_column = "close"
        self.trailing_enabled = trailing_enabled
        self.trailing_distance_pct = trailing_distance_pct
        self.trailing_activation_pct = trailing_activation_pct
        self.trailing_min_step_pct = trailing_min_step_pct
        self.takeprofit_pct = takeprofit_pct
        self.stoploss_pct = stoploss_pct
        self.breakeven_pct = breakeven_pct


class BrokerState:
    """Current state of the broker at a given bar.

    Attributes:
        data: Slice of the dataset around the current bar (up to _nbars lookback).
        current_bar: Index of the current bar being processed.
        is_last_bar: Whether this is the final bar in the dataset.
        last_price: Close price of the current bar.
        total_bar: Total number of bars in the dataset.
    """

    def __init__(self, current_bar: int, is_last_bar: bool, last_price: float, total_bar: int):
        self.data: pd.DataFrame = pd.DataFrame()
        self.current_bar = current_bar
        self.is_last_bar = is_last_bar
        self.last_price = last_price
        self.total_bar = total_bar
        self._nbars = 10


class BrokerAccount:
    """Manages balance, equity, open/closed trades, and per-bar history arrays.

    Operates in netting mode by default: opening a new position closes any existing one.
    """

    def __init__(self, broker: Broker):
        params = broker.params
        self.broker: Broker = broker
        self.params: Params = params
        self.balance: float = params.initial_balance
        self.equity: float = params.initial_balance
        self.hedging: bool = False
        self.netting: bool = True
        self.opened_trades: List[Trade] = []
        self.closed_trades: List[Trade] = []
        self.history_balance: np.ndarray = np.tile(params.initial_balance, len(params.dataset))
        self.history_equity: np.ndarray = np.tile(params.initial_balance, len(params.dataset))
        self.history_commission: np.ndarray = np.tile(0, len(params.dataset))
        self.commission_spent: float = 0

    def _check_risk_management(self) -> None:
        """Check and apply risk management rules (SL, TP, breakeven, trailing) to open trades."""
        params = self.params
        price = self.broker.state.last_price

        for trade in list(self.opened_trades):
            # Update trailing high/low
            if trade.trailing_high is None:
                trade.trailing_high = price
            else:
                trade.trailing_high = max(trade.trailing_high, price)

            if trade.trailing_low is None:
                trade.trailing_low = price
            else:
                trade.trailing_low = min(trade.trailing_low, price)

            # Compute profit percentage
            if trade.is_long:
                profit_pct = ((price / trade.entry_price) - 1) * 100
            else:
                profit_pct = ((trade.entry_price / price) - 1) * 100

            # Stop Loss
            if params.stoploss_pct > 0 and profit_pct <= -params.stoploss_pct:
                trade.exit_reason = "stoploss"
                self._BrokerAccount__close(trade)
                continue

            # Take Profit
            if params.takeprofit_pct > 0 and profit_pct >= params.takeprofit_pct:
                trade.exit_reason = "takeprofit"
                self._BrokerAccount__close(trade)
                continue

            # Breakeven: arm when profit reaches threshold
            if params.breakeven_pct > 0 and not trade.breakeven_triggered:
                if profit_pct >= params.breakeven_pct:
                    trade.breakeven_triggered = True

            # Breakeven stop level
            breakeven_stop = None
            if trade.breakeven_triggered:
                breakeven_stop = trade.entry_price

            # Trailing stop
            trailing_stop = None
            if params.trailing_enabled and params.trailing_distance_pct > 0:
                if profit_pct >= params.trailing_activation_pct:
                    if trade.is_long:
                        candidate = trade.trailing_high * (1 - params.trailing_distance_pct / 100)
                    else:
                        candidate = trade.trailing_low * (1 + params.trailing_distance_pct / 100)

                    # Apply min step filter: only update if moved enough
                    if trade.trailing_stop_price is not None:
                        if trade.is_long:
                            if params.trailing_min_step_pct > 0:
                                min_move = trade.trailing_stop_price * params.trailing_min_step_pct / 100
                                if candidate - trade.trailing_stop_price < min_move:
                                    candidate = trade.trailing_stop_price
                            # Stop never moves backward
                            candidate = max(candidate, trade.trailing_stop_price)
                        else:
                            if params.trailing_min_step_pct > 0:
                                min_move = trade.trailing_stop_price * params.trailing_min_step_pct / 100
                                if trade.trailing_stop_price - candidate < min_move:
                                    candidate = trade.trailing_stop_price
                            # Stop never moves backward (for shorts, lower is better)
                            candidate = min(candidate, trade.trailing_stop_price)

                    trade.trailing_stop_price = candidate
                    trailing_stop = candidate

            # Determine effective stop and check breach
            effective_stop = None
            dominant_reason = None

            if breakeven_stop is not None and trailing_stop is not None:
                if trade.is_long:
                    if trailing_stop >= breakeven_stop:
                        effective_stop = trailing_stop
                        dominant_reason = "trailing"
                    else:
                        effective_stop = breakeven_stop
                        dominant_reason = "breakeven"
                else:
                    if trailing_stop <= breakeven_stop:
                        effective_stop = trailing_stop
                        dominant_reason = "trailing"
                    else:
                        effective_stop = breakeven_stop
                        dominant_reason = "breakeven"
            elif breakeven_stop is not None:
                effective_stop = breakeven_stop
                dominant_reason = "breakeven"
            elif trailing_stop is not None:
                effective_stop = trailing_stop
                dominant_reason = "trailing"

            if effective_stop is not None:
                breached = False
                if trade.is_long and price <= effective_stop:
                    breached = True
                elif not trade.is_long and price >= effective_stop:
                    breached = True

                if breached:
                    trade.exit_reason = dominant_reason
                    self._BrokerAccount__close(trade)

    def refresh_values(self) -> None:
        """Recalculate equity, commissions, and update history arrays for the current bar."""
        self._check_risk_management()
        self.commission_spent = sum(trade.commissions for trade in self.opened_trades)
        self.commission_spent += sum(trade.commissions for trade in self.closed_trades)
        self.equity = round(self.balance + sum(trade.pl_value - trade.commissions for trade in self.opened_trades), 2)
        self.history_balance[self.broker.state.current_bar] = self.balance
        self.history_equity[self.broker.state.current_bar] = self.equity
        self.history_commission[self.broker.state.current_bar] = round(self.commission_spent, 2)

    def __open(self, is_long: bool = False, value: float | None = None, price: float | None = None, leverage: float | None = 1):
        """Open a new trade."""
        if self.netting:
            self.close()

        if self.broker.state.is_last_bar:
            return

        # calculate entry value based on parameters
        if value:
            entry_value = value
        elif self.params.default_entry_value <= 1:
            entry_value = min(self.balance * self.params.default_entry_value, self.params.default_entry_value_max)
        else:
            entry_value = min(self.params.default_entry_value, self.params.default_entry_value_max)

        if leverage is not None:
            entry_value = entry_value * leverage

        opened_trade = Trade(self.broker.state)
        opened_trade.leverage = leverage
        opened_trade.entry_commission = entry_value * self.params.commission
        opened_trade.entry_value = entry_value - opened_trade.entry_commission
        opened_trade.entry_price = price or self.broker.state.last_price
        opened_trade.entry_bar = self.broker.state.current_bar
        opened_trade.entry_time = self.params.dataset.iloc[opened_trade.entry_bar].name
        opened_trade.is_long = is_long
        self.opened_trades.append(opened_trade)

    def __close(self, trade: Trade, exit_price: float | None = None) -> None:
        """Close an existing trade and update the balance."""
        self.opened_trades.remove(trade)
        closed_trade = trade
        closed_trade.exit_bar = self.broker.state.current_bar
        closed_trade.exit_price = exit_price or self.broker.state.last_price
        closed_trade.exit_commission = (trade.entry_value + trade.pl_value) * self.params.commission
        closed_trade.exit_value = trade.pl_value + trade.entry_value
        closed_trade.exit_time = self.params.dataset.iloc[closed_trade.exit_bar].name
        self.closed_trades.append(closed_trade)
        self.balance += round(closed_trade.pl_value - closed_trade.exit_commission, 2)

    def close(self) -> None:
        """Close all open trades."""
        for trade in list(self.opened_trades):
            self.__close(trade)

    def buy(self, leverage: float | None = 1) -> None:
        """Open a long position."""
        self.__open(is_long=True, leverage=leverage)

    def sell(self, leverage: float | None = 1) -> None:
        """Open a short position."""
        self.__open(is_long=False, leverage=leverage)


class Broker:
    """Controls bar iteration and delegates trading operations to BrokerAccount."""

    def __init__(self, params: Params):
        self.params = params
        self.state: BrokerState = BrokerState(
            current_bar=0,
            is_last_bar=False,
            last_price=0.0,
            total_bar=len(params.dataset),
        )
        self.account_main: BrokerAccount = BrokerAccount(self)

    def set_next_bar(self, index: int) -> None:
        """Advance to the given bar index and refresh account values."""
        _start = max(index - self.state._nbars, 0)
        _end = index + 1

        self.state.current_bar = index
        self.state.data = self.params.dataset.iloc[_start:_end]
        self.state.is_last_bar = index + 1 == self.state.total_bar
        self.state.last_price = self.state.data.iloc[-1][self.params.close_column]  # fmt: off
        self.refresh()

    def refresh(self) -> None:
        """Refresh account values for the current bar."""
        self.account_main.refresh_values()

    def buy(self, leverage: float | None = 1) -> None:
        """Open a long position."""
        self.account_main.buy(leverage=leverage)

    def sell(self, leverage: float | None = 1) -> None:
        """Open a short position."""
        self.account_main.sell(leverage=leverage)

    def close(self) -> None:
        """Close all open trades."""
        self.account_main.close()


class Backtester:
    """Main interface for running backtests on historical data.

    Usage:
        bt = Backtester(dataset=df, initial_balance=10000)
        for broker in bt.run():
            # implement strategy logic using broker.buy() / broker.sell() / broker.close()
            pass
        print(bt.stats())
    """

    def __init__(
        self,
        dataset: pd.DataFrame,
        initial_balance: float = 10000.0,
        commission: float = 0.001,
        default_entry_value: float = 1,
        default_entry_value_max: float = 20000,
        trailing_enabled: bool = False,
        trailing_distance_pct: float = 0,
        trailing_activation_pct: float = 0,
        trailing_min_step_pct: float = 0,
        takeprofit_pct: float = 0,
        stoploss_pct: float = 0,
        breakeven_pct: float = 0,
    ) -> None:
        self.params: Params = Params(
            dataset,
            initial_balance,
            commission,
            default_entry_value,
            default_entry_value_max,
            trailing_enabled=trailing_enabled,
            trailing_distance_pct=trailing_distance_pct,
            trailing_activation_pct=trailing_activation_pct,
            trailing_min_step_pct=trailing_min_step_pct,
            takeprofit_pct=takeprofit_pct,
            stoploss_pct=stoploss_pct,
            breakeven_pct=breakeven_pct,
        )

    def trades(self) -> pd.DataFrame:
        """Return a DataFrame of all closed trades with entry/exit details and P&L."""
        trades = self.broker.account_main.closed_trades
        return pd.DataFrame(
            {
                "is_long": [t.is_long for t in trades],
                "leverage": [t.leverage for t in trades],
                "entry_value": [t.entry_value for t in trades],
                "entry_price": [t.entry_price for t in trades],
                "entry_bar": [t.entry_bar for t in trades],
                "entry_commission": [t.entry_commission for t in trades],
                "entry_time": [t.entry_time for t in trades],
                "exit_value": [t.exit_value for t in trades],
                "exit_price": [t.exit_price for t in trades],
                "exit_commission": [t.exit_commission for t in trades],
                "exit_bar": [t.exit_bar for t in trades],
                "exit_time": [t.exit_time for t in trades],
                "pnl": [t.pl_value for t in trades],
                "return_pct": [t.pl_pct for t in trades],
                "exit_reason": [t.exit_reason for t in trades],
            }
        )

    def history(self) -> pd.DataFrame:
        """Return a DataFrame with per-bar close, balance, equity, commission, and position signals."""
        indexs = self.params.dataset.index
        data = {
            "close": self.params.dataset[self.params.close_column],
            "balance": self.broker.account_main.history_balance,
            "equity": self.broker.account_main.history_equity,
            "commission": self.broker.account_main.history_commission,
            "long": np.tile(False, len(indexs)),
            "short": np.tile(False, len(indexs)),
            "signal": np.tile(0, len(indexs)),
        }

        history = pd.DataFrame(data, index=indexs)

        long_index = history.columns.get_loc("long")
        short_index = history.columns.get_loc("short")
        signal_index = history.columns.get_loc("signal")

        for row in self.trades().itertuples():
            if row.is_long:
                history.iloc[row.entry_bar : row.exit_bar, long_index] = True
                history.iloc[row.entry_bar : row.exit_bar, signal_index] = 1
            else:
                history.iloc[row.entry_bar : row.exit_bar, short_index] = True
                history.iloc[row.entry_bar : row.exit_bar, signal_index] = -1

        # -- buy and hold
        balance_start = self.params.initial_balance
        units = balance_start / history.iloc[0]["close"]
        history["buy_hold"] = history["close"] * units

        return history

    def run(self) -> Generator[Broker, None, None]:
        """Run the backtesting process, yielding the Broker at each bar for strategy logic."""
        self.broker = Broker(self.params)
        total = len(self.params.dataset)
        current = 1

        while current < total:
            self.broker.set_next_bar(current)
            yield self.broker
            current += 1

        self.broker.refresh()

        should_exit_on_last_bar = True
        if should_exit_on_last_bar:
            self.broker.close()
            self.broker.refresh()

    def stats(self) -> pd.Series:
        """Compute and return performance statistics as a pd.Series."""
        return stats(self.history(), self.trades())

    def plot(self, title: str | None = "Backtest", w: int = 1024, h: int = 900, show_signals: bool = False):
        """Display an interactive Plotly chart with price, positions, and balance."""
        return plot_basic(history=self.history(), params=self.params, title=title, w=w, h=h, show_signals=show_signals)

    def thumbnail(self, title: str | None = None, w: int = 4, h: int = 1):
        """Display a compact matplotlib summary thumbnail."""
        return plot_thumbnail(history=self.history(), params=self.params, stats=self.stats(), title=title, w=w, h=h)
