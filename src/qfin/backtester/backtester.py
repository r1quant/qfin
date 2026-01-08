"""
Backtester class to manage and analyze trading strategies.

It provides a way to run backtesting on historical data, perform trades,
and calculate profit/loss.
"""

from typing import List

import numpy as np
import pandas as pd

from .plot import plot_basic, plot_thumbnail
from .stats import stats


class Trade:
    """Represents a trade with entry and exit prices."""

    def __init__(self, state=None):
        self.state: BrokerState = state
        self.is_long: bool = None
        self.leverage: float | None = None
        self.entry_value: float = None
        self.entry_price: float = None
        self.entry_bar: int = None
        self.entry_time: str = None
        self.entry_commission: float = 0.0
        self.exit_value: float = None
        self.exit_price: float = None
        self.exit_bar: int = None
        self.exit_time: str = None
        self.exit_commission: float = 0.0

    @property
    def pl_value(self):
        """Trade profit (positive) or loss (negative) in cash units."""
        price = self.exit_price or self.state.last_price

        if self.is_long:
            perc = price / self.entry_price
        else:
            perc = self.entry_price / price

        return self.entry_value * (perc - 1)

    @property
    def pl_pct(self):
        """Trade profit (positive) or loss (negative) in percent."""
        price = self.exit_price or self.broker.last_price
        if self.is_long:
            perc = price / self.entry_price
        else:
            perc = self.entry_price / price

        return perc - 1

    @property
    def commissions(self):
        """Commissions spent on the trade."""
        return self.entry_commission + self.exit_commission


class Params:
    """
    Configuration parameters for the backtester.

    These parameters control various aspects of the backtesting process.
    """

    def __init__(
        self,
        dataset: pd.DataFrame = None,
        initial_balance: float = 10000,
        commission: float = 0.01,  # Default commission rate
        default_entry_value: float = 1,  # Between 0.01 and 1 (percent)
        default_entry_value_max: float = 1000000.0,
    ) -> None:
        self.dataset = dataset.copy()
        self.initial_balance = initial_balance
        self.commission = commission
        self.default_entry_value = default_entry_value
        self.default_entry_value_max = default_entry_value_max
        self.close_column = "close"


class BrokerState:
    """
    Represents the current state of the broker.

    It tracks open trades, closed trades, and other relevant information.
    """

    def __init__(self, current_bar: int, is_last_bar: bool, last_price: float, total_bar: int):
        self.data = []
        self.current_bar = current_bar
        self.is_last_bar = is_last_bar
        self.last_price = last_price
        self.total_bar = total_bar
        self._nbars = 10


class BrokerAccount:
    """
    Represents the broker's account.

    It tracks balance, equity, opened trades, and other relevant information.
    """

    def __init__(self, broker):
        params = broker.params
        self.broker: Broker = broker
        self.params: Params = params
        self.balance: float = params.initial_balance
        self.equity: float = params.initial_balance
        self.hedging: bool = False  # opening multiple positions
        self.netting: bool = True  # opening one position
        self.opened_trades: List[Trade] = []
        self.closed_trades: List[Trade] = []
        self.history_balance: np.ndarray = np.tile(params.initial_balance, len(params.dataset))
        self.history_equity: np.ndarray = np.tile(params.initial_balance, len(params.dataset))
        self.history_commission: np.ndarray = np.tile(0, len(params.dataset))
        self.commission_spent: float = 0

    def refresh_values(self):
        self.commission_spent = sum(trade.commissions for trade in self.opened_trades)
        self.commission_spent += sum(trade.commissions for trade in self.closed_trades)
        self.equity = round(self.balance + sum(trade.pl_value - trade.commissions for trade in self.opened_trades), 2)
        self.history_balance[self.broker.state.current_bar] = self.balance
        self.history_equity[self.broker.state.current_bar] = self.equity
        self.history_commission[self.broker.state.current_bar] = round(self.commission_spent, 2)

    def __open(self, is_long: bool = False, value: float = None, price: float = None, leverage: float | None = 1):
        """Open a new trade."""
        if self.netting:
            self.close()

        if self.broker.state.is_last_bar:
            # refrain from opening a new trade on the final bar.
            return

        # calculate entry value based on parameters
        if value:
            # by custom value
            entry_value = value
        elif self.params.default_entry_value <= 1:
            # by percent
            entry_value = min(self.balance * self.params.default_entry_value, self.params.default_entry_value_max)
        else:
            # by cash unit
            entry_value = min(self.params.default_entry_value, self.params.default_entry_value_max)

        if leverage is not None:
            entry_value = entry_value * leverage

        # create a new trade and store it in the account
        opened_trade = Trade(self.broker.state)
        opened_trade.leverage = leverage
        opened_trade.entry_commission = entry_value * self.params.commission
        opened_trade.entry_value = entry_value - opened_trade.entry_commission
        opened_trade.entry_price = price or self.broker.state.last_price
        opened_trade.entry_bar = self.broker.state.current_bar
        opened_trade.entry_time = self.params.dataset.iloc[opened_trade.entry_bar].name
        opened_trade.is_long = is_long
        self.opened_trades.append(opened_trade)

    def __close(self, trade: Trade, exit_price: float = None):
        """Close an existing trade."""
        # close the trade and update balances and commissions
        self.opened_trades.remove(trade)
        closed_trade = trade
        closed_trade.exit_bar = self.broker.state.current_bar
        closed_trade.exit_price = exit_price or self.broker.state.last_price
        closed_trade.exit_commission = (trade.entry_value + trade.pl_value) * self.params.commission
        closed_trade.exit_value = trade.pl_value + trade.entry_value
        closed_trade.exit_time = self.params.dataset.iloc[closed_trade.exit_bar].name
        self.closed_trades.append(closed_trade)
        self.balance += round(closed_trade.pl_value - closed_trade.exit_commission, 2)
        pass

    def close(self):
        """Close all open trades."""
        for trade in list(self.opened_trades):
            self.__close(trade)

    def buy(self, leverage=1):
        self.__open(is_long=True, leverage=leverage)
        pass

    def sell(self, leverage=1):
        self.__open(is_long=False, leverage=leverage)
        pass


class Broker:
    """
    Represents the broker in the backtester.

    It provides a way to set up and run the backtesting process.
    """

    def __init__(self, params: Params):
        self.params = params
        self.state: BrokerState = BrokerState(
            current_bar=0,
            is_last_bar=False,
            last_price=False,
            total_bar=len(params.dataset),
        )
        self.account_main: BrokerAccount = BrokerAccount(self)

    def set_next_bar(self, index: int):
        """Set the next bar to process."""
        _start = index - self.state._nbars if index - self.state._nbars > 0 else 0
        _end = index + 1

        self.state.current_bar = index
        self.state.data = self.params.dataset.iloc[_start:_end]
        self.state.is_last_bar = index + 1 == self.state.total_bar
        self.state.last_price = self.state.data.iloc[-1][self.params.close_column]  # fmt: off
        self.refresh()

    def refresh(self):
        self.account_main.refresh_values()

    def buy(self, leverage=1):
        """Start buying."""
        self.account_main.buy(leverage=leverage)

    def sell(self, leverage=1):
        """Start selling."""
        self.account_main.sell(leverage=leverage)

    def close(self):
        """Close all trades."""
        self.account_main.close()


class Backtester:
    """
    Manages the backtesting process.

    It provides a way to run backtesting on historical data and perform trades.
    """

    def __init__(
        self,
        dataset: pd.DataFrame,
        initial_balance: float = 10000.0,
        commission: float = 0.001,
        default_entry_value: float = 1,  # between 0.01 and 1 (percent)
        default_entry_value_max: float = 20000,
    ) -> None:
        self.params: Params = Params(
            dataset,
            initial_balance,
            commission,
            default_entry_value,
            default_entry_value_max,
        )

    def trades(self) -> pd.DataFrame:
        """Get the list of trades."""
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
            }
        )

    def history(self) -> pd.DataFrame:
        """Get the list of history."""
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

    def run(self):
        """Run the backtesting process."""
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

    def stats(self):
        return stats(self.history(), self.trades())

    def plot(self, w=1024, h=900, show_signals=False):
        return plot_basic(history=self.history(), params=self.params, w=w, h=h, show_signals=show_signals)

    def thumbnail(self, title=None, w=4, h=1):
        return plot_thumbnail(history=self.history(), params=self.params, stats=self.stats(), title=title, w=w, h=h)


# Example usage:
# if __name__ == "__main__":
# Create a dataset
# data = pd.DataFrame()

# Set up the backtester
# bt = Backtester(dataset=data, initial_balance=10000.0)

# Run the backtesting process
# for broker in bt.run():
#     current_bar = broker.state.data.iloc[-1]
#     previous_bar = broker.state.data.iloc[-2]
#     has_signal_changed = current_bar["signal"] != previous_bar["signal"]
#     if has_signal_changed:
#         print(f"Current bar: {current_bar['signal']} {current_bar['close']}")
#         print(f"Previous bar: {previous_bar['signal']} {previous_bar['close']}")
