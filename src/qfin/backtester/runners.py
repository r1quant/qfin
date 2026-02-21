"""Predefined backtest runners for common strategy patterns."""

from __future__ import annotations

import pandas as pd

from qfin.backtester.backtester import Backtester


def bt_signal_change(dataset: pd.DataFrame, leverage_column: str | None = None, **kwargs) -> Backtester:
    """Run a backtest that trades on signal changes.

    Buys when signal transitions to 1, sells (short) when signal transitions to -1,
    and closes all positions when signal transitions to any other value.

    Args:
        dataset: DataFrame with 'close' and 'signal' columns (DatetimeIndex).
        leverage_column: Optional column name for per-bar leverage values.
        **kwargs: Additional arguments passed to Backtester (initial_balance, commission, etc.).

    Returns:
        The completed Backtester instance with trades and statistics available.
    """
    bt = Backtester(dataset=dataset, **kwargs)

    for broker in bt.run():
        current_bar = broker.state.data.iloc[-1]
        previous_bar = broker.state.data.iloc[-2]
        leverage = current_bar[leverage_column] if leverage_column else None

        current_signal = current_bar["signal"]
        previous_signal = previous_bar["signal"]
        changed = current_signal != previous_signal

        if changed:
            if current_signal == 1:
                broker.buy(leverage=leverage)
            elif current_signal == -1:
                broker.sell(leverage=leverage)
            else:
                broker.close()

    return bt
