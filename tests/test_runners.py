"""Tests for predefined backtest runners."""

import numpy as np
import pandas as pd
import pytest

from qfin.backtester.runners import bt_signal_change


class TestBtSignalChange:
    def test_basic_signal_change(self, simple_dataset):
        bt = bt_signal_change(dataset=simple_dataset, initial_balance=10000, commission=0)
        trades = bt.trades()
        assert len(trades) > 0

    def test_creates_long_trades_on_signal_1(self, simple_dataset):
        bt = bt_signal_change(dataset=simple_dataset, initial_balance=10000, commission=0)
        trades = bt.trades()
        long_trades = trades[trades["is_long"]]
        assert len(long_trades) > 0

    def test_creates_short_trades_on_signal_neg1(self, simple_dataset):
        bt = bt_signal_change(dataset=simple_dataset, initial_balance=10000, commission=0)
        trades = bt.trades()
        short_trades = trades[~trades["is_long"]]
        assert len(short_trades) > 0

    def test_no_signal_change_no_trades(self, flat_dataset):
        """If signal never changes, no trades should be created."""
        bt = bt_signal_change(dataset=flat_dataset, initial_balance=10000, commission=0)
        trades = bt.trades()
        assert len(trades) == 0

    def test_constant_signal_1_creates_one_trade(self):
        """Signal always 1 from bar 0 — no change, so no trade."""
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        df = pd.DataFrame({"close": [100.0] * 10, "signal": [1] * 10}, index=dates)
        bt = bt_signal_change(dataset=df, initial_balance=10000, commission=0)
        trades = bt.trades()
        assert len(trades) == 0

    def test_single_transition_0_to_1(self):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        df = pd.DataFrame(
            {"close": np.linspace(100, 110, 10), "signal": [0, 0, 0, 1, 1, 1, 1, 1, 1, 1]},
            index=dates,
        )
        bt = bt_signal_change(dataset=df, initial_balance=10000, commission=0)
        trades = bt.trades()
        assert len(trades) == 1
        assert trades.iloc[0]["is_long"] == True

    def test_single_transition_0_to_neg1(self):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        df = pd.DataFrame(
            {"close": np.linspace(100, 90, 10), "signal": [0, 0, 0, -1, -1, -1, -1, -1, -1, -1]},
            index=dates,
        )
        bt = bt_signal_change(dataset=df, initial_balance=10000, commission=0)
        trades = bt.trades()
        assert len(trades) == 1
        assert trades.iloc[0]["is_long"] == False

    def test_multiple_signal_changes(self):
        dates = pd.date_range("2024-01-01", periods=15, freq="D")
        signals = [0, 0, 1, 1, 1, -1, -1, -1, 0, 0, 1, 1, 1, 0, 0]
        df = pd.DataFrame({"close": np.linspace(100, 114, 15), "signal": signals}, index=dates)
        bt = bt_signal_change(dataset=df, initial_balance=10000, commission=0)
        trades = bt.trades()
        # transitions: 0→1 (buy), 1→-1 (netting: close+sell), -1→0 (close), 0→1 (buy, closed at end)
        assert len(trades) >= 3

    def test_leverage_column(self):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        df = pd.DataFrame(
            {
                "close": np.linspace(100, 110, 10),
                "signal": [0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
                "lev": [1, 1, 1, 2, 2, 2, 2, 2, 2, 2],
            },
            index=dates,
        )
        bt = bt_signal_change(dataset=df, leverage_column="lev", initial_balance=10000, commission=0)
        trades = bt.trades()
        assert len(trades) == 1
        assert trades.iloc[0]["leverage"] == 2

    def test_returns_backtester_instance(self, simple_dataset):
        bt = bt_signal_change(dataset=simple_dataset, initial_balance=10000)
        assert hasattr(bt, "stats")
        assert hasattr(bt, "history")
        assert hasattr(bt, "trades")

    def test_stats_available_after_run(self, simple_dataset):
        bt = bt_signal_change(dataset=simple_dataset, initial_balance=10000, commission=0.001)
        s = bt.stats()
        assert isinstance(s, pd.Series)
        assert "Sharpe Ratio" in s.index
        assert "Total Trades" in s.index

    def test_with_commission(self, simple_dataset):
        bt_no_comm = bt_signal_change(dataset=simple_dataset, initial_balance=10000, commission=0)
        bt_with_comm = bt_signal_change(dataset=simple_dataset, initial_balance=10000, commission=0.01)

        final_no_comm = bt_no_comm.history().iloc[-1]["balance"]
        final_with_comm = bt_with_comm.history().iloc[-1]["balance"]
        assert final_with_comm < final_no_comm
