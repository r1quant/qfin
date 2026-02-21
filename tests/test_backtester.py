"""Tests for the core Backtester engine."""

import numpy as np
import pandas as pd
import pytest

from qfin.backtester.backtester import Backtester, Broker, BrokerState, Params, Trade


class TestTrade:
    def test_pl_value_long_profit(self):
        state = BrokerState(current_bar=0, is_last_bar=False, last_price=110.0, total_bar=10)
        trade = Trade(state)
        trade.is_long = True
        trade.entry_value = 1000.0
        trade.entry_price = 100.0
        trade.exit_price = 110.0
        assert trade.pl_value == pytest.approx(100.0)

    def test_pl_value_long_loss(self):
        state = BrokerState(current_bar=0, is_last_bar=False, last_price=90.0, total_bar=10)
        trade = Trade(state)
        trade.is_long = True
        trade.entry_value = 1000.0
        trade.entry_price = 100.0
        trade.exit_price = 90.0
        assert trade.pl_value == pytest.approx(-100.0)

    def test_pl_value_short_profit(self):
        state = BrokerState(current_bar=0, is_last_bar=False, last_price=90.0, total_bar=10)
        trade = Trade(state)
        trade.is_long = False
        trade.entry_value = 1000.0
        trade.entry_price = 100.0
        trade.exit_price = 90.0
        # short profit: entry_value * (entry_price / exit_price - 1)
        expected = 1000.0 * (100.0 / 90.0 - 1)
        assert trade.pl_value == pytest.approx(expected)

    def test_pl_value_uses_state_price_when_no_exit(self):
        state = BrokerState(current_bar=0, is_last_bar=False, last_price=105.0, total_bar=10)
        trade = Trade(state)
        trade.is_long = True
        trade.entry_value = 1000.0
        trade.entry_price = 100.0
        trade.exit_price = None
        assert trade.pl_value == pytest.approx(50.0)

    def test_pl_pct_long(self):
        state = BrokerState(current_bar=0, is_last_bar=False, last_price=110.0, total_bar=10)
        trade = Trade(state)
        trade.is_long = True
        trade.entry_price = 100.0
        trade.exit_price = 110.0
        assert trade.pl_pct == pytest.approx(0.1)

    def test_pl_pct_short(self):
        state = BrokerState(current_bar=0, is_last_bar=False, last_price=90.0, total_bar=10)
        trade = Trade(state)
        trade.is_long = False
        trade.entry_price = 100.0
        trade.exit_price = 90.0
        expected = 100.0 / 90.0 - 1
        assert trade.pl_pct == pytest.approx(expected)

    def test_pl_pct_uses_state_when_no_exit(self):
        """Verify the bug fix: pl_pct uses self.state.last_price, not self.broker."""
        state = BrokerState(current_bar=0, is_last_bar=False, last_price=105.0, total_bar=10)
        trade = Trade(state)
        trade.is_long = True
        trade.entry_price = 100.0
        trade.exit_price = None
        assert trade.pl_pct == pytest.approx(0.05)

    def test_commissions(self):
        trade = Trade()
        trade.entry_commission = 1.5
        trade.exit_commission = 2.0
        assert trade.commissions == pytest.approx(3.5)


class TestParams:
    def test_dataset_is_copied(self, simple_dataset):
        params = Params(dataset=simple_dataset)
        params.dataset.iloc[0, 0] = -999
        assert simple_dataset.iloc[0, 0] != -999

    def test_defaults(self, simple_dataset):
        params = Params(dataset=simple_dataset)
        assert params.initial_balance == 10000
        assert params.close_column == "close"


class TestBacktester:
    def test_run_yields_broker_per_bar(self, simple_dataset):
        bt = Backtester(dataset=simple_dataset, initial_balance=10000)
        bars = list(bt.run())
        # run() starts at bar 1 and goes to the end
        assert len(bars) == len(simple_dataset) - 1

    def test_buy_and_sell_creates_trades(self, simple_dataset):
        bt = Backtester(dataset=simple_dataset, initial_balance=10000, commission=0)
        for broker in bt.run():
            current = broker.state.data.iloc[-1]
            previous = broker.state.data.iloc[-2]
            if current["signal"] != previous["signal"]:
                if current["signal"] == 1:
                    broker.buy()
                elif current["signal"] == -1:
                    broker.sell()
                else:
                    broker.close()

        trades = bt.trades()
        assert len(trades) > 0
        assert "pnl" in trades.columns
        assert "return_pct" in trades.columns

    def test_netting_mode_closes_existing_position(self, simple_dataset):
        """In netting mode, opening a new position closes the previous one."""
        bt = Backtester(dataset=simple_dataset, initial_balance=10000, commission=0)
        for broker in bt.run():
            if broker.state.current_bar == 2:
                broker.buy()
            elif broker.state.current_bar == 5:
                broker.sell()  # should close the buy first

        trades = bt.trades()
        # Should have the first trade (buy closed by sell) + the sell (closed at end)
        assert len(trades) >= 1

    def test_no_trade_on_last_bar(self, simple_dataset):
        bt = Backtester(dataset=simple_dataset, initial_balance=10000, commission=0)
        for broker in bt.run():
            if broker.state.is_last_bar:
                broker.buy()  # this should be ignored

        trades = bt.trades()
        # All trades should have been force-closed at end, no new opens on last bar
        buy_on_last = [t for t in trades.itertuples() if t.entry_bar == len(simple_dataset) - 1]
        assert len(buy_on_last) == 0

    def test_history_has_expected_columns(self, simple_dataset):
        bt = Backtester(dataset=simple_dataset, initial_balance=10000)
        list(bt.run())  # exhaust the generator
        history = bt.history()
        expected_cols = {"close", "balance", "equity", "commission", "long", "short", "signal", "buy_hold"}
        assert expected_cols.issubset(set(history.columns))

    def test_history_buy_hold(self, simple_dataset):
        bt = Backtester(dataset=simple_dataset, initial_balance=10000)
        list(bt.run())
        history = bt.history()
        # buy_hold should scale with close price
        first_close = history.iloc[0]["close"]
        last_close = history.iloc[-1]["close"]
        first_hold = history.iloc[0]["buy_hold"]
        last_hold = history.iloc[-1]["buy_hold"]
        assert last_hold / first_hold == pytest.approx(last_close / first_close, rel=1e-6)

    def test_no_trades_balance_unchanged(self, flat_dataset):
        """If no trades are executed, balance stays at initial."""
        bt = Backtester(dataset=flat_dataset, initial_balance=10000)
        list(bt.run())
        history = bt.history()
        assert history.iloc[-1]["balance"] == 10000

    def test_commission_reduces_balance(self, simple_dataset):
        bt_no_comm = Backtester(dataset=simple_dataset, initial_balance=10000, commission=0)
        for broker in bt_no_comm.run():
            if broker.state.current_bar == 2:
                broker.buy()

        bt_with_comm = Backtester(dataset=simple_dataset, initial_balance=10000, commission=0.01)
        for broker in bt_with_comm.run():
            if broker.state.current_bar == 2:
                broker.buy()

        final_no_comm = bt_no_comm.history().iloc[-1]["balance"]
        final_with_comm = bt_with_comm.history().iloc[-1]["balance"]
        assert final_with_comm < final_no_comm

    def test_trades_dataframe_columns(self, simple_dataset):
        bt = Backtester(dataset=simple_dataset, initial_balance=10000, commission=0)
        for broker in bt.run():
            if broker.state.current_bar == 3:
                broker.buy()

        trades = bt.trades()
        expected_cols = [
            "is_long",
            "leverage",
            "entry_value",
            "entry_price",
            "entry_bar",
            "entry_commission",
            "entry_time",
            "exit_value",
            "exit_price",
            "exit_commission",
            "exit_bar",
            "exit_time",
            "pnl",
            "return_pct",
        ]
        assert list(trades.columns) == expected_cols

    def test_default_entry_value_percentage(self, simple_dataset):
        """default_entry_value=0.5 should use 50% of balance."""
        bt = Backtester(dataset=simple_dataset, initial_balance=10000, commission=0, default_entry_value=0.5)
        for broker in bt.run():
            if broker.state.current_bar == 2:
                broker.buy()

        trades = bt.trades()
        assert trades.iloc[0]["entry_value"] == pytest.approx(5000.0)

    def test_default_entry_value_cash(self, simple_dataset):
        """default_entry_value=3000 (>1) should use $3000 as cash."""
        bt = Backtester(dataset=simple_dataset, initial_balance=10000, commission=0, default_entry_value=3000)
        for broker in bt.run():
            if broker.state.current_bar == 2:
                broker.buy()

        trades = bt.trades()
        assert trades.iloc[0]["entry_value"] == pytest.approx(3000.0)

    def test_entry_value_max_cap(self, simple_dataset):
        """default_entry_value_max should cap the position size."""
        bt = Backtester(
            dataset=simple_dataset, initial_balance=10000, commission=0, default_entry_value=1, default_entry_value_max=5000
        )
        for broker in bt.run():
            if broker.state.current_bar == 2:
                broker.buy()

        trades = bt.trades()
        assert trades.iloc[0]["entry_value"] <= 5000.0

    def test_leverage(self, simple_dataset):
        """Leverage should multiply the entry value."""
        bt = Backtester(dataset=simple_dataset, initial_balance=10000, commission=0, default_entry_value=1)
        for broker in bt.run():
            if broker.state.current_bar == 2:
                broker.buy(leverage=2)

        trades = bt.trades()
        assert trades.iloc[0]["entry_value"] == pytest.approx(20000.0)
