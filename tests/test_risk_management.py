"""Tests for risk management features: stop loss, take profit, breakeven, trailing stop."""

import pandas as pd
import pytest

from qfin.backtester.backtester import Backtester, Params


def _make_df(prices):
    return pd.DataFrame({"close": prices}, index=pd.date_range("2024-01-01", periods=len(prices), freq="D"))


class TestStopLoss:
    def test_stop_loss_long(self):
        # Buy at bar 1 (price=100), bar 2 price drops to 94 → 6% loss > 5% SL
        df = _make_df([100, 100, 94, 95, 96])
        bt = Backtester(dataset=df, commission=0, stoploss_pct=5)
        for broker in bt.run():
            if broker.state.current_bar == 1:
                broker.buy()
        trades = bt.trades()
        assert len(trades) == 1
        assert trades.iloc[0]["exit_reason"] == "stoploss"
        assert trades.iloc[0]["exit_bar"] == 2

    def test_stop_loss_short(self):
        # Sell at bar 1 (price=100), bar 2 price rises to 106 → 6% loss for short > 5% SL
        df = _make_df([100, 100, 106, 105, 104])
        bt = Backtester(dataset=df, commission=0, stoploss_pct=5)
        for broker in bt.run():
            if broker.state.current_bar == 1:
                broker.sell()
        trades = bt.trades()
        assert len(trades) == 1
        assert trades.iloc[0]["exit_reason"] == "stoploss"
        assert trades.iloc[0]["exit_bar"] == 2


class TestTakeProfit:
    def test_take_profit_long(self):
        # Buy at bar 1 (price=100), bar 2 price rises to 106 → 6% profit > 5% TP
        df = _make_df([100, 100, 106, 105, 104])
        bt = Backtester(dataset=df, commission=0, takeprofit_pct=5)
        for broker in bt.run():
            if broker.state.current_bar == 1:
                broker.buy()
        trades = bt.trades()
        assert len(trades) == 1
        assert trades.iloc[0]["exit_reason"] == "takeprofit"
        assert trades.iloc[0]["exit_bar"] == 2

    def test_take_profit_short(self):
        # Sell at bar 1 (price=100), bar 2 price drops to 94 → short profit > 5%
        df = _make_df([100, 100, 94, 95, 96])
        bt = Backtester(dataset=df, commission=0, takeprofit_pct=5)
        for broker in bt.run():
            if broker.state.current_bar == 1:
                broker.sell()
        trades = bt.trades()
        assert len(trades) == 1
        assert trades.iloc[0]["exit_reason"] == "takeprofit"
        assert trades.iloc[0]["exit_bar"] == 2


class TestBreakeven:
    def test_breakeven_triggers_and_closes(self):
        # Buy at bar 1 (price=100), bar 2 price=102 (arms breakeven at 1%),
        # bar 3 price=101, bar 4 price=100 (<=entry → breakeven fires)
        df = _make_df([100, 100, 102, 101, 100, 99, 98])
        bt = Backtester(dataset=df, commission=0, breakeven_pct=1)
        for broker in bt.run():
            if broker.state.current_bar == 1:
                broker.buy()
        trades = bt.trades()
        assert len(trades) == 1
        assert trades.iloc[0]["exit_reason"] == "breakeven"
        assert trades.iloc[0]["exit_bar"] == 4


class TestTrailingStop:
    def test_trailing_stop_fires(self):
        # Buy at bar 1 (price=100), bar 2=108, bar 3=110 (trailing_high=110, stop=110*0.97=106.7),
        # bar 4=107, bar 5=106 (106 < 106.7 → trailing fires)
        df = _make_df([100, 100, 108, 110, 107, 106, 105])
        bt = Backtester(dataset=df, commission=0, trailing_enabled=True, trailing_distance_pct=3)
        for broker in bt.run():
            if broker.state.current_bar == 1:
                broker.buy()
        trades = bt.trades()
        assert len(trades) == 1
        assert trades.iloc[0]["exit_reason"] == "trailing"
        assert trades.iloc[0]["exit_bar"] == 5

    def test_trailing_takes_over_breakeven(self):
        # Buy at bar 1 (price=100), breakeven arms early, trailing stop moves above entry
        # trailing_stop > entry → trailing dominates
        df = _make_df([100, 100, 106, 110, 108, 106, 105])
        bt = Backtester(dataset=df, commission=0, breakeven_pct=1, trailing_enabled=True, trailing_distance_pct=3)
        for broker in bt.run():
            if broker.state.current_bar == 1:
                broker.buy()
        trades = bt.trades()
        assert len(trades) == 1
        assert trades.iloc[0]["exit_reason"] == "trailing"

    def test_breakeven_wins_over_wide_trailing(self):
        # Buy at bar 1 (price=100), breakeven arms at bar 2 (102, 2%>1%),
        # trailing with 20% distance → stop = ~81.6 which is < entry (100)
        # so breakeven dominates, closes when price hits 100
        df = _make_df([100, 100, 102, 101, 100, 99, 98])
        bt = Backtester(dataset=df, commission=0, breakeven_pct=1, trailing_enabled=True, trailing_distance_pct=20)
        for broker in bt.run():
            if broker.state.current_bar == 1:
                broker.buy()
        trades = bt.trades()
        assert len(trades) == 1
        assert trades.iloc[0]["exit_reason"] == "breakeven"


class TestValidation:
    def test_negative_pct_raises(self):
        df = _make_df([100, 100, 100])
        with pytest.raises(ValueError):
            Backtester(dataset=df, stoploss_pct=-1)

    def test_over_100_pct_raises(self):
        df = _make_df([100, 100, 100])
        with pytest.raises(ValueError):
            Backtester(dataset=df, takeprofit_pct=101)

    def test_trailing_distance_negative_raises(self):
        df = _make_df([100, 100, 100])
        with pytest.raises(ValueError):
            Backtester(dataset=df, trailing_distance_pct=-5)

    def test_breakeven_over_100_raises(self):
        df = _make_df([100, 100, 100])
        with pytest.raises(ValueError):
            Backtester(dataset=df, breakeven_pct=101)


class TestManualExitReason:
    def test_default_exit_reason_is_manual(self):
        df = _make_df([100, 100, 105, 110, 115])
        bt = Backtester(dataset=df, commission=0)
        for broker in bt.run():
            if broker.state.current_bar == 1:
                broker.buy()
            elif broker.state.current_bar == 3:
                broker.close()
        trades = bt.trades()
        assert len(trades) == 1
        assert trades.iloc[0]["exit_reason"] == "manual"
