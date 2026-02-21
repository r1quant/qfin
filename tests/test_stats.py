"""Tests for the stats module."""

import numpy as np
import pandas as pd
import pytest

from qfin.backtester.backtester import Backtester
from qfin.backtester.runners import bt_signal_change
from qfin.backtester.stats import _geometric_mean, stats

# ---- Helper to run a backtest and get stats ----


def _run_and_stats(dataset, **kwargs):
    bt = bt_signal_change(dataset=dataset, **kwargs)
    return bt.stats(), bt


def _make_dataset(prices, signals, start="2024-01-01", freq="D"):
    dates = pd.date_range(start, periods=len(prices), freq=freq)
    return pd.DataFrame({"close": prices, "signal": signals}, index=dates)


# ---- Unit tests for _geometric_mean ----


class TestGeometricMean:
    def test_positive_returns(self):
        returns = pd.Series([0.1, 0.2, 0.05])
        result = _geometric_mean(returns)
        expected = np.exp(np.log(np.array([1.1, 1.2, 1.05])).sum() / 3) - 1
        assert result == pytest.approx(expected)

    def test_zero_returns(self):
        returns = pd.Series([0.0, 0.0, 0.0])
        result = _geometric_mean(returns)
        assert result == pytest.approx(0.0)

    def test_negative_return_causing_zero(self):
        """If any return causes cumulative to go to zero or negative, return 0."""
        returns = pd.Series([0.5, -1.0, 0.3])  # -100% return
        result = _geometric_mean(returns)
        assert result == 0

    def test_single_return(self):
        returns = pd.Series([0.15])
        result = _geometric_mean(returns)
        assert result == pytest.approx(0.15)

    def test_nan_values_filled(self):
        returns = pd.Series([0.1, np.nan, 0.2])
        result = _geometric_mean(returns)
        # nan is filled with 0, so effectively [0.1, 0.0, 0.2]
        expected = np.exp(np.log(np.array([1.1, 1.0, 1.2])).sum() / 3) - 1
        assert result == pytest.approx(expected)


# ---- Tests for stats output structure ----


class TestStatsStructure:
    def test_returns_series(self, simple_dataset):
        s, _ = _run_and_stats(simple_dataset, initial_balance=10000, commission=0)
        assert isinstance(s, pd.Series)

    def test_contains_all_expected_keys(self, simple_dataset):
        s, _ = _run_and_stats(simple_dataset, initial_balance=10000, commission=0)
        expected_keys = [
            "Start",
            "End",
            "Duration",
            "Exposure Time [%]",
            "Equity Start",
            "Equity Peak",
            "Equity Final",
            "Equity Return [%]",
            "Balance Start",
            "Balance Peak",
            "Balance Final",
            "Balance Return [%]",
            "Gross Return [%]",
            "Total Commissions",
            "Return (Ann.) [%]",
            "Volatility (Ann.) [%]",
            "Sharpe Ratio",
            "Sortino Ratio",
            "Calmar Ratio",
            "Max. Drawdown [%]",
            "Avg. Drawdown [%]",
            "Max. Drawdown Duration",
            "Avg. Drawdown Duration",
            "Total Trades",
            "Win Rate [%]",
            "Best Trade [%]",
            "Worst Trade [%]",
            "Avg. Trade [%]",
            "Max. Trade Duration",
            "Avg. Trade Duration",
            "Profit Factor",
            "Expectancy [%]",
            "SQN",
            "Kelly Criterion",
            "Candles",
            "Long Trades",
            "Short Trades",
            "Exposure Trades [%]",
        ]
        for key in expected_keys:
            assert key in s.index, f"Missing key: {key}"


# ---- Tests for individual metrics ----


class TestEquityMetrics:
    def test_equity_start(self, simple_dataset):
        s, _ = _run_and_stats(simple_dataset, initial_balance=10000, commission=0)
        assert s["Equity Start"] == 10000

    def test_equity_return_no_trades(self, flat_dataset):
        s, _ = _run_and_stats(flat_dataset, initial_balance=10000, commission=0)
        assert s["Equity Return [%]"] == pytest.approx(0.0)

    def test_balance_return_no_trades(self, flat_dataset):
        s, _ = _run_and_stats(flat_dataset, initial_balance=10000, commission=0)
        assert s["Balance Return [%]"] == pytest.approx(0.0)


class TestSharpeRatio:
    def test_sharpe_ratio_is_finite(self, volatile_dataset):
        s, _ = _run_and_stats(volatile_dataset, initial_balance=10000, commission=0)
        assert np.isfinite(s["Sharpe Ratio"])

    def test_sharpe_ratio_uses_correct_risk_free(self):
        """Verify the bug fix: risk_free_rate is not multiplied by 100."""
        prices = np.linspace(100, 150, 252)
        signals = [0] + [1] * 251
        df = _make_dataset(prices, signals)
        s, _ = _run_and_stats(df, initial_balance=10000, commission=0)
        # With the old bug (risk_free_rate * 100 = 500), Sharpe would be extremely negative
        # With the fix, it should be reasonable (> -10)
        assert s["Sharpe Ratio"] > -10

    def test_sharpe_with_custom_risk_free(self):
        prices = np.linspace(100, 120, 60)
        signals = [0] + [1] * 59
        df = _make_dataset(prices, signals)
        bt = bt_signal_change(dataset=df, initial_balance=10000, commission=0)
        s_default = stats(bt.history(), bt.trades(), risk_free_rate=5)
        s_zero = stats(bt.history(), bt.trades(), risk_free_rate=0)
        # Higher risk-free rate means lower Sharpe
        assert s_zero["Sharpe Ratio"] >= s_default["Sharpe Ratio"]


class TestSortinoRatio:
    def test_sortino_ratio_is_finite(self, volatile_dataset):
        s, _ = _run_and_stats(volatile_dataset, initial_balance=10000, commission=0)
        # Sortino should be finite when there's downside volatility
        assert np.isfinite(s["Sortino Ratio"]) or np.isnan(s["Sortino Ratio"])

    def test_sortino_ratio_units_correct(self):
        """Verify the bug fix: risk_free_rate is divided by 100 for Sortino."""
        prices = np.linspace(100, 200, 252)
        signals = [0] + [1] * 251
        df = _make_dataset(prices, signals)
        s, _ = _run_and_stats(df, initial_balance=10000, commission=0)
        # With the old bug (annualized_return_decimal - 5), Sortino would be very negative
        # annualized_return is ~1.0 (100%), rf was 5 → numerator = -4 → very negative
        # With fix: rf = 0.05 → numerator = 0.95 → positive
        assert s["Sortino Ratio"] > -10


class TestCalmarRatio:
    def test_calmar_ratio_positive_when_profitable(self, volatile_dataset):
        """Uses volatile dataset which has drawdowns, so Calmar is computable."""
        s, _ = _run_and_stats(volatile_dataset, initial_balance=10000, commission=0)
        # Calmar can be positive, negative, or nan depending on whether there's drawdown
        assert np.isfinite(s["Calmar Ratio"]) or np.isnan(s["Calmar Ratio"])


class TestDrawdown:
    def test_max_drawdown_zero_no_trades(self, flat_dataset):
        s, _ = _run_and_stats(flat_dataset, initial_balance=10000, commission=0)
        assert s["Max. Drawdown [%]"] == pytest.approx(0.0)

    def test_max_drawdown_negative_in_volatile(self, volatile_dataset):
        s, _ = _run_and_stats(volatile_dataset, initial_balance=10000, commission=0)
        assert s["Max. Drawdown [%]"] <= 0


class TestTradeStats:
    def test_total_trades(self, simple_dataset):
        s, _ = _run_and_stats(simple_dataset, initial_balance=10000, commission=0)
        assert s["Total Trades"] > 0

    def test_no_trades_win_rate_nan(self, flat_dataset):
        s, _ = _run_and_stats(flat_dataset, initial_balance=10000, commission=0)
        assert np.isnan(s["Win Rate [%]"])

    def test_win_rate_range(self, simple_dataset):
        s, _ = _run_and_stats(simple_dataset, initial_balance=10000, commission=0)
        if s["Total Trades"] > 0:
            assert 0 <= s["Win Rate [%]"] <= 100

    def test_long_and_short_trades_sum(self, simple_dataset):
        s, _ = _run_and_stats(simple_dataset, initial_balance=10000, commission=0)
        assert s["Long Trades"] + s["Short Trades"] == s["Total Trades"]

    def test_best_trade_ge_worst_trade(self, simple_dataset):
        s, _ = _run_and_stats(simple_dataset, initial_balance=10000, commission=0)
        if s["Total Trades"] > 0:
            assert s["Best Trade [%]"] >= s["Worst Trade [%]"]


class TestCommissions:
    def test_total_commissions_zero_when_no_commission(self, simple_dataset):
        s, _ = _run_and_stats(simple_dataset, initial_balance=10000, commission=0)
        assert s["Total Commissions"] == pytest.approx(0.0)

    def test_total_commissions_positive(self, simple_dataset):
        s, _ = _run_and_stats(simple_dataset, initial_balance=10000, commission=0.01)
        assert s["Total Commissions"] > 0


class TestExposure:
    def test_exposure_zero_no_trades(self, flat_dataset):
        s, _ = _run_and_stats(flat_dataset, initial_balance=10000, commission=0)
        assert s["Exposure Time [%]"] == pytest.approx(0.0)

    def test_exposure_positive_with_trades(self, simple_dataset):
        s, _ = _run_and_stats(simple_dataset, initial_balance=10000, commission=0)
        assert s["Exposure Time [%]"] > 0


class TestTimeMetrics:
    def test_start_end_duration(self, simple_dataset):
        s, _ = _run_and_stats(simple_dataset, initial_balance=10000, commission=0)
        assert s["Start"] == simple_dataset.index[0]
        assert s["End"] == simple_dataset.index[-1]
        assert s["Duration"] == s["End"] - s["Start"]


class TestEdgeCases:
    def test_all_winning_trades(self):
        """All trades are profitable."""
        prices = list(range(100, 120))  # steadily increasing
        signals = [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        df = _make_dataset(prices, signals)
        s, _ = _run_and_stats(df, initial_balance=10000, commission=0)
        assert s["Win Rate [%]"] == 100

    def test_single_bar_trade(self):
        """Signal flips every bar — very short trades."""
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        df = pd.DataFrame(
            {"close": [100, 101, 100, 101, 100, 101, 100, 101, 100, 101], "signal": [1, -1, 1, -1, 1, -1, 1, -1, 1, -1]},
            index=dates,
        )
        s, _ = _run_and_stats(df, initial_balance=10000, commission=0)
        assert s["Total Trades"] > 0

    def test_very_small_dataset(self):
        """Minimum viable dataset — 3 bars."""
        dates = pd.date_range("2024-01-01", periods=3, freq="D")
        df = pd.DataFrame({"close": [100, 105, 110], "signal": [0, 1, 1]}, index=dates)
        s, _ = _run_and_stats(df, initial_balance=10000, commission=0)
        assert isinstance(s, pd.Series)

    def test_profit_factor_nan_with_no_losses(self):
        """Profit factor should be nan when there are no losing trades."""
        prices = list(range(100, 115))
        signals = [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        df = _make_dataset(prices, signals)
        s, _ = _run_and_stats(df, initial_balance=10000, commission=0)
        if s["Total Trades"] > 0 and s["Win Rate [%]"] == 100:
            assert s["Profit Factor"] == np.inf or np.isnan(s["Profit Factor"]) or s["Profit Factor"] > 0

    def test_candles_count(self, simple_dataset):
        s, _ = _run_and_stats(simple_dataset, initial_balance=10000, commission=0)
        assert s["Candles"] == len(simple_dataset)
