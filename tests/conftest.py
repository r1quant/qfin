"""Shared test fixtures for backtester tests."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def simple_dataset():
    """A simple 20-bar dataset with linearly increasing prices and alternating signals."""
    dates = pd.date_range("2024-01-01", periods=20, freq="D")
    df = pd.DataFrame(
        {
            "close": np.linspace(100, 110, 20),
            "signal": [0] * 3 + [1] * 5 + [-1] * 5 + [1] * 5 + [0] * 2,
        },
        index=dates,
    )
    return df


@pytest.fixture
def flat_dataset():
    """A 20-bar dataset with constant prices and no signal."""
    dates = pd.date_range("2024-01-01", periods=20, freq="D")
    df = pd.DataFrame(
        {
            "close": [100.0] * 20,
            "signal": [0] * 20,
        },
        index=dates,
    )
    return df


@pytest.fixture
def volatile_dataset():
    """A dataset with up-down-up price movement for testing drawdown calculations."""
    dates = pd.date_range("2024-01-01", periods=30, freq="D")
    prices = [100.0]
    for i in range(1, 30):
        if i < 10:
            prices.append(prices[-1] * 1.02)  # +2% daily
        elif i < 20:
            prices.append(prices[-1] * 0.97)  # -3% daily
        else:
            prices.append(prices[-1] * 1.01)  # +1% daily
    df = pd.DataFrame(
        {
            "close": prices,
            "signal": [1] * 10 + [-1] * 10 + [1] * 10,
        },
        index=dates,
    )
    return df
