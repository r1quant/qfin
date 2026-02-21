# qfin — Quantitative Finance Research Tools

## Overview

`qfin` is a Python library for quantitative finance research. It provides:

1. **API wrappers** — Unified interface to fetch historical price data from Yahoo Finance, Bybit, TradingView, and FRED
2. **Backtester** — Event-driven backtesting engine for testing trading strategies on historical data
3. **Indicators** — Signal transformation utilities (crossovers, echo/forward-fill, etc.)

## File Structure

```
src/qfin/
├── __init__.py                  # Re-exports Backtester
├── api/
│   ├── bybit.py                 # Bybit exchange data (requires API keys)
│   ├── fred.py                  # FRED macroeconomic data
│   ├── tv.py                    # TradingView data
│   └── yahoo.py                 # Yahoo Finance data
├── backtester/
│   ├── backtester.py            # Core backtesting engine
│   ├── plot.py                  # Plotly and matplotlib visualizations
│   ├── runners.py               # Predefined strategy runners
│   └── stats.py                 # Performance statistics computation
└── indicators/
    └── common.py                # Signal utilities (crossover, echo, etc.)
```

## Installation

```bash
uv add git+https://github.com/r1quant/qfin

# upgrade
uv add git+https://github.com/r1quant/qfin --upgrade-package qfin
```

---

## API Module

All API functions return a pandas DataFrame with lowercase OHLCV columns (`open`, `high`, `low`, `close`, `volume`) and a DatetimeIndex named `date`.

### `yahoo(ticker, start, end, interval, period, ...)`

Fetch data from Yahoo Finance via `yfinance`.

```python
from qfin.api.yahoo import yahoo
df = yahoo(ticker="^SPX", start="2025-01-01", interval="1d")
```

| Parameter  | Type            | Default | Description                               |
| ---------- | --------------- | ------- | ----------------------------------------- |
| `ticker`   | `str` or `list` | —       | Ticker symbol(s)                          |
| `start`    | `str`           | `None`  | Start date (YYYY-MM-DD)                   |
| `end`      | `str`           | `None`  | End date (YYYY-MM-DD)                     |
| `interval` | `str`           | `"1d"`  | Data interval (1d, 1wk, 1mo, etc.)        |
| `period`   | `str`           | `"max"` | Period to download (1d, 5d, 1mo, 1y, max) |

### `bybit(ticker, start, end, interval, ...)`

Fetch data from Bybit. Requires `BYBIT_API_KEY` and `BYBIT_API_SECRET` environment variables.

```python
from qfin.api.bybit import bybit
df = bybit(ticker="BTCUSD", start="2014-01-01", end=None, interval="d")
```

| Parameter    | Type      | Default | Description                                                                        |
| ------------ | --------- | ------- | ---------------------------------------------------------------------------------- |
| `ticker`     | `str`     | —       | Trading pair (e.g., "BTCUSD")                                                      |
| `start`      | `str`     | —       | Start date (YYYY-MM-DD), **required**                                              |
| `end`        | `str`     | `None`  | End date; if None, fetches up to present                                           |
| `interval`   | `str/int` | `240`   | Candle interval: 1,3,5,15,30,60,120,240,360,720,D,W,M or aliases (1h, 4h, d, etc.) |
| `limit`      | `int`     | `1000`  | Max candles per API call                                                           |
| `sleep_time` | `float`   | `1.5`   | Seconds between paginated API calls                                                |

### `fred(series)`

Fetch time series from the Federal Reserve Economic Data.

```python
from qfin.api.fred import fred
df = fred("M2SL")  # M2 Money Supply
```

### `TvDatafeed` (TradingView)

```python
from qfin.api.tv import Interval, TvDatafeed
tv = TvDatafeed()
df = tv.get_hist(symbol="SPX", exchange="SP", interval=Interval.in_daily, n_bars=260)
```

---

## Backtester Module

### Core Classes

#### `Backtester`

Main interface. Accepts a DataFrame with at least a `close` column and a DatetimeIndex.

```python
import qfin

bt = qfin.Backtester(
    dataset=df,
    initial_balance=10000,
    commission=0.001,
    default_entry_value=1,          # 1 = 100% of balance per trade
    default_entry_value_max=20000,  # max $20,000 per trade
)
```

| Parameter                 | Type           | Default   | Description                                           |
| ------------------------- | -------------- | --------- | ----------------------------------------------------- |
| `dataset`                 | `pd.DataFrame` | —         | Historical data with `close` column and DatetimeIndex |
| `initial_balance`         | `float`        | `10000.0` | Starting cash balance                                 |
| `commission`              | `float`        | `0.001`   | Commission rate per trade (0.001 = 0.1%)              |
| `default_entry_value`     | `float`        | `1`       | If ≤ 1: percentage of balance; if > 1: cash amount    |
| `default_entry_value_max` | `float`        | `20000`   | Maximum position size cap                             |

**Key Methods:**

| Method                     | Returns             | Description                                                   |
| -------------------------- | ------------------- | ------------------------------------------------------------- |
| `run()`                    | `Generator[Broker]` | Yields a `Broker` at each bar for strategy logic              |
| `trades()`                 | `pd.DataFrame`      | All closed trades with entry/exit details and P&L             |
| `history()`                | `pd.DataFrame`      | Per-bar close, balance, equity, commission, signals, buy_hold |
| `stats()`                  | `pd.Series`         | Performance statistics                                        |
| `plot(w, h, show_signals)` | —                   | Interactive Plotly chart                                      |
| `thumbnail(title, w, h)`   | —                   | Compact matplotlib thumbnail                                  |

#### `Broker` (yielded by `run()`)

The strategy interacts with the broker at each bar:

| Method                    | Description              |
| ------------------------- | ------------------------ |
| `broker.buy(leverage=1)`  | Open a long position     |
| `broker.sell(leverage=1)` | Open a short position    |
| `broker.close()`          | Close all open positions |

**Properties available via `broker.state`:**

| Property            | Description                                             |
| ------------------- | ------------------------------------------------------- |
| `state.data`        | DataFrame slice of recent bars (up to 10 lookback bars) |
| `state.current_bar` | Index of the current bar                                |
| `state.is_last_bar` | Whether this is the final bar                           |
| `state.last_price`  | Close price of the current bar                          |

#### `Trade`

Represents a single trade. Key properties:

| Property      | Description                                         |
| ------------- | --------------------------------------------------- |
| `pl_value`    | Profit/loss in cash units                           |
| `pl_pct`      | Profit/loss as a decimal fraction (e.g., 0.05 = 5%) |
| `commissions` | Total entry + exit commissions                      |

### Usage Example

```python
import pandas as pd
import qfin


# -- data.csv
# date,        close,    signal
# 2023-01-03,  3824.13,   1
# 2023-01-04,  3852.96,   1
# 2023-01-05,  3808.10,   1
# ...
# 2025-04-03,  5396.52,  -1
# 2025-04-04,  5074.08,  -1
# 2025-04-07,  5062.25,  -1

df = pd.read_csv("data.csv", index_col=0, parse_dates=[0], sep=",")

backtest_params = {
    "initial_balance": 10000,
    "default_entry_value": 1, # 100% (that will be $10000 per trade)
    "default_entry_value_max": 20000, # but max $20000
    "commission": 0.001,
}

bt = qfin.Backtester(dataset=df, **backtest_params)

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

print(bt.stats())

bt.plot()
bt.thumbnail()
```

### Predefined Runners

#### `bt_signal_change(dataset, leverage_column=None, **kwargs)`

Trades on signal column transitions: buys on 1, sells (short) on -1, closes on other values.

```python
from qfin.backtester.runners import bt_signal_change

bt = bt_signal_change(dataset=df, initial_balance=10000, commission=0.001)
print(bt.stats())
```

With per-bar leverage:

```python
bt = bt_signal_change(dataset=df, leverage_column="leverage", initial_balance=10000)
```

### Key Behaviors

- **Netting mode** (default): Opening a new position automatically closes any existing one.
- **Signal convention**: `1` = long, `-1` = short, `0` = flat.
- **No trade on last bar**: The engine prevents opening new trades on the final bar.
- **Position sizing**: `default_entry_value ≤ 1` is treated as a percentage of current balance; `> 1` is treated as a fixed cash amount, capped at `default_entry_value_max`.
- **Commission**: Applied at both entry and exit, calculated as `entry_value * commission_rate`.

---

## Performance Metrics (`stats()`)

The `stats()` method returns a `pd.Series` with the following metrics:

### Time & Exposure

| Metric              | Description                              |
| ------------------- | ---------------------------------------- |
| `Start`             | First date in the dataset                |
| `End`               | Last date in the dataset                 |
| `Duration`          | Total time span                          |
| `Exposure Time [%]` | Percentage of bars with an open position |

### Returns

| Metric               | Formula                                                 | Description                                   |
| -------------------- | ------------------------------------------------------- | --------------------------------------------- |
| `Equity Return [%]`  | `(equity_final - equity_start) / equity_start × 100`    | Total return including unrealized P&L         |
| `Balance Return [%]` | `(balance_final - balance_start) / balance_start × 100` | Total return from closed trades               |
| `Gross Return [%]`   | `sum(return_pct) × 100`                                 | Sum of all trade returns (before commissions) |
| `Return (Ann.) [%]`  | `((1 + gmean_daily)^trading_days - 1) × 100`            | Annualized compounded return                  |
| `CAGR [%]`           | `(equity_final / equity_start)^(1/years) - 1) × 100`    | Compound Annual Growth Rate                   |

### Risk Ratios

| Metric                  | Formula                                        | Description                                        |
| ----------------------- | ---------------------------------------------- | -------------------------------------------------- |
| `Sharpe Ratio`          | `(Return_ann% - risk_free%) / Volatility_ann%` | Risk-adjusted return (default risk-free = 5%)      |
| `Sortino Ratio`         | `(return_ann - rf) / downside_deviation`       | Like Sharpe but only penalizes downside volatility |
| `Calmar Ratio`          | `annualized_return / abs(max_drawdown)`        | Return relative to worst drawdown                  |
| `Volatility (Ann.) [%]` | Compounded annualized standard deviation       | Annualized portfolio volatility                    |

**Sharpe Ratio formula:**

```
Sharpe = (Return_annualized_% - Risk_Free_Rate_%) / Volatility_annualized_%
```

Where `risk_free_rate` defaults to 5 (i.e., 5% per year). Both numerator and denominator are in percentage points.

**Sortino Ratio formula:**

```
Sortino = (annualized_return - risk_free_rate/100) / (sqrt(mean(min(daily_returns, 0)²)) × sqrt(trading_days))
```

Only negative daily returns contribute to the denominator (downside deviation).

**Calmar Ratio formula:**

```
Calmar = annualized_return / |max_drawdown|
```

### Drawdown

| Metric                   | Description                                       |
| ------------------------ | ------------------------------------------------- |
| `Max. Drawdown [%]`      | Largest peak-to-trough decline in equity          |
| `Avg. Drawdown [%]`      | Average peak drawdown across all drawdown periods |
| `Max. Drawdown Duration` | Longest drawdown recovery period                  |
| `Avg. Drawdown Duration` | Average drawdown recovery period                  |

### Trade Statistics

| Metric                | Description                            |
| --------------------- | -------------------------------------- |
| `Total Trades`        | Number of closed trades                |
| `Win Rate [%]`        | Percentage of trades with positive P&L |
| `Best Trade [%]`      | Highest single-trade return            |
| `Worst Trade [%]`     | Lowest single-trade return             |
| `Avg. Trade [%]`      | Geometric mean of trade returns        |
| `Max. Trade Duration` | Longest trade duration                 |
| `Avg. Trade Duration` | Average trade duration                 |
| `Long Trades`         | Number of long trades                  |
| `Short Trades`        | Number of short trades                 |

### Advanced Metrics

| Metric            | Formula                                              | Description                                       |
| ----------------- | ---------------------------------------------------- | ------------------------------------------------- |
| `Profit Factor`   | `sum(positive_returns) / abs(sum(negative_returns))` | Ratio of gross profits to gross losses            |
| `Expectancy [%]`  | `mean(return_pct) × 100`                             | Average expected return per trade                 |
| `SQN`             | `sqrt(n_trades) × mean(pnl) / std(pnl)`              | System Quality Number — measures strategy quality |
| `Kelly Criterion` | `win_rate - (1 - win_rate) / (avg_win / avg_loss)`   | Optimal fraction of capital to bet                |

---

## Indicators Module

### `continue_echo(dataserie, initial_value=None, skip_values=[None])`

Forward-fills a signal series, carrying the last non-skip value forward.

```python
# Input:  [0, -1,  0,  0,  0,  1,  0,  0,  0]
# Output: [0, -1, -1, -1, -1,  1,  1,  1,  1]
from qfin.indicators.common import continue_echo
result = continue_echo(df["signal"])
```

### `revert_echo(dataserie, empty_value=None)`

Inverse of `continue_echo` — collapses consecutive identical values to a single event.

```python
# Input:  [0, -1, -1, -1, -1,  1,  1,  1,  1]
# Output: [0, -1,  0,  0,  0,  1,  0,  0,  0]
from qfin.indicators.common import revert_echo
result = revert_echo(df["signal"], empty_value=0)
```

### `crossover(dataserie_a, dataserie_b=None, echo=False, nosignal_value=0)`

Detect crossover between two series. Returns `1` when A crosses above B, `-1` when A crosses below B.

```python
from qfin.indicators.common import crossover
signal = crossover(df["fast_ma"], df["slow_ma"], echo=True)
```

If `dataserie_b` is None, compares against the shifted version of `dataserie_a` (direction of change).

### `direction(dataserie_a, dataserie_b=None, echo=True)`

Alias for `crossover(..., echo=True, nosignal_value=0)`.

### `crossover3(dataserie_a, dataserie_b, dataserie_c, echo=False, nosignal_value=0)`

Three-series crossover producing six regimes:

| Value | Label        | Condition                       |
| ----- | ------------ | ------------------------------- |
| `3`   | bullish      | A > B > C                       |
| `2`   | accumulation | A > C > B (A > B, A > C, B < C) |
| `1`   | recovery     | A < C < B (A > B, A < C, B < C) |
| `-1`  | warning      | A < B, A > C, B > C             |
| `-2`  | distribution | A < B, A < C, B > C             |
| `-3`  | bearish      | A < B < C                       |

```python
from qfin.indicators.common import crossover3, crossover3_labels
signal = crossover3(df["fast"], df["mid"], df["slow"], echo=True)
```

## License

This project is licensed under the MIT License.
