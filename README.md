# qfin — Quantitative Finance Research Tools

## Overview

`qfin` is a Python library for quantitative finance research. It provides:

1. **Backtester** — Event-driven backtesting engine for testing trading strategies on historical data, with built-in risk management (trailing stops, breakeven, take profit, stop loss)

## Installation

```bash
uv add git+https://github.com/r1quant/qfin

# upgrade
uv add git+https://github.com/r1quant/qfin --upgrade-package qfin
```

## Example

```python
from qfin.backtester.runners import bt_signal_change

backtest_params = {
    "initial_balance": 10000,
    "default_entry_value": 1,
    "default_entry_value_max": 10000,
    "commission": 0.001,
    "trailing_enabled": True,
    "trailing_activation_pct": 2,
    "trailing_distance_pct": 2,
    "trailing_min_step_pct": 1,
    "takeprofit_pct": 10,
    "stoploss_pct": 1,
    "breakeven_pct": 2,
    "leverage_column": "leverage",
}

bt = bt_signal_change(data, **backtest_params)
bt.trades()
bt.stats()
bt.thumbnail()
bt.plot()
```

## Documentation

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
    trailing_enabled=True,
    trailing_activation_pct=2,      # activate after 2% profit
    trailing_distance_pct=1,        # 1% distance from high/low
    trailing_min_step_pct=0.5,      # min 0.5% step to update stop
    takeprofit_pct=10,              # close at 10% profit
    stoploss_pct=3,                 # close at 3% loss
    breakeven_pct=2,                # move stop to entry after 2% profit
)
```

| Parameter                 | Type           | Default     | Description                                           |
| ------------------------- | -------------- | ----------- | ----------------------------------------------------- |
| `dataset`                 | `pd.DataFrame` | —           | Historical data with `close` column and DatetimeIndex |
| `initial_balance`         | `float`        | `10000.0`   | Starting cash balance                                 |
| `commission`              | `float`        | `0.01`      | Commission rate per trade (0.01 = 1%)                 |
| `default_entry_value`     | `float`        | `1`         | If ≤ 1: percentage of balance; if > 1: cash amount    |
| `default_entry_value_max` | `float`        | `1000000.0` | Maximum position size cap                             |
| `trailing_enabled`        | `bool`         | `False`     | Enable trailing stop functionality                    |
| `trailing_distance_pct`   | `float`        | `0`         | Distance from trailing high/low for stop (0–100%)     |
| `trailing_activation_pct` | `float`        | `0`         | Profit threshold to activate trailing stop (0–100%)   |
| `trailing_min_step_pct`   | `float`        | `0`         | Minimum step size for trailing stop updates (0–100%)  |
| `takeprofit_pct`          | `float`        | `0`         | Take profit at this profit percentage (0–100%)        |
| `stoploss_pct`            | `float`        | `0`         | Stop loss at this loss percentage (0–100%)            |
| `breakeven_pct`           | `float`        | `0`         | Move stop to entry price after this profit (0–100%)   |

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

| Property      | Description                                                                              |
| ------------- | ---------------------------------------------------------------------------------------- |
| `pl_value`    | Profit/loss in cash units                                                                |
| `pl_pct`      | Profit/loss as a decimal fraction (e.g., 0.05 = 5%)                                      |
| `commissions` | Total entry + exit commissions                                                           |
| `exit_reason` | Why the trade was closed: `"manual"`, `"stoploss"`, `"takeprofit"`, `"breakeven"`, or `"trailing"` |

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

### Risk Management

Risk management checks run automatically at each bar before strategy logic. All percentage parameters must be between 0 and 100. Set to 0 (default) to disable.

- **Stop Loss** (`stoploss_pct`): Closes the trade when unrealized loss reaches the threshold.
- **Take Profit** (`takeprofit_pct`): Closes the trade when unrealized profit reaches the threshold.
- **Breakeven** (`breakeven_pct`): Once profit reaches this threshold, the stop is moved to the entry price. If price returns to entry, the trade is closed at break-even.
- **Trailing Stop** (`trailing_enabled`): Tracks the highest price (long) or lowest price (short) since entry.
  - Activates only after profit reaches `trailing_activation_pct`.
  - Stop is placed at `trailing_distance_pct` from the trailing high/low.
  - Stop only moves forward (never backwards), and only updates when the move exceeds `trailing_min_step_pct`.
- **Stop dominance**: When both breakeven and trailing stops are active, the most restrictive (closest to current price) takes priority.

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

## License

This project is licensed under the MIT License.
