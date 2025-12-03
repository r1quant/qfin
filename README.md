
Quantitative finance research tools in Python

# Install

```bash
# install 
uv add git+https://github.com/r1quant/qfin

# upgrade
uv add git+https://github.com/r1quant/qfin --upgrade-package qfin 
```

# How to use 

### API 

```python 

# yahoo
from qfin.api.yahoo import yahoo
yahoo(ticker="^SPX", start="2025-01-01", interval="1d")

# tradingview
from qfin.api.tv import Interval, TvDatafeed
tv = TvDatafeed()
tv.get_hist(symbol="SPX", exchange="SP", interval=Interval.in_daily, n_bars=260)

# bybit
# .env: BYBIT_API_KEY=<your-api-key>
# .env: BYBIT_API_SECRET=<your-api-secret>
from qfin.api.bybit import bybit
bybit(ticker="BTCUSD", start="2014-01-01", end=None, interval="d")

# fred
from qfin.api.fred import fred
fred("M2SL")
```

### Backtest Engine

```python 
import qfin

# date,        close,    signal
# 2023-01-03,  3824.13,   1
# 2023-01-04,  3852.96,   1
# 2023-01-05,  3808.10,   1
# ...    
# 2025-04-03,  5396.52,  -1
# 2025-04-04,  5074.08,  -1
# 2025-04-07,  5062.25,  -1

df = pd.read_csv("./my_table_above.csv", index_col=0, parse_dates=[0], sep=",")

backtest_params = {
    "initial_balance": 10000,
    "default_entry_value": 1, # 100% (that will be $10000 per trade)
    "default_entry_value_max": 20000, # but max $20000
    "commission": 0.001,
}

bt = qfin.Backtester(dataset=df, **backtest_params)

# ---- running strategy ------------
for broker in bt.run():
    current_bar = broker.state.data.iloc[-1]
    previous_bar = broker.state.data.iloc[-2]

    current_signal = current_bar["signal"]
    previous_signal = previous_bar["signal"]
    changed = current_signal != previous_signal

    if changed:
        if current_signal == 1:
            broker.buy()
        elif current_signal == -1:
            broker.sell()
        else:
            broker.close()
```

#### Backtest Result

```python
# ---- print statistics ------------
print(bt.stats())

# ---- plot result ------------
bt.plot()
bt.thumbnail()
```

#### Predefined Backtest Runners

Instead of creating a new strategy each time, you can reuse certain predefined strategies, which might be more efficient and effective in the long term.

```python
from qfin.backtester.runners import bt_signal_change

# date,        close,    signal
# 2023-01-03,  3824.13,   1
# 2023-01-04,  3852.96,   1
# 2023-01-05,  3808.10,   1
# ...    
# 2025-04-03,  5396.52,  -1
# 2025-04-04,  5074.08,  -1
# 2025-04-07,  5062.25,  -1

df = pd.read_csv("./my_table_above.csv", index_col=0, parse_dates=[0], sep=",")

backtest_params = {
    "initial_balance": 10000,
    "default_entry_value": 1,
    "default_entry_value_max": 20000,
    "commission": 0.001,
}

bt = bt_signal_change(dataset=df, **backtest_params)

# print statistics
print(bt.stats())
```

## License

This project is licensed under the MIT License.