import os
import time
from datetime import datetime, timedelta

import pandas as pd
from pybit.unified_trading import HTTP


# date to timestamp
def to_timestamp_ms(value: str, is_end_time=False) -> int:
    """helper function"""
    dt = datetime.strptime(value, "%Y-%m-%d")
    if is_end_time:
        dt = dt + timedelta(hours=23)
        dt = dt + timedelta(minutes=59)
    return int(dt.timestamp()) * 1000


def pybit(ticker, start=None, end=None, interval=720, limit=1000, BYBIT_API_KEY=None, BYBIT_API_SECRET=None):
    """helper function
    doc: https://bybit-exchange.github.io/docs/v5/market/kline
    interval: 1,3,5,15,30,60,120,240,360,720,D,W,M
    """
    symbol = ticker.replace("-", "").replace("/", "")
    BYBIT_API_KEY = BYBIT_API_KEY or os.environ["BYBIT_API_KEY"]
    BYBIT_API_SECRET = BYBIT_API_SECRET or os.environ["BYBIT_API_SECRET"]
    session = HTTP(testnet=False, api_key=BYBIT_API_KEY, api_secret=BYBIT_API_SECRET)

    kLineIntervalDict = {
        "m15": 15,
        "1h": 60,
        "1H": 60,
        "h1": 60,
        "H1": 60,
        "4h": 240,
        "4H": 240,
        "h4": 240,
        "H4": 240,
        "h6": 360,
        "h12": 720,
        "d1": "d",
        "1d": "d",
        "D1": "d",
        "1D": "d",
        "D": "d",
    }

    response = session.get_kline(
        # category="inverse",
        symbol=symbol,
        interval=kLineIntervalDict.get(interval, interval),
        start=to_timestamp_ms(start),
        end=to_timestamp_ms(end, True) if end else None,
        limit=limit,
    )

    meta_data = response["result"]["list"]
    pybit_df = pd.DataFrame(meta_data)

    columns = {
        0: "date",
        1: "open",
        2: "high",
        3: "low",
        4: "close",
        5: "volume",
        6: "turnover",
    }

    pybit_df = pybit_df.rename(columns=columns)
    pybit_df["date"] = pd.to_numeric(pybit_df["date"])
    pybit_df["date"] = pd.to_datetime(pybit_df["date"], unit="ms")
    pybit_df = pybit_df.set_index("date")
    pybit_df = pybit_df.sort_index()
    return pybit_df


def bybit(
    ticker,
    start=None,
    end=None,
    interval=240,
    limit=1000,
    sleep_time=1.5,
    verbose=False,
    BYBIT_API_KEY=None,
    BYBIT_API_SECRET=None,
):
    arr = []

    i = 0
    is_true = True
    prev_start = start

    if verbose:
        print("ticker=", ticker)
        print("start=", start, "end=", end)
        print("limit=", limit, "interval=", interval)
        print("-" * 30)

    if start == None:
        print("'start' is required")
        return []

    _end = end
    _start = start

    if _end == None:
        while is_true:
            if verbose:
                print(f"[running.{i}]", _start, "to", _end)

            result = pybit(ticker, _start, _end, interval, limit, BYBIT_API_KEY=BYBIT_API_KEY, BYBIT_API_SECRET=BYBIT_API_SECRET)
            arr.append(result)

            if verbose:
                print("   start:", result.iloc[0].name, "end:", result.iloc[-1].name, "total:", len(result))

            next_start = result.iloc[-1].name
            next_start = next_start - timedelta(days=1)
            next_start = datetime.strftime(next_start, "%Y-%m-%d")

            is_true = (len(result) == limit) & (prev_start != next_start) & (end != next_start)

            if is_true:
                i = i + 1
                _start = next_start
                time.sleep(sleep_time)
    else:
        while is_true:
            if verbose:
                print(f"[running.{i}]", _start, "to", _end)

            result = pybit(ticker, _start, _end, interval, limit, BYBIT_API_KEY=BYBIT_API_KEY, BYBIT_API_SECRET=BYBIT_API_SECRET)
            arr.append(result)

            if verbose:
                print("   start:", result.iloc[0].name, "end:", result.iloc[-1].name, "total:", len(result), "diff:", result.iloc[-1].name - result.iloc[0].name)  # fmt: off

            next_end = result.iloc[0].name
            next_end = next_end + timedelta(days=1)

            diff = result.iloc[-1].name - result.iloc[0].name
            diff = abs(diff.days) * 2

            next_start = next_end - timedelta(days=int(diff))

            is_true = len(result) == limit

            if next_start < datetime.strptime(start, "%Y-%m-%d"):
                next_start = datetime.strptime(start, "%Y-%m-%d") - timedelta(days=1)

            if next_end < datetime.strptime(start, "%Y-%m-%d"):
                is_true = False

            if next_end <= next_start:
                is_true = False

            if (is_true) & (i <= 5):
                i = i + 1
                _start = datetime.strftime(next_start, "%Y-%m-%d")
                _end = datetime.strftime(next_end, "%Y-%m-%d")
                time.sleep(sleep_time)
            else:
                is_true = False

    merged_df = pd.concat(arr)
    merged_df = merged_df.drop_duplicates(keep="first")
    merged_df = merged_df.sort_index()

    merged_df["open"] = merged_df["open"].astype(float)
    merged_df["high"] = merged_df["high"].astype(float)
    merged_df["low"] = merged_df["low"].astype(float)
    merged_df["close"] = merged_df["close"].astype(float)
    merged_df["volume"] = merged_df["volume"].astype(float)

    return merged_df
