"""Visualization functions for backtesting results.

Provides interactive Plotly charts and compact matplotlib thumbnails.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_basic(
    history: pd.DataFrame,
    params,
    title: str = "",
    w: int = 1024,
    h: int = 900,
    show_signals: bool = False,
) -> None:
    """Generate an interactive Plotly chart with price, positions, and balance.

    Args:
        history: DataFrame from Backtester.history() with close, balance, equity, long, short, signal columns.
        params: Backtester Params instance.
        title: Optional plot title.
        w: Plot width in pixels.
        h: Plot height in pixels.
        show_signals: If True, add a third subplot showing the signal values.
    """
    _s = "&#36;"
    hdf = history

    balance_start = params.initial_balance
    balance_end = hdf.iloc[-1]["balance"]
    balance_final_str = str(f"{_s}{hdf.iloc[-1]['balance']:,.2f}")
    balance_final_perc = round(((balance_end / balance_start) - 1) * 100, 2)

    describe = f"<sup>({balance_final_perc}%)</sup>"
    title = f"{title}<br>{describe}"

    subplot_titles = ("Price", "Result", "")
    row_heights = [0.4, 0.3, 0.2] if show_signals else [0.4, 0.6]
    rows = 3 if show_signals else 2

    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=row_heights, subplot_titles=subplot_titles)  # fmt: off

    # Row 1: Price with long/short overlays
    fig.add_trace(go.Scatter(x=hdf.index, name="Close Price", y=hdf['close'], line=dict(color="grey")), row=1, col=1)  # fmt: off
    fig.add_trace(go.Scatter(x=hdf.index, name="Long", y=np.where((hdf["long"]) | (hdf["long"].shift(1)), hdf['close'], None), line_color="blue"), row=1, col=1)  # fmt: off
    fig.add_trace(go.Scatter(x=hdf.index, name="Short", y=np.where((hdf["short"]) | (hdf["short"].shift(1)), hdf['close'], None), line_color="red"), row=1, col=1)  # fmt: off

    # Row 2: Balance, equity, and buy-and-hold
    fig.add_trace(go.Scatter(x=hdf.index, name="Balance", y=hdf['balance'],  line=dict(color="green")), row=2, col=1)  # fmt: off
    fig.add_trace(go.Scatter(x=hdf.index, name="Equity", y=hdf['equity'], line=dict(color="orange", width=0.5), visible='legendonly'), row=2, col=1)  # fmt: off
    fig.add_trace(go.Scatter(x=hdf.index, name="Buy & Hold", y=hdf['buy_hold'], line=dict(color="grey", width=0.7, dash="dot"), visible='legendonly'), row=2, col=1)  # fmt: off

    # Row 3: Signal values (optional)
    if show_signals:
        fig.add_trace(go.Scatter(x=hdf.index, name="Signal", y=hdf['signal'], line=dict(color="black", width=1), mode="lines"), row=3, col=1)  # fmt: off

    fig.layout.annotations[0].text = "Price"
    fig.layout.annotations[0].font = dict(size=12)
    fig.layout.annotations[1].text = "Result"
    fig.layout.annotations[1].font = dict(size=12)

    fig.update_layout(width=w, height=h, title_text=title, font_color="blue", title_font_color="black", font=dict(size=11, color="Black"))  # fmt: off
    fig.update_layout(yaxis3=dict(range=[-1, 1], dtick=1))
    fig.update_layout(hovermode="x unified")
    fig.update_traces(xaxis="x2")

    # Footer with trading vs buy-and-hold results
    hold_final = round(hdf.iloc[-1]["buy_hold"], 2)
    hold_final_str = str(f"{_s}{hold_final:,.2f}")
    hold_perc = round(((hold_final / balance_start) - 1) * 100, 2)

    footer_result = f"     trading: {balance_final_str} ({balance_final_perc}%)<br>"
    footer_result += f"buy & hold: {hold_final_str} ({hold_perc}%)<br>"

    fig.add_annotation(text=footer_result, showarrow=False, x=0.0, y=0, xref="paper", yref="paper", xanchor="left", yanchor="bottom", xshift=-1, yshift=-60, font=dict(size=10, color="grey"), align="left")  # fmt: off
    fig.update_layout(margin=dict(l=30, r=10, t=50, b=100))

    fig.show()


def plot_thumbnail(
    history: pd.DataFrame,
    params,
    stats: pd.Series,
    title: str = "Backtest",
    w: int = 4,
    h: int = 1,
) -> None:
    """Display a compact matplotlib summary thumbnail.

    Args:
        history: DataFrame from Backtester.history().
        params: Backtester Params instance.
        stats: pd.Series from Backtester.stats().
        title: Label to display on the thumbnail.
        w: Figure width in inches.
        h: Figure height in inches.
    """
    hdf = history

    balance_start = params.initial_balance
    balance_end = hdf.iloc[-1]["balance"]
    balance_final_perc = round(((balance_end / balance_start) - 1) * 100, 2)

    color = "green" if balance_final_perc > 0 else "red"

    text = f"{balance_final_perc}%"
    text2 = ""

    if title:
        text = f"{title} {text}"

    plt.figure(figsize=(w, h))
    plt.plot(hdf.index, hdf["balance"], color=color)

    if isinstance(stats, pd.Series):
        max_drawdown = round(stats["Max. Drawdown [%]"], 2)
        avg_drawdown = round(stats["Avg. Drawdown [%]"], 2)
        text2 = text2 + f"\nmax dd:{max_drawdown}%\n avg dd:{avg_drawdown}"

    plt.annotate(text, xy=(0, 1), xycoords="axes fraction", size=10)
    plt.annotate(text2, xy=(0, 0.6), xycoords="axes fraction", color="grey", size=8)

    plt.axis("off")
    plt.show()
