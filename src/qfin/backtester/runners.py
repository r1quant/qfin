from qfin.backtester.backtester import Backtester


def bt_signal_change(dataset, leverage_column: str | None = None, **karg):
    bt = Backtester(dataset=dataset, **karg)

    for broker in bt.run():
        current_bar = broker.state.data.iloc[-1]
        previous_bar = broker.state.data.iloc[-2]
        leverage = current_bar[leverage_column] if leverage_column else None

        current_signal = current_bar["signal"]
        previous_signal = previous_bar["signal"]
        changed = current_signal != previous_signal

        if changed:
            if current_signal == 1:
                broker.buy(leverage=leverage)
            elif current_signal == -1:
                broker.sell(leverage=leverage)
            else:
                broker.close()

    return bt
