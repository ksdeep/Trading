import backtrader as bt
import pandas as pd
import math
from datetime import datetime, timedelta


# ==============================
# SIGNAL-CENTRIC PORTFOLIO STRATEGY
# ==============================
class WeeklyRankedPortfolio(bt.Strategy):

    params = dict(
        max_positions=4,
        capital_fraction=0.10,      # 10% per trade
        annual_interest=0.06,
        days_in_year=365
    )

    def __init__(self):

        # Indicators per stock
        self.inds = {}

        # Signal & execution state
        self.pending_buy = {}          # data -> ranking score
        self.pending_sell = set()      # data
        self.signal_date = {}
        self.last_week = None

        # Trade ledger
        self.active_trade = {}         # data -> open trade
        self.trades = []               # completed trades

        for data in self.datas:
            self.inds[data] = dict(
                ema50=bt.ind.EMA(data.close, period = 50),
                ema100=bt.ind.EMA(data.close, period = 100),
                ema200=bt.ind.EMA(data.close, period = 200),
                roc5=bt.ind.ROC(data.close, period = 5)
            )

    # ==============================
    # ORDER HANDLING (ENTRY & EXIT)
    # ==============================
    def notify_order(self, order):

        if order.status in [order.Submitted, order.Accepted]:
            return

        data = order.data
        exec_date = bt.num2date(order.executed.dt).date()

        if order.status == order.Completed:

            # ---------------- BUY ----------------
            if order.isbuy():
                self.active_trade[data] = {
                    'symbol': data._name,
                    'signal_date': self.signal_date.get(data),
                    'execution_date': exec_date,
                    'entry_price': order.executed.price,
                    'size': order.executed.size,
                    'entry_ema50': self.inds[data]['ema50'][0],
                    'entry_ema100': self.inds[data]['ema100'][0],
                    'entry_ema200': self.inds[data]['ema200'][0],
                }

            # ---------------- SELL ----------------
            elif order.issell():
                trade = self.active_trade.pop(data, None)
                if trade is None:
                    return

                trade.update({
                    'exit_date': exec_date,
                    'exit_price': order.executed.price,
                    'exit_ema50': self.inds[data]['ema50'][0],
                    'exit_ema100': self.inds[data]['ema100'][0],
                    'exit_ema200': self.inds[data]['ema200'][0],
                })

                trade['pnl'] = (
                    (trade['exit_price'] - trade['entry_price'])
                    * trade['size']
                )

                trade['equity_after_trade'] = self.broker.getvalue()

                self.trades.append(trade)

    # ==============================
    # MAIN STRATEGY LOOP
    # ==============================
    def next(self):

        # -------- Apply interest on idle cash --------
        cash = self.broker.getcash()
        if cash > 0:
            daily_rate = self.p.annual_interest / self.p.days_in_year
            self.broker.add_cash(cash * daily_rate)

        # -------- Detect week change --------
        current_date = self.datetime.date()
        current_week = current_date.isocalendar()[1]
        new_week = self.last_week is not None and current_week != self.last_week

        # -------- Generate signals (no execution) --------
        for data in self.datas:
            pos = self.getposition(data)

            # SELL SIGNAL
            if pos:
                if data.close[0] < self.inds[data]['ema100'][0]:
                    self.pending_sell.add(data)
                    if data not in self.signal_date:
                        self.signal_date[data] = current_date

            # BUY SIGNAL
            else:
                if (
                    data.close[0] > self.inds[data]['ema50'][0]
                    and self.inds[data]['ema100'][0] > self.inds[data]['ema200'][0]
                ):
                    roc = self.inds[data]['roc5'][0]
                    if not math.isnan(roc):
                        self.pending_buy[data] = roc
                        if data not in self.signal_date:
                            self.signal_date[data] = current_date

        # -------- Execute once per new week --------
        if new_week:

            # Execute sells first
            for data in list(self.pending_sell):
                if self.getposition(data):
                    self.close(data=data)
            self.pending_sell.clear()

            # Rank buys by weekly momentum
            ranked = sorted(
                self.pending_buy.items(),
                key=lambda x: x[1],
                reverse=True
            )

            open_positions = sum(
                1 for d in self.datas if self.getposition(d)
            )

            slots = self.p.max_positions - open_positions

            for data, score in ranked[:slots]:
                if not self.getposition(data):
                    cash = self.broker.getcash()
                    size = math.floor(
                        cash * self.p.capital_fraction / data.close[0]
                    )
                    if size > 0:
                        self.buy(data=data, size=size)

            self.pending_buy.clear()
            self.signal_date.clear()

        self.last_week = current_week


# ==============================
# BACKTEST RUNNER
# ==============================
if __name__ == "__main__":

    cerebro = bt.Cerebro()

    cerebro.addstrategy(WeeklyRankedPortfolio)

    cerebro.broker.setcash(100000)
    cerebro.broker.setcommission(
        commission=0.005,
        commtype=bt.CommInfoBase.COMM_PERC
    )

    symbols = ['RELIANCE', 'TCS', 'WIPRO', 'HDFCBANK', 'TITAN']

    start = datetime(2008, 1, 1)
    end = datetime(2020, 12, 31)

    for sym in symbols:
        df = pd.read_csv(
            rf"C:\Users\ksdee\Documents\Trading\Data\stocks\{sym}.csv"
        )
        df = df[['date', 'open', 'high', 'low', 'close', 'volume']]
        df['date'] = pd.to_datetime(df['date'])
        df['openinterest'] = 0
        df.columns = ['datetime', 'open', 'high', 'low', 'close', 'volume', 'openinterest']
        df = df[(df.datetime >= start) & (df.datetime <= end)]
        df.set_index('datetime', inplace=True)

        cerebro.adddata(bt.feeds.PandasData(dataname=df), name=sym)

    print("Starting equity:", cerebro.broker.getvalue())
    strat = cerebro.run()[0]
    print("Final equity:", cerebro.broker.getvalue())

    trades_df = pd.DataFrame(strat.trades)
    trades_df.to_csv(TEMP/"weekly_signal_trades.csv", index=False)
