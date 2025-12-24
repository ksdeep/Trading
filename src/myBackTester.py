from GetFreshMarketData import *

def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def roc(series, period):
    return series.pct_change(periods=period)

symbols = ['RELIANCE', 'TCS', 'WIPRO', 'HDFCBANK', 'TITAN']
data = {}



initial_cash = 100000
start = datetime(2010, 1, 1)
end = datetime(2020, 12, 31)
cash = initial_cash
positions = {}      # symbol -> position dict
active_trades = {}  # symbol -> trade dict
completed_trades = []

max_positions = 4
capital_fraction = 0.10
commission = 0.005
annual_interest = 0.06
trading_days = 365

pending_buy = {}
pending_sell = set()
signal_date = {}
data_start_date = start - timedelta(365*2)

for sym in symbols:
    df = pd.read_csv(STOCK_DIR / f"{sym}.csv", parse_dates=['date'])
    df = df[['date', 'open', 'high', 'low', 'close', 'volume']]
    df = df.sort_values('date').set_index('date')
    df = df.loc[((df.index>=data_start_date) & (df.index<=end)),:]

    df['ema50'] = ema(df['close'], 50)
    df['ema100'] = ema(df['close'], 100)
    df['ema200'] = ema(df['close'], 200)
    df['roc5'] = roc(df['close'], 5)

    data[sym] = df

dates = sorted(set().union(*[df.index for df in data.values()]))

last_week = None




for date in dates:
    if not (date >=start and date<=end):
        # dates outside back test dates. 
        continue
    current_week = date.isocalendar()[1]
    new_week = last_week is not None and current_week != last_week
    
    for sym, df in data.items():
        if date not in df.index:
            continue

        row = df.loc[date]
        pos = positions.get(sym)
        # SELL signal
        if pos and row['close'] < row['ema100']:
            pending_sell.add(sym)
            signal_date.setdefault(sym, date)
        
        # BUY signal
        if not pos:
            if (row['close'] > row['ema50'] and row['ema100'] > row['ema200'] and not np.isnan(row['roc5'])):
                pending_buy[sym] = {'score': row['roc5'],
                                    'signal_week': current_week
                                    }
                signal_date.setdefault(sym, date)


    if new_week:
        for sym in list(pending_sell):
            if sym in positions:
                exit_price = data[sym].loc[date, 'open']
                trade = active_trades.pop(sym)

                trade['exit_date'] = date
                trade['exit_price'] = exit_price
                trade['pnl'] = (
                    (exit_price - trade['entry_price'])
                    * trade['size']
                    - commission * exit_price * trade['size']
                )

                cash += exit_price * trade['size'] * (1 - commission)
                completed_trades.append(trade)

                del positions[sym]
        pending_sell.clear()

        eligible = {sym: info for sym, info in pending_buy.items() if info['signal_week'] < current_week}
        ranked = sorted(eligible.items(),
                        key=lambda x: x[1]['score'],
                        reverse=True
                        )
        slots = max_positions - len(positions)
        for sym, score in ranked[:slots]:
            if sym not in positions:
                open_price = data[sym].loc[date, 'open']
                size = math.floor(
                    cash * capital_fraction / open_price
                )
                if size > 0:
                    cost = open_price * size * (1 + commission)
                    if cost <= cash:
                        cash -= cost
                        positions[sym] = size
                        active_trades[sym] = {
                            'symbol': sym,
                            'signal_date': signal_date[sym],
                            'execution_date': date,
                            'entry_price': open_price,
                            'size': size,
                        }
        for sym, _ in ranked[:slots]:
            pending_buy.pop(sym, None)
            signal_date.pop(sym, None)
    last_week = current_week

trades_df = pd.DataFrame(completed_trades)
print(trades_df)

trades_df.to_csv("custom_weekly_backtest_trades.csv", index=False)

print("Final cash:", round(cash, 2))
print("Total trades:", len(trades_df))

