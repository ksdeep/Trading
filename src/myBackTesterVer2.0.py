from GetFreshMarketData import *

def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def roc(series, period):
    return series.pct_change(periods=period)

symbols = ['RELIANCE', 'TCS', 'WIPRO', 'HDFCBANK', 'TITAN']




initial_cash = 100000
start_back_test = datetime(2010, 1, 1)
end_back_test = datetime(2020, 12, 31)
cash = initial_cash
positions = {}      # symbol -> position dict
trades = {}  

pending_buy = {}
pending_sell = {} 

all_signals = {}
data = {}


max_positions = 4
capital_fraction = 0.10
commission = 0.005
annual_interest = 0.06
trading_days = 365
which_days_to_trade = [1] # 0: Sunday 6 : Satuday. list all days you would like to trade. 
signal_strength_checked_on_trading_day = True # mark false if singal strength to be check during generationan and not during placing orders 
position_size_fixed = False # mark true if position size is fixed else variable based on available equity value. 
allow_multiple_position_for_same_stock = False # mark true if multiple entries allowed else false.




data_start_date = start_back_test - timedelta(365*2)

for sym in symbols:
    df = pd.read_csv(STOCK_DIR / f"{sym}.csv", parse_dates=['date'])
    df = df[['date', 'open', 'high', 'low', 'close', 'volume']]
    df = df.sort_values('date').set_index('date')
    df = df.loc[((df.index>=data_start_date) & (df.index<=end_back_test)),:]

    df['ema50'] = ema(df['close'], 50)
    df['ema100'] = ema(df['close'], 100)
    df['ema200'] = ema(df['close'], 200)

    df['roc5'] = roc(df['close'], 5)

    data[sym] = df

dates = sorted(set().union(*[pd.Series(df.index).apply(lambda x : x.date()) for df in data.values()]))



current_day = start_back_test
# note - signal generation happens everyday but trade execution happens only on specified days of week. 
# also the signals are generted frist and then exectuted on next trading day. signals are never executed on the same day.
while current_day <= end_back_test:
    cash = cash + cash * annual_interest/trading_days
    if current_day.date() not in dates: # market not open no action. 
        print(f'not a trading day {current_day}')        
        current_day += timedelta(days=1)
        continue
    # check if the current day is a trade execution day then execute the buy and sell signals. 
    if current_day.weekday() in which_days_to_trade and current_day != start_back_test :
        print(current_day,'Trade execution day')
        if len(pending_sell)>0: # check if there are any item pending for sell
            # look through all pending sell items check if they have any matching position and trade then peform the reverse trade to close the trade. 
            for key, value in pending_sell.items():
                # write the code here after data structure of pending sell and position is ready. 
                # after sell transaction need to increase the cash amount. 
                # after sell transaction need to remove the position from positions dict.
                # after sell traction need to put the trade in completed trades list.
                None
        
        if len(pending_buy)>0: #check if there are items to buy and execute them. 
            # look through all the pending buy items and check if we have capacity to buy more positions.   
            # after buy transaction need to decrease the cash amount.
            # need to check the signal strength on the day of trade execution if the flag is set. 
            # need to add the position to positions dict.
            # decide how much position to buy based on capital fraction and available cash and position_size_fixed flag. if fixed then buy fixed amount based  on initial cash. 
            # else buy based on current equity * capital fraction. if this then need to check if cash is available to buy that amount.
            # if allow_multiple_position_for_same_stock is false then need to check if we already have position in that stock.
            for key, value in pending_buy.items():
                # write the code here after data structure of pending buy and position is ready. 
                if len(positions)>=max_positions:
                    value['signal_processed_message'] = 'Not considered - max positions reached'
                else:
                    None
                value['signal_processed'] = True
                all_signals[key] = value

            pending_buy.clear()                
    # for all days need to check and generate buy and sell signals.

    for sym, df in data.items():
        current_day_df = df.loc[df.index.date == current_day.date()]
        if current_day_df.empty:
            continue
        key = f"{sym}_{current_day.strftime('%Y%m%d')}"
        row = current_day_df.iloc[0]
        if row['close'] < row['ema100']:
            pending_sell[key] = {'symbol': sym,
                                'signal_date': current_day.date(),
                                'signal_type': 'SELL',
                                'signal_processed': False,
                                'signal_processed_message': 'Generated'
                                }   
        if (row['close'] > row['ema50'] and row['ema100'] > row['ema200']):
            pending_buy[key] = {'symbol': sym,
                               'signal_date': current_day.date(),
                                'signal_type': 'BUY',
                                'signal_processed': False,
                                'signal_processed_message': 'Generated'
                               }

    current_day += timedelta(days=1)

pd.DataFrame(all_signals).T.to_csv('all_signals.csv')