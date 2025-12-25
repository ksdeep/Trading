from GetFreshMarketData import *

def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def roc(series, period):
    return series.pct_change(periods=period)

def atr(df, period=14):
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr


def get_position_for_symbol(lv_symbol : str, lv_equity : float,
                            lv_initial_cash : float, lv_current_cash : float, 
                            lv_df_ohlc : pd.DataFrame, lv_capital_fraction : float,
                            lf_dt_current_day : datetime, lv_position_size_type : int,
                            lv_existing_positions : dict, lv_reason : str ='', 
                            lv_atr_factor =2) -> dict:
    """
    lv_symbol : str : stock symbol for which position to be calculated
    lv_equity : float : current equity value
    lv_initial_cash : float : initial cash value at start of backtest
    lv_current_cash : float : current available cash value
    lv_df_ohlc : pd.DataFrame : dataframe containing ohlc data for the symbol
    lv_capital_fraction : float : fraction of capital to be used for position sizing
    lf_dt_current_day : datetime : current date for which position is to be calculated
    lv_position_size_type : int : type of position sizing method
    lv_atr_factor : int : factor to be used for volatility based position sizing

    Determine the number of shares to buy for a given symbol based on equity, cash, and position sizing rules.
    1 -  Fixed position size based on initial cash and capital fraction.
    2 -  Variable position size based on current equity and capital fraction.
    3 -  Check if there is existing positions then only take 1/2 of existing position 
    4 -  Volatility based position sizing , OHLC should have ATR column precomputed.
    allways check if there is sufficient cash available to take the position.

    Retuns a dictionary with number of shares to buy and reason.

    """
    if lv_position_size_type == 1:
        lv_position_value = lv_initial_cash * lv_capital_fraction
        lv_num_shares = math.floor(lv_position_value / lv_df_ohlc.loc[lf_dt_current_day,['open','close','high','low']].mean())
        lv_reason = f'Fixed position size based on initial cash {lv_initial_cash} and capital fraction {lv_capital_fraction}.' +  lv_reason
        if lv_num_shares * lv_df_ohlc.loc[lf_dt_current_day,'high'] > lv_current_cash:
            lv_num_shares = math.floor(lv_current_cash / lv_df_ohlc.loc[lf_dt_current_day,['open','close','high','low']].mean())
            lv_reason = f'Adjusted position size based on available cash {lv_current_cash} and capital fraction {lv_capital_fraction}' +  lv_reason
            if lv_num_shares * lv_df_ohlc.loc[lf_dt_current_day,'high'] > lv_current_cash:
                lv_num_shares = 0
                lv_reason = f'No sufficient cash available to take position.' +  lv_reason
        return {'num_shares': lv_num_shares, 'reason': lv_reason}
    elif lv_position_size_type == 2:
        lv_position_value = lv_equity * lv_capital_fraction
        lv_num_shares = math.floor(lv_position_value / lv_df_ohlc.loc[lf_dt_current_day,['open','close','high','low']].mean())
        lv_reason = f'Variable position size based on current equity {lv_equity} and capital fraction {lv_capital_fraction}' +  lv_reason
        if lv_num_shares * lv_df_ohlc.loc[lf_dt_current_day,'high'] > lv_current_cash:
            lv_num_shares = math.floor(lv_current_cash / lv_df_ohlc.loc[lf_dt_current_day,['open','close','high','low']].mean())
            lv_reason = f'Adjusted position size based on available cash {lv_current_cash} and capital fraction {lv_capital_fraction}' +  lv_reason
            if lv_num_shares * lv_df_ohlc.loc[lf_dt_current_day,'high'] > lv_current_cash:
                return get_position_for_symbol(lv_symbol, lv_equity, lv_initial_cash, lv_current_cash, 
                                       lv_df_ohlc, lv_capital_fraction, lf_dt_current_day, 1, 
                                       lv_existing_positions, lv_reason= f'No sufficient cash available to take position. Switching to fixed position sizing.' )
        return {'num_shares': lv_num_shares, 'reason': lv_reason}
    elif lv_position_size_type == 3:
        for key, existing_position_details in lv_existing_positions.items():
            existing_position_symbol = existing_position_details['symbol']
            if existing_position_symbol == lv_symbol:
                existing_num_shares = existing_position_details['num_shares']
                lv_num_shares = math.floor(existing_num_shares / 2)
                lv_reason = f'Existing position found for {lv_symbol}. Taking half of existing position size {existing_num_shares}' +  lv_reason
                if lv_num_shares * lv_df_ohlc.loc[lf_dt_current_day,'high'] > lv_current_cash:
                    lv_num_shares = math.floor(lv_current_cash / lv_df_ohlc.loc[lf_dt_current_day,['open','close','high','low']].mean())
                    lv_reason = f'Adjusted position size based on available cash {lv_current_cash} and existing position size.' +  lv_reason
                    if lv_num_shares * lv_df_ohlc.loc[lf_dt_current_day,'high'] > lv_current_cash:
                        return get_position_for_symbol(lv_symbol, lv_equity, lv_initial_cash, lv_current_cash, 
                                       lv_df_ohlc, lv_capital_fraction, lf_dt_current_day, 1, 
                                       lv_existing_positions, lv_reason= f'No sufficient cash available to take position. Switching to fixed position sizing.' )
                return {'num_shares': lv_num_shares, 'reason': lv_reason}
        # no existing position found
        # fall back to fixed position sizing, calling the same function with type 1
        return get_position_for_symbol(lv_symbol, lv_equity, lv_initial_cash, lv_current_cash, 
                                       lv_df_ohlc, lv_capital_fraction, lf_dt_current_day, 1, 
                                       lv_existing_positions, lv_reason='No existing position found so switching to fixed position sizing.')
    elif lv_position_size_type == 4:
        lv_atr_value = lv_df_ohlc.loc[lf_dt_current_day,'atr']
        if 'atr' not in lv_df_ohlc.columns:
            return get_position_for_symbol(lv_symbol, lv_equity, lv_initial_cash, 
                                           lv_current_cash, lv_df_ohlc, lv_capital_fraction, 
                                           lf_dt_current_day, 1, lv_existing_positions, lv_reason='No ATR data found so switching to fixed position sizing.')
        elif lv_atr_value < 0.1:
            return get_position_for_symbol(lv_symbol, lv_equity, lv_initial_cash, 
                                           lv_current_cash, lv_df_ohlc, lv_capital_fraction, 
                                           lf_dt_current_day, 1, lv_existing_positions, 
                                           lv_reason=f'ATR is <0.1 for {lv_symbol} on {lf_dt_current_day} so switching to fixed position sizing.')   
        else:
            lv_num_shares = math.floor((lv_equity * lv_capital_fraction) / (lv_atr_value * lv_atr_factor))
            lv_reason = f'Volatility based position sizing using ATR value {lv_atr_value} and factor {lv_atr_factor}' +  lv_reason
            if lv_num_shares * lv_df_ohlc.loc[lf_dt_current_day,'high'] > lv_current_cash:
                
                position_size_dict = get_position_for_symbol(lv_symbol, lv_equity, lv_initial_cash, lv_current_cash, 
                                                    lv_df_ohlc, lv_capital_fraction/10, lf_dt_current_day, 4, 
                                                    lv_existing_positions, lv_reason=f'ATR based position sizing but no sufficient cash available to take position, so reducing capital fraction {lv_capital_fraction=} by 10 times to take position.')
                lv_num_shares = position_size_dict['num_shares']
                lv_reason = position_size_dict['reason']                
                if lv_num_shares * lv_df_ohlc.loc[lf_dt_current_day,'high']> lv_current_cash:
                    return get_position_for_symbol(lv_symbol, lv_equity, lv_initial_cash, lv_current_cash, 
                                                    lv_df_ohlc, lv_capital_fraction, lf_dt_current_day, 1, 
                                                    lv_existing_positions, lv_reason='ATR based position sizing but no sufficient cash available to take position so switching to fixed position sizing.')
            return {'num_shares': lv_num_shares, 'reason': lv_reason}
    else:
        raise ValueError("Invalid position size type. Must be 1, 2, 3, or 4.")

def rank_candidate_signals(lv_pending_buy: dict,
                           lv_trading_date: datetime,
                           lv_data : dict) -> dict:
    """
    Rank the candidate buy signals based on their rank value on the trading date.
    lv_pending_buy : dict : dictionary of pending buy signals
    lv_trading_date : datetime : date on which ranking is to be done
    returns a sorted list of dictionaries based on rank value
    """
    for key, value in lv_pending_buy.items():
        sym = value['symbol']
        df = lv_data[sym]
        if lv_trading_date in df.index:
            row = df.loc[lv_trading_date]
            value['score'] = row['rank']
        else:
            value['score'] = -np.inf  # assign a very low score if data not available

    lv_sorted_pending_buy = dict(sorted(lv_pending_buy.items(), key=lambda item: item[1]["score"], reverse=True))
    lv_sorted_pending_buy_unique = {}
    seen = set()
    for key, value in lv_sorted_pending_buy.items():
        sym = value['symbol']
        if sym not in seen:
            lv_sorted_pending_buy_unique[key] = value
            seen.add(sym)
    
    return lv_sorted_pending_buy_unique
        







positions = {}      # symbol -> position dict
trades = {}  
pending_buy = {}
pending_sell = {} 
all_signals = {}
all_positions = {}
data = {}



symbols = ['RELIANCE', 'TCS', 'WIPRO', 'HDFCBANK', 'TITAN']
start_back_test = datetime(2007, 1, 1)
end_back_test = datetime(2024, 12, 31)
max_positions = 8
capital_fraction = 0.10
commission = 0.005
annual_interest = 0.06
trading_days = 365
stop_loss_pct = 0.10 # stop loss percentage
which_days_to_trade = [0] # 0: Monday 6 : Sunday. list all days you would like to trade. 
signal_strength_checked_on_trading_day = True # mark false if singal strength to be check during generationan and not during placing orders 
position_szie_type = 1   # 1 - fixed based on initial cash and capital fraction, 2 - variable based on current equity and capital fraction, 3 - based on existing position size, 4 - volatility based position sizing using ATR.
allow_multiple_position_for_same_stock = False # mark true if multiple entries allowed else false.

initial_cash = 100000
equity_value = initial_cash

cash = initial_cash
data_start_date = start_back_test - timedelta(365*2)


dates = set()
for sym in symbols:
    df = pd.read_csv(STOCK_DIR / f"{sym}.csv", parse_dates=['date'])
    df = df[['date', 'open', 'high', 'low', 'close', 'volume']]
    df = df.sort_values('date').set_index('date')
    df = df.loc[((df.index>=data_start_date) & (df.index<=end_back_test)),:]

    df['ema50'] = ema(df['close'], 50)
    df['ema100'] = ema(df['close'], 100)
    df['ema200'] = ema(df['close'], 200)

    df['rank'] = roc(df['close'], 5)
    df['atr'] = atr(df, period=14)

    data[sym] = df
    if len(dates)==0:
        dates = dates.union(df.index.date)
    else:
        dates = dates.intersection(set(df.index.date))


dates =  pd.Series(list(dates)).sort_values().to_list()


cash_qeuality_history = {}
current_day = start_back_test
# note - signal generation happens everyday but trade execution happens only on specified days of week. 
# also the signals are generted frist and then exectuted on next trading day. signals are never executed on the same day.
day_activity = ''
while current_day <= end_back_test:
    if current_day.date() not in dates: # market not open no action. 
        # print(f'not a trading day {current_day}')        
        daily_interest = cash * annual_interest/trading_days
        cash = cash + daily_interest
        equity_value = equity_value +  daily_interest
        day_activity = 'Market closed - no action'
        cash_qeuality_history[current_day.date()] = {'cash': cash, 'equity': equity_value, 'activity': day_activity}
        current_day += timedelta(days=1)
        continue
    day_activity = ''
    # check if the current day is a trade execution day then execute the buy and sell signals. 
    if current_day.weekday() in which_days_to_trade and current_day != start_back_test :
        # print(current_day,'Trade execution day')
        if len(pending_sell)>0: # check if there are any item pending for sell
            # look through all pending sell items check if they have any matching position and trade then peform the reverse trade to close the trade. 
            for key, value in pending_sell.items():
                # write the code here after data structure of pending sell and position is ready. 
                # after sell transaction need to increase the cash amount. 
                # after sell transaction need to remove the position from positions dict.
                # after sell traction need to put the trade in completed trades list.
                sell_symbol = value['symbol']
                position_to_remove = [] # to handle multiple positions for same stock if allowed.
                for position_key, position_details in positions.items():
                    position_symbol = position_details['symbol']
                    if position_symbol == sell_symbol:
                        position_to_remove.append(position_key)
                        buy_price = position_details['buy_price']
                        num_shares = position_details['num_shares']
                        sell_price = data[sell_symbol].loc[current_day,['close','open','high','low']].mean()
                        position_value_at_exit = num_shares * sell_price * (1 - commission)
                        profit_loss_amount = position_value_at_exit - position_details['position_cost']
                        profit_loss_pct = profit_loss_amount / position_details['position_cost']
                        cash = cash + position_value_at_exit    
                        trades[position_key]['position_value_at_exit'] = position_value_at_exit
                        trades[position_key]['sell_date'] = current_day.date()
                        trades[position_key]['sell_price'] = sell_price 
                        trades[position_key]['pct_change_in_price'] = (sell_price - buy_price) / buy_price
                        trades[position_key]['profit_loss_amount'] = profit_loss_amount
                        trades[position_key]['profit_loss_pct'] = profit_loss_pct
                        trades[position_key]['number_of_days_held'] = (current_day.date() - position_details['buy_date']).days
                        trades[position_key]['trade_completed'] = True
                        trades[position_key]['type_of_exit'] = value['signal_type']
                        value['signal_processed_message'] = f'Sell executed for {num_shares} shares at price {sell_price}'
                        value['signal_processed'] = True
                        day_activity = day_activity + f'; Date {current_day.date()} Sell executed on symbol {sell_symbol} for {num_shares} shares at price {sell_price:.2f}, postion value at exit {position_value_at_exit:.2f}, total P/L: {profit_loss_amount:.2f} ({profit_loss_pct*100:.2f}%)'
                for pos_key in position_to_remove:
                    del positions[pos_key]
                all_signals[key] = value
            pending_sell.clear()        
        if len(pending_buy)>0: #check if there are items to buy and execute them. 
            # look through all the pending buy items and check if we have capacity to buy more positions.   
            # after buy transaction need to decrease the cash amount.
            # need to check the signal strength on the day of trade execution if the flag is set. 
            # need to add the position to positions dict.
            # decide how much position to buy based on capital fraction and available cash and position_size_fixed flag. if fixed then buy fixed amount based  on initial cash. 
            # else buy based on current equity * capital fraction. if this then need to check if cash is available to buy that amount.
            pending_buy = rank_candidate_signals(pending_buy, current_day, data)
            for key, value in pending_buy.items():
                # write the code here after data structure of pending buy and position is ready. 
                if len(positions)>=max_positions:
                    value['signal_processed_message'] = 'Not considered - max positions reached'
                    value['signal_processed'] = True
                else:
                    position_size_dict = get_position_for_symbol(
                        lv_symbol = value['symbol'],
                        lv_equity = equity_value,
                        lv_initial_cash = initial_cash, 
                        lv_current_cash = cash,
                        lv_df_ohlc = data[value['symbol']],
                        lv_capital_fraction = capital_fraction,    
                        lf_dt_current_day = current_day,
                        lv_position_size_type = position_szie_type,
                        lv_existing_positions = positions,  
                        lv_reason = '')
                    num_shares_to_buy = position_size_dict['num_shares']
                    reason_for_position_size = position_size_dict['reason']
                    
                    print(f"On {current_day.date()} for symbol {value['symbol']} determined position size to buy is {num_shares_to_buy} shares. Reason: {reason_for_position_size}")

                    position_cost = num_shares_to_buy * data[value['symbol']].loc[current_day,['close','open','high','low']].mean() * (1 + commission)
                    if num_shares_to_buy >0 and position_cost <= cash:
                        # proceed with buy
                        cash = cash - position_cost
                        position_key = f"{value['symbol']}_{current_day.strftime('%Y%m%d')}"
                        positions[position_key] = {'symbol': value['symbol'],
                                        'num_shares': num_shares_to_buy,
                                        'buy_date': current_day.date(),
                                        'buy_price': data[value['symbol']].loc[current_day,['close','open','high','low']].mean(),
                                        'position_cost': position_cost,
                                        'position_reason': reason_for_position_size
                                        }
                        trades[position_key] = {'symbol': value['symbol'],
                                        'num_shares': num_shares_to_buy,
                                        'buy_date': current_day.date(),
                                        'buy_price': data[value['symbol']].loc[current_day,['close','open','high','low']].mean(),
                                        'sell_date': None,
                                        'sell_price': None,
                                        'pct_change_in_price': None,
                                        'profit_loss_amount': None,
                                        'profit_loss_pct': None,
                                        'position_value_at_entry': position_cost,
                                        'position_value_at_exit': None,
                                        'number_of_days_held': None,
                                        'trade_completed': False,
                                        'type_of_exit': None,
                                        'type_of_trade': 'LONG'
                                        }
                        day_activity = day_activity + f'; Date {current_day.date()} Buy executed on symbol {value["symbol"]} for {num_shares_to_buy} shares at price {data[value["symbol"]].loc[current_day,["close","open","high","low"]].mean():.2f},position cost {position_cost:.2f}'
                        all_positions[position_key] = positions[position_key]
                        value['signal_processed_message'] = f'Buy executed for {num_shares_to_buy} shares at price {data[value["symbol"]].loc[current_day,["close","open","high","low"]].mean()}'
                        value['signal_processed'] = True
                    else:
                        value['signal_processed_message'] = 'Not executed - insufficient cash to take position'
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
            day_activity = day_activity + f'; Date {current_day.date()} Sell signal generated for symbol {sym} as close {row["close"]:.2f} < ema100 {row["ema100"]:.2f}'
            
        if (row['close'] > row['ema50'] and row['ema100'] > row['ema200']):
            pending_buy[key] = {'symbol': sym,
                               'signal_date': current_day.date(),
                                'signal_type': 'BUY',
                                'signal_processed': False,
                                'signal_processed_message': 'Generated'
                               }
            day_activity = day_activity + f'; Date {current_day.date()} Buy signal generated for symbol {sym} as close {row["close"]:.2f} > ema50 {row["ema50"]:.2f} and ema100 {row["ema100"]:.2f} > ema200 {row["ema200"]:.2f}'
    for key, value in positions.items():
        buy_price = value['buy_price']
        symbol = value['symbol']
        current_price = data[symbol].loc[current_day, 'low']
        pct_change = (current_price - buy_price) / buy_price 
        if pct_change <= -stop_loss_pct:
            pending_sell[key] = {'symbol': value['symbol'],
                                'signal_date': current_day.date(),
                                'signal_type': 'STOP_LOSS_SELL',
                                'signal_processed': False,
                                'signal_processed_message': 'Generated due to stop loss'
                                }
            day_activity = day_activity + f'; Date {current_day.date()} Stop Loss Sell signal generated for symbol {symbol} as price dropped to {current_price:.2f} which is {pct_change*100:.2f}% below buy price {buy_price:.2f}'


    # Update cash and equity value at end of day
    cash = cash + cash * annual_interest/trading_days
    equity_value = cash + sum([ pos['num_shares'] * data[pos['symbol']].loc[current_day,'close'] for pos in positions.values() ])
    cash_qeuality_history[current_day.date()] = {'cash': cash, 'equity': equity_value, 'activity': day_activity.strip('; ')}
    current_day += timedelta(days=1)
# exit all remaining positions at the end of backtest period
day_activity = ''
last_back_test_date = datetime(dates[-1].year,dates[-1].month,dates[-1].day)
for position_key, position_details in positions.items():
    sell_symbol = position_details['symbol']
    buy_price = position_details['buy_price']
    num_shares = position_details['num_shares']
    sell_price = data[sell_symbol].loc[last_back_test_date,['close','open','high','low']].mean()
    position_value_at_exit = num_shares * sell_price * (1 - commission)
    profit_loss_amount = position_value_at_exit - position_details['position_cost']
    profit_loss_pct = profit_loss_amount / position_details['position_cost']
    cash = cash + position_value_at_exit    
    trades[position_key]['position_value_at_exit'] = position_value_at_exit
    trades[position_key]['sell_date'] = last_back_test_date.date()
    trades[position_key]['sell_price'] = sell_price 
    trades[position_key]['pct_change_in_price'] = (sell_price - buy_price) / buy_price
    trades[position_key]['profit_loss_amount'] = profit_loss_amount
    trades[position_key]['profit_loss_pct'] = profit_loss_pct
    trades[position_key]['number_of_days_held'] = (last_back_test_date.date() - position_details['buy_date']).days
    trades[position_key]['trade_completed'] = True
    trades[position_key]['type_of_exit'] = 'END_OF_BACKTEST_SELL'
    day_activity = day_activity + f'; Date {last_back_test_date.date()} Final Sell executed on symbol {sell_symbol} for {num_shares} shares at price {sell_price:.2f}, postion value at exit {position_value_at_exit:.2f}, total P/L: {profit_loss_amount:.2f} ({profit_loss_pct*100:.2f}%)'
    all_signals[position_key] = {'symbol': sell_symbol,
                                'signal_date': last_back_test_date.date(),
                                'signal_type': 'END_OF_BACKTEST_SELL',
                                'signal_processed': True,
                                'signal_processed_message': f'Final sell executed for {num_shares} shares at price {sell_price}'
                                }
cash_qeuality_history[last_back_test_date.date()] = {'cash': cash, 'equity': cash, 'activity': day_activity.strip('; ')}



pd.DataFrame(all_signals).T.to_csv('all_signals.csv')
pd.DataFrame(all_positions).T.to_csv('all_positions.csv')
pd.DataFrame(trades).T.to_csv('trades.csv')
pd.DataFrame(cash_qeuality_history).T.to_csv('cash_quality_history.csv')
print(f'Final equity value: {equity_value}, Final cash value: {cash}')