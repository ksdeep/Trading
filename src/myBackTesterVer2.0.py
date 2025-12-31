from GetFreshMarketData import *
import matplotlib.pyplot as plt
import logging
import pickle
import backtest_dashboard 
pd.set_option('future.no_silent_downcasting', True)


logging.basicConfig(
    filename=TEMP/'backTester.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
     force=True
)

logging.info("Starting backtester ver 2.0")


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

def get_last_day(lv_year:int, lv_month:int):
    # Go to the first day of the next month
    if lv_month == 12:
        next_month = datetime(lv_year + 1, 1, 1)
    else:
        next_month = datetime(lv_year, lv_month + 1, 1)
    
    # Subtract one day
    return (next_month - timedelta(days=1))

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

def close_all_positions_at_end_of_backtest(lv_back_test_end_date: datetime = None,
                                           lv_positions: dict = None,
                                           lv_data: dict = None,
                                           lv_trades: dict = None,
                                           lv_cash : float = None,
                                           lv_all_signals : dict = None)-> tuple:
    """
    Close all open positions at the end of the backtest period.
    """
    global logger,commission
    logger.info(f'Closing all open positions at the end of backtest period')

    day_activity = ''
     
    for position_key, position_details in lv_positions.items():
        sell_symbol = position_details['symbol']
        buy_price = position_details['buy_price']
        num_shares = position_details['num_shares']
        sell_price = lv_data[sell_symbol].loc[lv_back_test_end_date,['close','open','high','low']].mean()
        position_value_at_exit = num_shares * sell_price * (1 - commission)
        profit_loss_amount = position_value_at_exit - position_details['position_cost']
        profit_loss_pct = profit_loss_amount / position_details['position_cost']
        lv_cash = lv_cash + position_value_at_exit    
        lv_trades[position_key]['position_value_at_exit'] = position_value_at_exit
        lv_trades[position_key]['sell_date'] = lv_back_test_end_date.date()
        lv_trades[position_key]['sell_price'] = sell_price 
        lv_trades[position_key]['pct_change_in_price'] = (sell_price - buy_price) / buy_price
        lv_trades[position_key]['profit_loss_amount'] = profit_loss_amount
        lv_trades[position_key]['profit_loss_pct'] = profit_loss_pct
        lv_trades[position_key]['number_of_days_held'] = (lv_back_test_end_date.date() - position_details['buy_date']).days
        lv_trades[position_key]['trade_completed'] = True
        lv_trades[position_key]['type_of_exit'] = 'END_OF_BACKTEST_SELL'
        day_activity = day_activity + f'; Date {lv_back_test_end_date.date()} Final END OF BACK TEST Sell executed on symbol {sell_symbol} for {num_shares} shares at price {sell_price:.2f}, postion value at exit {position_value_at_exit:.2f}, total P/L: {profit_loss_amount:.2f} ({profit_loss_pct*100:.2f}%)'
        lv_all_signals[position_key] = {'symbol': sell_symbol,
                                    'signal_date': lv_back_test_end_date.date(),
                                    'signal_type': 'END_OF_BACKTEST_SELL',
                                    'signal_processed': True,
                                    'signal_processed_message': f'Final sell executed for {num_shares} shares at price {sell_price}'
                                    }
    lv_positions.clear()
    return day_activity, lv_cash
    
def generate_back_test_performance_report(lv_trades: dict,
                                          lv_cash_equity_history: dict,
                                          lv_initial_cash: int,
                                          lv_start_date: datetime,
                                          lv_end_date: datetime)-> tuple:
    """
    Generate and log the overall performance report of the backtest.
    """
    global logger
    logger.info(f'Generating overall backtest performance report')
    
    cash_df = pd.DataFrame(lv_cash_equity_history).T
    trades_df = pd.DataFrame(lv_trades).T
    trades_df.buy_date = pd.to_datetime(trades_df.buy_date)
    trades_df.sell_date = pd.to_datetime(trades_df.sell_date)

    # --- 2. Calculate Equity & Drawdown Curves ---
    equity_curve = cash_df['equity']
    
    # Calculate Running Maximum
    running_max = equity_curve.cummax()

    # Calculate Drawdown (%)
    drawdown = (equity_curve - running_max) / running_max

    # --- 3. Calculate Performance Metrics ---
    
    # Time Calculations
    start_date = equity_curve.index[0]
    end_date = equity_curve.index[-1]
    total_days = (end_date - start_date).days
    years = total_days / 365.25

    # CAGR
    final_equity = equity_curve.iloc[-1]
    cagr = (final_equity / lv_initial_cash) ** (1 / years) - 1

    # Max Drawdown
    max_dd = drawdown.min()

    # Max Drawdown Duration (Time between new equity highs)
    highs = equity_curve[equity_curve == running_max].index
    if len(highs) > 1:
        max_dd_duration_days = pd.Series(highs).diff().dt.days.max()
    else:
        max_dd_duration_days = total_days
    
    # Daily Returns for Risk Metrics
    daily_returns = equity_curve.pct_change().dropna()
    
    # Sharpe Ratio (assuming 0% risk-free rate for simplicity)
    # Annualized by sqrt(252)
    sharpe_ratio = np.sqrt(252) * (daily_returns.mean() / daily_returns.std())

    # Sortino Ratio (punishes only negative volatility)
    negative_returns = daily_returns[daily_returns < 0]
    sortino_ratio = np.sqrt(252) * (daily_returns.mean() / negative_returns.std())

    # Trade Statistics
    total_trades = len(trades_df)
    winning_trades = trades_df[trades_df['profit_loss_amount'] > 0]
    losing_trades = trades_df[trades_df['profit_loss_amount'] <= 0]
    
    win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
    
    gross_profit = winning_trades['profit_loss_amount'].sum()
    gross_loss = abs(losing_trades['profit_loss_amount'].sum())
    profit_factor = gross_profit / gross_loss if gross_loss != 0 else np.inf
    
    avg_profit_per_trade = trades_df['profit_loss_amount'].mean()

    # --- 4. Print Results ---
    print("-" * 30)
    print("PERFORMANCE REPORT")
    print("-" * 30)
    print(f"Start Date:       {lv_start_date.date()}")
    print(f"End Date:         {lv_end_date.date()}")
    print(f"Duration:         {years:.2f} years")
    print(f"Initial Capital:  {lv_initial_cash:,.2f}")
    print(f"Final Equity:     {final_equity:,.2f}")
    print(f"CAGR:             {cagr:.2%}")
    print(f"Max Drawdown:     {max_dd:.2%}")
    print(f"CAR/MDD:          {cagr / -max_dd if max_dd != 0 else np.inf:.2f}")
    print(f"Max DD Duration:  {max_dd_duration_days:.0f} days")
    print(f"Sharpe Ratio:     {sharpe_ratio:.2f}")
    print(f"Sortino Ratio:    {sortino_ratio:.2f}")
    print("-" * 30)
    print("TRADE STATISTICS")
    print("-" * 30)
    print(f"Total Trades:     {total_trades}")
    print(f"Win Rate:         {win_rate:.2%}")
    print(f"Profit Factor:    {profit_factor:.2f}")
    print(f"Avg Trade P/L:    {avg_profit_per_trade:.2f}")

    # --- 4. Plotting (Fixed for the TypeError) ---
    # plt.figure(figsize=(12, 10))

    # # Subplot 1: Equity Curve
    # plt.subplot(2, 1, 1)
    # plt.plot(equity_curve.index, equity_curve.values, label='Equity', color='blue')
    # plt.title('Equity Curve')
    # plt.ylabel('Capital')
    # plt.grid(True, alpha=0.3)
    # plt.legend()

    # # Subplot 2: Drawdown Curve
    # plt.subplot(2, 1, 2)
    # # FIX: Use .index and .values explicitly to avoid type issues in fill_between
    # plt.plot(drawdown.index, drawdown.values, label='Drawdown', color='red')
    # # plt.fill_between(drawdown.index, drawdown.values, 0, color='red', alpha=0.3)
    
    # plt.title('Drawdown Curve')
    # plt.ylabel('Drawdown %')
    # plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
    # plt.grid(True, alpha=0.3)
    # plt.legend()

    # plt.tight_layout()
    # plt.savefig('backtest_report.png')
    # plt.show()
    return equity_curve,drawdown

def generate_monthly_and_yearly_performance(lv_eqity_cash_history : pd.DataFrame, lv_initial_equity_target:float):
    lv_eqity_cash_history = lv_eqity_cash_history.copy()

    lv_eqity_cash_history = lv_eqity_cash_history.set_index('date')

    initial_actual = lv_eqity_cash_history['equity'].iloc[0]
    scaling_factor = lv_initial_equity_target / initial_actual
    lv_eqity_cash_history['equity_adj'] = lv_eqity_cash_history['equity'] * scaling_factor

    monthly_equity = lv_eqity_cash_history['equity_adj'].resample('ME').last()

    # 4. Calculate Monthly Return %
    # The first month's return is calculated against the $100,000 starting point
    monthly_returns = monthly_equity.pct_change().infer_objects(copy=False) 
    monthly_returns.iloc[0] = (monthly_equity.iloc[0] / lv_initial_equity_target - 1) 

    # 5. Prepare data for the Pivot Table
    returns_df = monthly_returns.reset_index()
    returns_df.columns = ['date', 'return_pct']
    returns_df['Year'] = returns_df['date'].dt.year
    returns_df['Month'] = returns_df['date'].dt.strftime('%b')

    # Define month order for the columns
    month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    # 6. Pivot the data: Rows=Year, Columns=Month
    performance_grid = returns_df.pivot(index='Year', columns='Month', values='return_pct')
    performance_grid = performance_grid.reindex(columns=month_order)

    # 7. Calculate Total Yearly Return %
    yearly_equity = lv_eqity_cash_history['equity_adj'].resample('YE').last()
    yearly_returns = yearly_equity.pct_change().infer_objects(copy=False) 
    yearly_returns.iloc[0] = (yearly_equity.iloc[0] / lv_initial_equity_target - 1) 

    performance_grid['Total Year %'] = yearly_returns.values

    # 8. Output the result
    return performance_grid


def run_monte_carlo(lv_equity_curve, lv_initial_capital=100000, lv_num_simulations=1000)-> tuple:
    
    global logger
    lv_equity_curve = lv_equity_curve.copy().reset_index()
    lv_equity_curve.columns = ['date', 'equity']
    lv_equity_curve = lv_equity_curve.set_index('date')                               
    lv_equity_curve = lv_equity_curve.sort_index()

    # Calculate Daily Returns
    # We use daily returns instead of trade list shuffling to preserve
    # the correlation between overlapping positions.
    # print(lv_equity_curve.columns)
    lv_equity_curve['daily_return'] = lv_equity_curve['equity'].pct_change().fillna(0)
    daily_returns = lv_equity_curve['daily_return'].values
    
    n_days = len(daily_returns)
    duration_years = n_days / 252  # Approximate trading days per year
    
    logger.info(f"Loaded {n_days} days of data. Starting {lv_num_simulations} simulations...")

    # 2. Monte Carlo Simulation (Bootstrapping with Replacement)
    results_cagr = []
    results_max_dd = []
    results_curves = []

    np.random.seed(42)  # For reproducible results

    for i in range(lv_num_simulations):
        # Resample daily returns with replacement (Bootstrapping)
        # This simulates "what if" market days happened in a different frequency/order
        shuffled_returns = np.random.choice(daily_returns, size=n_days, replace=True)
        
        # Reconstruct Equity Curve
        equity_curve = lv_initial_capital * np.cumprod(1 + shuffled_returns)
        
        # Calculate Metrics
        final_equity = equity_curve[-1]
        cagr = (final_equity / lv_initial_capital) ** (1 / duration_years) - 1
        
        # Max Drawdown
        running_max = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - running_max) / running_max
        max_dd = drawdown.min()
        
        results_cagr.append(cagr)
        results_max_dd.append(max_dd)
        
        # Store first 100 curves for plotting to save memory
        if i < 100:
            results_curves.append(equity_curve)

    # # 3. Plotting Results
    # plt.figure(figsize=(14, 10))

    # # Plot A: Equity Curves
    # plt.subplot(2, 2, 1)
    # for curve in results_curves:
    #     plt.plot(curve, color='gray', alpha=0.1)
    # plt.plot(np.mean(results_curves, axis=0), color='blue', linewidth=2, label='Average')
    # plt.title(f'Simulated Equity Curves ({lv_num_simulations} Runs)')
    # plt.ylabel('Equity')
    # plt.grid(True, alpha=0.3)

    # # Plot B: Max Drawdown Distribution
    # plt.subplot(2, 2, 2)
    # plt.hist(results_max_dd, bins=50, color='firebrick', alpha=0.7, edgecolor='black')
    worst_5_pct_dd = np.percentile(results_max_dd, 5)
    # plt.axvline(worst_5_pct_dd, color='black', linestyle='--', label=f'Worst 5%: {worst_5_pct_dd:.2%}')
    # plt.title('Distribution of Max Drawdown')
    # plt.legend()
    # plt.grid(True, alpha=0.3)

    # # Plot C: CAGR Distribution
    # plt.subplot(2, 2, 3)
    # plt.hist(results_cagr, bins=50, color='seagreen', alpha=0.7, edgecolor='black')
    worst_5_pct_cagr = np.percentile(results_cagr, 5)
    # plt.axvline(worst_5_pct_cagr, color='black', linestyle='--', label=f'Worst 5%: {worst_5_pct_cagr:.2%}')
    # plt.title('Distribution of CAGR')
    # plt.legend()
    # plt.grid(True, alpha=0.3)

    # # Plot D: Risk vs Reward Scatter
    # plt.subplot(2, 2, 4)
    # plt.scatter(results_max_dd, results_cagr, alpha=0.5, s=15, color='purple')
    # plt.title('CAGR vs Max Drawdown')
    # plt.xlabel('Max Drawdown')
    # plt.ylabel('CAGR')
    # plt.grid(True, alpha=0.3)

    # plt.tight_layout()
    # plt.show()

    # 4. Print Statistics
    print("-" * 30)
    print("MONTE CARLO STATISTICS")
    print("-" * 30)
    print(f"Mean CAGR:          {np.mean(results_cagr):.2%}")
    print(f"Median CAGR:        {np.median(results_cagr):.2%}")
    print(f"Worst 5% CAGR:      {worst_5_pct_cagr:.2%}")
    print("-" * 30)
    print(f"Mean Max DD:        {np.mean(results_max_dd):.2%}")
    print(f"Median Max DD:      {np.median(results_max_dd):.2%}")
    print(f"Worst 5% Max DD:    {worst_5_pct_dd:.2%}")
    print("-" * 30)

    return results_curves,results_max_dd,results_cagr

def prepare_monte_carlo_data(results_curves, results_max_dd, results_cagr):
    """
    Generate Monte Carlo simulation data from backtest results
    Call this after run_monte_carlo() in your backtest
    
    Usage in myBackTesterVer2.0.py (after run_monte_carlo):
    -------
    results_curves, results_max_dd, results_cagr = run_monte_carlo(...)
    from backtest_dashboard import prepare_monte_carlo_data
    prepare_monte_carlo_data(results_curves, results_max_dd, results_cagr, TEMP)
    """

    # Sample curve to reduce file size // not sampling now. 
    sampled_curves = results_curves

    mc_data = {
        'cagr_distribution': np.array(results_cagr).tolist(),
        'max_dd_distribution': np.array(results_max_dd).tolist(),
        'sample_curves': [[float(v) for v in curve] for curve in sampled_curves],
        'stats': {
            'mean_cagr': float(np.mean(results_cagr)),
            'median_cagr': float(np.median(results_cagr)),
            'std_cagr': float(np.std(results_cagr)),
            'worst_5pct_cagr': float(np.percentile(results_cagr, 5)),
            'best_95pct_cagr': float(np.percentile(results_cagr, 95)),
            'mean_max_dd': float(np.mean(results_max_dd)),
            'median_max_dd': float(np.median(results_max_dd)),
            'std_max_dd': float(np.std(results_max_dd)),
            'worst_5pct_dd': float(np.percentile(results_max_dd, 5)),
            'best_95pct_dd': float(np.percentile(results_max_dd, 95)),
        }
    }

    with open(TEMP / 'monte_carlo_results.pkl', 'wb') as f:
        pickle.dump(mc_data, f)
    
    print("Monte Carlo data saved to monte_carlo_results.pkl")


positions = {}      # symbol -> position dict
trades = {}  
pending_buy = {}
pending_sell = {} 
all_signals = {}
all_positions = {}
data = {}
mothnly_performance = {}
annual_performance = {}



symbols = ['RELIANCE', 'TCS', 'OIL', 'HDFCBANK', 'TITAN']
start_back_test = datetime(2007, 1, 1)
end_back_test = datetime(2017, 12, 31)
max_positions = 8
capital_fraction = 0.10
commission = 0.005
annual_interest = 0.06
trading_days = 365.25
stop_loss_pct = 0.10 # stop loss percentage
which_days_to_trade = [0] # 0: Monday 6 : Sunday. list all days you would like to trade. 
position_szie_type = 4   # 1 - fixed based on initial cash and capital fraction, 2 - variable based on current equity and capital fraction, 3 - based on existing position size, 4 - volatility based position sizing using ATR.
allow_multiple_position_for_same_stock = False # mark true if multiple entries allowed else false.

initial_cash = 100000
equity_value = initial_cash

cash = initial_cash
data_start_date = start_back_test - timedelta(365*2)

mothnly_performance['Start Back Test'] = {'equity_value': initial_cash,
                                          'change_in_equity_pct': 0.0}
annual_performance['Start Back Test'] = {'equity_value': initial_cash,
                                          'change_in_equity_pct': 0.0}

## Generate all helping data required for generating signals. 
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
        dates = dates.union(set(df.index.date))


dates =  pd.Series(list(dates)).sort_values().to_list()


# note - signal generation happens everyday but trade execution happens only on specified days of week. 
# also the signals are generted frist and then exectuted on next trading day. signals are never executed on the same day.
# on the day of execution of trades signals strength is checked. 

cash_qeuality_history = {}
current_day = start_back_test
day_activity = ''


with tqdm(total=(end_back_test - start_back_test).days) as pbar:
    while current_day <= end_back_test:
        if current_day.date() not in dates: # market not open no action. 
            # print(f'not a trading day {current_day}')        
            daily_interest = cash * annual_interest/trading_days
            cash = cash + daily_interest
            equity_value = equity_value +  daily_interest
            day_activity = 'Market closed - no action'
            cash_qeuality_history[current_day.date()] = {'cash': cash, 'equity': equity_value, 'activity': day_activity}
            current_day += timedelta(days=1)
            pbar.update(1) 
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
                        
                        logging.info(f"On {current_day.date()} for symbol {value['symbol']} determined position size to buy is {num_shares_to_buy} shares. Reason: {reason_for_position_size}")

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
        # for new strategy change this block only
        # STRATEGY SECTION START
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
            symbol = value['symbol']
            selected_low = data[symbol].loc[data[symbol].index.date <= current_day.date(), 'low'].iloc[-1]    
            buy_price = value['buy_price']
            
            current_price = selected_low
            pct_change = (current_price - buy_price) / buy_price 
            if pct_change <= -stop_loss_pct:
                pending_sell[key] = {'symbol': value['symbol'],
                                    'signal_date': current_day.date(),
                                    'signal_type': 'STOP_LOSS_SELL',
                                    'signal_processed': False,
                                    'signal_processed_message': 'Generated due to stop loss'
                                    }
                day_activity = day_activity + f'; Date {current_day.date()} Stop Loss Sell signal generated for symbol {symbol} as price dropped to {current_price:.2f} which is {pct_change*100:.2f}% below buy price {buy_price:.2f}'
        # STRATEGY SECTION END. 

        # Update cash and equity value at end of day
        cash = cash + cash * annual_interest/trading_days
        equity_value = cash + sum([ pos['num_shares'] * data[pos['symbol']].loc[data[pos['symbol']].index.date <= current_day.date(), ['close','open','high','low']].iloc[-1].mean() for pos in positions.values() ])
        cash_qeuality_history[current_day.date()] = {'cash': cash, 'equity': equity_value, 'activity': day_activity.strip('; ')}
        current_day += timedelta(days=1)

        pbar.update(1) 
# exit all remaining positions at the end of backtest period
last_back_test_date = datetime(dates[-1].year,dates[-1].month,dates[-1].day)

day_activity,cash = close_all_positions_at_end_of_backtest(lv_back_test_end_date=last_back_test_date,
                                                    lv_positions=positions,
                                                    lv_data=data,
                                                    lv_all_signals=all_signals,
                                                    lv_trades=trades,
                                                    lv_cash=cash)
equity_value = cash
cash_qeuality_history[last_back_test_date.date()] = {'cash': cash, 'equity': cash, 'activity': day_activity.strip('; ')}
# the equity value at the end of backtest is equal to cash as all positions are closed. then only call the performance report generation function.

equity_curve,drawdown = generate_back_test_performance_report(trades, cash_qeuality_history,initial_cash, start_back_test, end_back_test)

results_curves,results_max_dd,results_cagr = run_monte_carlo(lv_equity_curve=equity_curve,lv_initial_capital=initial_cash,lv_num_simulations=1000)

prepare_monte_carlo_data(results_curves,results_max_dd,results_cagr)

# write all performance data to csv files for further analysis if needed.
all_signals = pd.DataFrame(all_signals).T
all_signals = all_signals.reset_index()
all_signals = all_signals.rename(columns={'index':'key'})
all_signals.to_csv(TEMP/'all_signals.csv')
all_positions = pd.DataFrame(all_positions).T
all_positions = all_positions.reset_index()
all_positions = all_positions.rename(columns={'index':'key'})
all_positions.to_csv(TEMP/'all_positions.csv')
trades = pd.DataFrame(trades).T
trades = trades.reset_index()
trades = trades.rename(columns={'index':'key'})
trades.to_csv(TEMP/'trades.csv')
cash_qeuality_history = pd.DataFrame(cash_qeuality_history).T

cash_qeuality_history = cash_qeuality_history.reset_index()
cash_qeuality_history = cash_qeuality_history.rename(columns={'index':'date'})
cash_qeuality_history['date'] = pd.to_datetime(cash_qeuality_history['date'])
cash_qeuality_history.to_csv(TEMP/'cash_quality_history.csv')

performance_grid = generate_monthly_and_yearly_performance(lv_eqity_cash_history=cash_qeuality_history,
                                                           lv_initial_equity_target=initial_cash)
performance_grid.to_csv(TEMP/'performance_grid.csv')

equity_curve = equity_curve.reset_index()
equity_curve = equity_curve.rename(columns = {'index':'date'})
equity_curve.date = pd.to_datetime(equity_curve.date)

drawdown = drawdown.reset_index()
drawdown = drawdown.rename(columns = {'index':'date'})
drawdown.date = pd.to_datetime(drawdown.date)

equity_curve.to_csv(TEMP/'equity_curve.csv')
drawdown.to_csv(TEMP/'drawdown.csv')

logging.info(f'Final equity value: {equity_value}, Final cash value: {cash}')


backtest_dashboard.display_results()


