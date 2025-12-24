from GetFreshMarketData import *

def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def roc(series, period):
    return series.pct_change(periods=period)

symbols = ['RELIANCE', 'TCS', 'WIPRO', 'HDFCBANK', 'TITAN']
data = {}



initial_cash = 100000
start_back_test = datetime(2010, 1, 1)
end_back_test = datetime(2020, 12, 31)
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

dates = sorted(set().union(*[df.index for df in data.values()]))

trading_days = 
current_day = start_back_test
while current_day <= end_back_test:
    cash = cash + cash * annual_interest/trading_days
    if current_day not in dates:
        # not a trading day
        current_day += timedelta(days=1)
        continue
    
    
    current_day += timedelta(days=1)
    