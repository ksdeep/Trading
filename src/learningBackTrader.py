import backtrader as bt
import pandas as pd 



class TestStrategy(bt.Strategy):

    def log(self, txt, dt=None):
        ''' Logging function for this strategy'''
        dt = dt or self.datas[0].datetime.date(0)
        print('%s, %s' % (dt.isoformat(), txt))

    def __init__(self):
        # Keep a reference to the "close" line in the data[0] dataseries
        self.dataclose = self.datas[0].close

    def next(self):
        # Simply log the closing price of the series from the reference
        self.log('Close, %.2f' % self.dataclose[0])

        if self.dataclose[0] < self.dataclose[-1]:
            if self.dataclose[-1] < self.dataclose[-2]:
                self.log('BUY CREATE, %.2f' % self.dataclose[0])
                self.buy()


if __name__ == '__main__':
    cerebro = bt.Cerebro()

    cerebro.addstrategy(TestStrategy)

    cerebro.broker.setcash(100000.0)

    data = pd.read_csv(r"C:\Users\ksdee\Documents\Trading\Data\stocks\APOLLOHOSP.csv")
    data = data.loc[:,['date', 'open', 'high', 'low', 'close', 'volume']]
    data['OpenInterest'] = 0
    data.date = pd.to_datetime(data.date)
    
    data.columns = ['Date','Open','High','Low','Close','Volume','OpenInterest']

    data = data.set_index('Date')
    data = data.sort_index()

    data = bt.feeds.PandasData(dataname=data)
    cerebro.adddata(data)

    print("starting portfolio value: %.2f" % cerebro.broker.getvalue())
    cerebro.run()
    print("Final portfolio value: %.2f" % cerebro.broker.getvalue())

    cerebro.plot(style='bar')
