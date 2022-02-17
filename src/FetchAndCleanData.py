
import logging
import pandas as pd
import yfinance   as yf
import numpy as np
from pynse import *
from tvDatafeed import TvDatafeed, Interval
import datetime as dt
import matplotlib.pyplot as plt
import mplfinance as mpf
import talib as ta
import sys
import os
Nifty_index_list = ['nifty50', 'niftynext50', 'nifty100', 'nifty200', 'nifty500', 'niftymidcap50', 'niftymidcap100',
                  'niftymidcap150', 'niftysmallcap50', 'niftysmallcap100', 'niftysmallcap250',
                  'niftylargemidcap250', 'niftymidsmallcap400', ]

def NSEStockList_All_Download(mypath):
    global  Nifty_index_list 
    index_path = mypath + os.path.sep + 'NSEData' + os.path.sep + 'index.csv'
    try:
        index = pd.concat([pd.read_csv('https://archives.nseindia.com/content/indices/ind_' + nList + 'list.csv').assign(Index = nList.upper()) for nList in Nifty_index_list])
        index.to_csv(index_path, index=False)
    except:
        index = pd.read_csv(index_path)
    logging.info('NSE All Stock List Downloaded '+str(index.shape))
    return index
def downlaodBhavCopy(mypath):
    from_date = dt.date(2020,1,1)
    to_date = dt.date.today()
    bhavCopyPath = mypath + os.path.sep + 'NSEData' + os.path.sep + 'bhavCopyPath.csv'


    if os.path.exists(bhavCopyPath):
        bhavDf = pd.read_csv(bhavCopyPath)
    else :
        bhavDf = pd.DataFrame()

    nse = Nse()
    for dd in pd.date_range(from_date, to_date, freq='B').to_pydatetime():
        if (bhavDf.DATE1==dt.date(2020,1,1)).sum()==0:
            df = nse.bhavcopy(dt.date(dd.year,dd.month, dd.day))
            bhavDf = bhavDf.append( df)
    bhavDf.to_csv(bhavCopyPath,index=False)

    return bhavDf.rename(columns = {'DATE1':'DATE'})

