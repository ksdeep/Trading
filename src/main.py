import FetchAndCleanData
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

logging.basicConfig(format='Date-Time : %(asctime)s : Line No. : %(lineno)d - %(message)s', level = logging.INFO)

DATA_FLRD = r'C:\Users\ksdee\Documents\PersonalFinance\Trading\Trading_Data'
if __name__ == '__main__':
    print('Run Code')
    FetchAndCleanData.NSEStockList_All_Download(DATA_FLRD)
    