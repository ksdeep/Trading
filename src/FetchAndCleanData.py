import logging
import pandas as pd
import yfinance as yf
from pynse import *
import nsepy as npy
import os
from dateutil.relativedelta import relativedelta, TH
import datetime as dt
import os

logging.basicConfig(format='Date-Time : %(asctime)s : Line No. : %(lineno)d - %(message)s', level=logging.INFO)

Nifty_index_list = ['nifty50', 'niftynext50', 'nifty100', 'nifty200', 'nifty500', 'niftymidcap50', 'niftymidcap100',
                    'niftymidcap150', 'niftysmallcap50', 'niftysmallcap100', 'niftysmallcap250',
                    'niftylargemidcap250', 'niftymidsmallcap400', ]


def NSEStockList_All_Download(mypath):
    global Nifty_index_list
    index_path = mypath + os.path.sep + 'NSEData' + os.path.sep + 'index.csv'
    try:
        index = pd.concat([pd.read_csv(
            'https://archives.nseindia.com/content/indices/ind_' + nList + 'list.csv').assign(Index=nList.upper()) for
                           nList in Nifty_index_list])
        index.to_csv(index_path, index=False)
    except:
        index = pd.read_csv(index_path)
    logging.info('NSE All Stock List Downloaded ' + str(index.shape))
    return index


def downlaodBhavCopy(mypath):
    from_date = dt.date(2020, 1, 1)
    to_date = dt.date.today()
    bhavCopyPath = mypath + os.path.sep + 'NSEData' + os.path.sep + 'bhavCopyPath.csv'

    if os.path.exists(bhavCopyPath):
        bhavDf = pd.read_csv(bhavCopyPath)
    else:
        bhavDf = pd.DataFrame()

    nse = Nse()
    for dd in pd.date_range(from_date, to_date, freq='B').to_pydatetime():
        if (bhavDf.DATE1 == dt.date(2020, 1, 1)).sum() == 0:
            df = nse.bhavcopy(dt.date(dd.year, dd.month, dd.day))
            bhavDf = bhavDf.append(df)
    bhavDf.to_csv(bhavCopyPath, index=False)

    return bhavDf.rename(columns={'DATE1': 'DATE'})


def getLast10YrsAdjustedEODData(mypath):
    symbols = pd.read_csv(mypath + os.path.sep + 'NSEData' + os.path.sep + 'FNO.csv')
    correctedEODData = pd.concat([yf.download(sym + '.NS',
                                              start=dt.date.today().replace(year=dt.date.today().year - 10),
                                              end=dt.date.today(),
                                              interval='1d',
                                              progress=False,
                                              auto_adjust=True).assign(Symbol=sym) for sym in symbols.SYMBOL])
    correctedEODData = correctedEODData.loc[correctedEODData.Open > 0, :]
    correctedEODData = correctedEODData.loc[correctedEODData.High > 0, :]
    correctedEODData = correctedEODData.loc[correctedEODData.Low > 0, :]
    correctedEODData = correctedEODData.loc[correctedEODData.Close > 0, :]
    correctedEODData = correctedEODData.loc[correctedEODData.Volume > 0, :]
    correctedEODData.rename(columns={'Open': 'correctedOpen',
                                     'Close': 'correctedClose',
                                     'High': 'correctedHigh',
                                     'Low': 'correctedLow'}, inplace=True)
    correctedEODData = correctedEODData.drop(columns=['Volume'])
    correctedEODData = correctedEODData.reset_index()

    correctedEODData['Symbol_Date'] = correctedEODData.apply(
        lambda x: x['Symbol'] + '_' + dt.datetime.strftime(x['Date'], '%Y-%m-%d'), axis=1)

    logging.info('corrected data downloaded from Yahaoo ' + str(correctedEODData.shape))

    rawData = pd.concat([npy.get_history(symbol=sym,
                                         start=dt.date.today().replace(year=dt.date.today().year - 10),
                                         end=dt.date.today()) for sym in symbols.SYMBOL])
    rawData = rawData.reset_index()

    rawData['Symbol_Date'] = rawData.apply(
        lambda x: x['Symbol'] + '_' + dt.datetime.strftime(x['Date'], '%Y-%m-%d'), axis=1)

    data = pd.merge(left=rawData,
                    right=correctedEODData,
                    left_on=['Symbol_Date'],
                    right_on=['Symbol_Date'],
                    how='left')
    data = data.drop(columns=['Date_y', 'Symbol_y'], inplace=True)
    data.rename(columns={'Date_x': 'Date',
                         'Symbol_x': 'Symbol'}, inplace=True)
    data.set_index('Date', inplace=True)
    data['VolumePerTrade'] = data['Volume'] / data['Trades']
    logging.info('NSE data and Yahoo corrected data ready for use ' + str(data.shape))
    return data


def getFeaturesOIDataForLast6Months(mypath):
    holidays = pd.read_excel(mypath + os.path.sep + 'NSEData' + os.path.sep + 'nse_holidays.xlsx')
    holidays.holidays = holidays.holidays.apply(lambda x: x.date())
    symbols = pd.read_csv(mypath + os.path.sep + 'NSEData' + os.path.sep + 'FNO.csv')
    today = dt.datetime.today()
    from_date = today - relativedelta(months=6)
    dtList = list(pd.date_range(from_date, today, freq='B').to_pydatetime())
    expriyDf = pd.DataFrame()
    for dd in dtList:
        if holidays.holidays.isin([dt.date(dd.year, dd.month, dd.day)]).sum() == 0:
            end_of_month = dd + relativedelta(day=31)
            expiry_date = end_of_month + relativedelta(weekday=TH(-1))
            while True:
                if holidays.holidays.isin([dt.date(expiry_date.year, expiry_date.month, expiry_date.day)]).sum() == 0:
                    break;
                else:
                    expiry_date = expiry_date - relativedelta(day=1)
            end_of_next_month = dd + relativedelta(months=1, day=31)
            expiry_date_next_month = end_of_next_month + relativedelta(weekday=TH(-1))
            while True:
                if holidays.holidays.isin([dt.date(expiry_date_next_month.year, expiry_date_next_month.month,
                                                   expiry_date_next_month.day)]).sum() == 0:
                    break;
                else:
                    expiry_date_next_month = expiry_date_next_month - relativedelta(day=1)

            end_of_month_after_next_month = dd + relativedelta(months=2, day=31)
            expiry_date_month_after_next_month = end_of_month_after_next_month + relativedelta(weekday=TH(-1))
            while True:
                if holidays.holidays.isin([dt.date(expiry_date_month_after_next_month.year,
                                                   expiry_date_month_after_next_month.month,
                                                   expiry_date_month_after_next_month.day)]).sum() == 0:
                    break;
                else:
                    expiry_date_month_after_next_month = expiry_date_month_after_next_month - relativedelta(day=1)
            if dd > expiry_date:
                expiry_date = expiry_date_next_month
                expiry_date_next_month = expiry_date_month_after_next_month
                end_of_month_after_next_month = dd + relativedelta(months=3, day=31)
                expiry_date_month_after_next_month = end_of_month_after_next_month + relativedelta(weekday=TH(-1))
                while True:
                    if holidays.holidays.isin([dt.date(expiry_date_month_after_next_month.year,
                                                       expiry_date_month_after_next_month.month,
                                                       expiry_date_month_after_next_month.day)]).sum() == 0:
                        break;
                    else:
                        expiry_date_month_after_next_month = expiry_date_month_after_next_month - relativedelta(day=1)

            expriyDf = expriyDf.append(pd.DataFrame({'date': [dd],
                                                     'current_exp_date': [expiry_date],
                                                     'next_month_exp_date': [expiry_date_next_month],
                                                     'month_after_next_month_exp_date': [
                                                         expiry_date_month_after_next_month]}))
    logging.info(
        'experiy to be feteched for %d dates ' % (expriyDf.shape[0]))
    oiData = pd.DataFrame()
    for symbol in symbols.SYMBOL:
        for _, row in expriyDf.iterrows():
            logging.info('fetching features data for %s for date %s exp dates %s %s %s' % (
            symbol,
            dt.datetime.strftime(row['date'].date(), '%Y-%m-%d'),
            dt.datetime.strftime(row['current_exp_date'].date(), '%Y-%m-%d'),
            dt.datetime.strftime(row['next_month_exp_date'].date(), '%Y-%m-%d'),
            dt.datetime.strftime(row['month_after_next_month_exp_date'].date(), '%Y-%m-%d')))
            current_month_features = npy.get_history(symbol=symbol,
                                                     start=row['date'].date(),
                                                     end=row['date'].date(),
                                                     futures=True,
                                                     expiry_date=row['current_exp_date'].date())
            next_month_features = npy.get_history(symbol=symbol,
                                                  start=row['date'].date(),
                                                  end=row['date'].date(),
                                                  futures=True,
                                                  expiry_date=row['next_month_exp_date'].date())
            month_after_next_month_features = npy.get_history(symbol=symbol,
                                                              start=row['date'].date(),
                                                              end=row['date'].date(),
                                                              futures=True,
                                                              expiry_date=row['month_after_next_month_exp_date'].date())
            if len(current_month_features['Open Interest'].values) ==1:
                oiData = oiData.append(pd.DataFrame({'Symbol': [symbol],
                                                     'Date': [row['date'].date()],
                                                     'cummOI': [current_month_features['Open Interest'].values[0] +
                                                                next_month_features['Open Interest'].values[0] +
                                                                month_after_next_month_features['Open Interest'].values[0]
                                                                ]}))
    return oiData




expriyDf.to_csv(r'C:\Users\ksdee\Documents\PersonalFinance\Trading\Trading_Data\Temp\exp.csv')

npy.get_history(symbol='AARTIIND',
                                                              start=dt.date(2021,11,23),
                                                              end=dt.date(2021,11,23),
                                                              futures=True,
                                                              expiry_date=dt.date(2021,12,30))