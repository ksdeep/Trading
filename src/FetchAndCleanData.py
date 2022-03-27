import logging

import numpy as np
import pandas as pd
import yfinance as yf
from tvDatafeed import TvDatafeed, Interval
from pynse import *
import nsepy as npy
from dateutil.relativedelta import relativedelta, TH
import datetime as dt
import os
import re
import talib as ta
import matplotlib.pyplot as plot
import mplfinance as mpf

logging.basicConfig(format='Date-Time : %(asctime)s : Line No. : %(lineno)d - %(message)s', level=logging.INFO)

Nifty_index_list = ['nifty50', 'niftynext50', 'nifty100', 'nifty200', 'nifty500', 'niftymidcap50', 'niftymidcap100',
                    'niftymidcap150', 'niftysmallcap50', 'niftysmallcap100', 'niftysmallcap250',
                    'niftylargemidcap250', 'niftymidsmallcap400', ]

DATA_FLRD = r'C:\Users\ksdee\Documents\PersonalFinance\Trading\Trading_Data'


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
    try:
        existingData = pd.read_csv(mypath + os.path.sep + 'NSEData' + os.path.sep + 'NSEBhavCopy.csv')
        existingData['Date'] = pd.to_datetime(existingData['Date'])
        existingData = existingData.set_index('Date')
        existingData = existingData.drop(
            columns=['correctedOpen', 'correctedClose', 'correctedHigh', 'correctedLow', 'Symbol_Date'])
    except:
        existingData = pd.DataFrame()

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

    logging.info('corrected data downloaded from Yahaoo' + str(correctedEODData.shape))

    if existingData.shape[0] > 0:
        start_date = existingData.index.max().date() + relativedelta(days=1)
    else:
        start_date = dt.date.today().replace(year=dt.date.today().year - 10)
        end_date = dt.date.today()

    try:
        if start_date < end_date:
            dateRange = pd.Series(pd.date_range(start_date, end_date, freq='B').values)
            holiday_list = pd.read_excel(DATA_FLRD + os.path.sep + 'NSEData' + os.path.sep + 'nse_holidays.xlsx')
            dateRange = dateRange.loc[~dateRange.isin(holiday_list.holidays)]

            rawData = pd.DataFrame()
            if dateRange.shape[0] > 0:
                for sym in symbols.SYMBOL:

                    logging.info(
                        'NSE hist data for %s %s %s' % (sym, dt.datetime.strftime(dateRange.iloc[0], '%Y-%m-%d'),
                                                        dt.datetime.strftime(dateRange.iloc[-1], '%Y-%m-%d')))
                    df = npy.get_history(symbol=re.sub('&', '%26', sym), start=dateRange.iloc[0], end=dateRange.iloc[-1])
                    logging.info(
                        'NSE hist data for %s %s %s Size %s' % (
                            sym, dt.datetime.strftime(dateRange.iloc[0], '%Y-%m-%d'),
                            dt.datetime.strftime(dateRange.iloc[-1], '%Y-%m-%d'),
                            str(df.shape[0])))
                    if df.index.max() != dateRange.iloc[-1]:
                        logging.info('NSE hist data for %s missing max date %s RERUNNING' % (
                            sym, dt.datetime.strftime(dateRange.iloc[-1], '%Y-%m-%d')))
                        df1 = npy.get_history(symbol=re.sub('&', '%26', sym),
                                              start=dateRange.iloc[-1],
                                              end=dateRange.iloc[-1])
                        logging.info('NSE hist data for %s missing max date %s DATA SIZE %s' % (
                            sym, dt.datetime.strftime(dateRange.iloc[-1], '%Y-%m-%d'),
                            str(df1.shape[0])))
                        if len(df1) > 0:
                            df = df.append(df1)
                    allDates = dateRange.apply(lambda x: dt.datetime.strftime(x, '%Y%m%d'))
                    fndDates = pd.Series(df.index).apply(lambda x: dt.datetime.strftime(x, '%Y%m%d'))

                    st = pd.Series(list(set(allDates) - set(fndDates))).apply(
                        lambda x: dt.datetime.strptime(x, '%Y%m%d'))
                    missingDates = correctedEODData.loc[((correctedEODData.Symbol == sym) &
                                                         (correctedEODData.Date.isin(st))), 'Date']
                    if len(missingDates) > 0:
                        for dd in missingDates:
                            logging.info('NSE hist data for %s missing date %s so runnig again' % (
                                sym, dd))
                            df1 = npy.get_history(symbol=re.sub('&', '%26', sym),
                                                  start=dd,
                                                  end=dd)
                            logging.info('NSE hist data for %s missing date %s DATA SIZE %s' % (
                                sym, dd,str(df1.shape[0])))
                            if len(df1) > 0:
                                df = df.append(df1)

                    rawData = rawData.append(df)
    except:
        logging.error('Error to pull NSE data')
        rawData = pd.DataFrame()

    if rawData.shape[0] > 0:
        # rawData['VolumePerTrade'] = rawData['Volume'] / rawData['Trades']

        mergedData = pd.concat([existingData, rawData])
        mergedData = mergedData.reset_index()

        mergedData['Symbol_Date'] = mergedData.apply(
            lambda x: x['Symbol'] + '_' + dt.datetime.strftime(x['Date'], '%Y-%m-%d'), axis=1)

        data = pd.merge(left=mergedData,
                        right=correctedEODData,
                        left_on=['Symbol_Date'],
                        right_on=['Symbol_Date'],
                        how='left')
        data.drop(columns=['Date_y', 'Symbol_y'], inplace=True)
        data.rename(columns={'Date_x': 'Date',
                             'Symbol_x': 'Symbol'}, inplace=True)
        data.set_index('Date', inplace=True)

        newRowsAdded = data.shape[0] - existingData.shape[0]

        logging.info('new rows added %d max date %s' % (newRowsAdded, data.index.max().strftime('%Y-%m-%d')))
        data.to_csv(mypath + os.path.sep + 'NSEData' + os.path.sep + 'NSEBhavCopy.csv')
        return data
    else:
        data = pd.read_csv(mypath + os.path.sep + 'NSEData' + os.path.sep + 'NSEBhavCopy.csv')
        data['Date'] = pd.to_datetime(data['Date'])
        data = data.set_index('Date')
        logging.info('no new data found. max date %s ' % (data.index.max().strftime('%Y-%m-%d')))
        return data


def getFeaturesOIDataForLast6Months(mypath):
    existingOIData = pd.read_csv(mypath + os.path.sep + 'NSEData' + os.path.sep + 'oiData.csv')
    existingOIData['Date'] = pd.to_datetime(existingOIData['Date'])
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
    expriyDf = expriyDf.loc[~expriyDf.date.apply(lambda x: x.strftime('%Y-%m-%d')).isin(
        existingOIData.Date.apply(lambda x: x.strftime('%Y-%m-%d'))), :]

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
            if (len(current_month_features['Open Interest'].values) == 1) \
                    and (len(next_month_features['Open Interest'].values) > 0) \
                    and (len(month_after_next_month_features['Open Interest'].values) > 0):
                oiData = oiData.append(pd.DataFrame({'Symbol': [symbol],
                                                     'Date': [row['date'].date()],
                                                     'cummOI': [current_month_features['Open Interest'].values[0] +
                                                                next_month_features['Open Interest'].values[0] +
                                                                month_after_next_month_features['Open Interest'].values[
                                                                    0]
                                                                ]}))

    if oiData.shape[0] > 0:
        newOi = existingOIData.append(oiData)
        newOi.to_csv(mypath + os.path.sep + 'NSEData' + os.path.sep + 'oiData.csv', index=False)
        return newOi
    else:
        return existingOIData


def getFIIInvestmentData(mypath):
    try:
        existingfii = pd.read_csv(mypath + os.path.sep + 'NSEData' + os.path.sep + 'fii.csv')
    except:
        existingfii = pd.DataFrame()
    
    


def getForexCommodityData(mypath):
    tv = TvDatafeed()
    symbols = ['USDINR', 'USDBRO', 'GOLD1!', 'SILVER1!', 'COPPER1!', 'ALUMINIUM1!', 'IN1!']
    exch = ['IDC', 'IDC', 'MCX', 'MCX', 'MCX', 'MCX', 'SGX']
    symExch = pd.DataFrame({'symbols': symbols, 'exch': exch})
    try:
        existingfii = pd.read_csv(mypath + os.path.sep + 'NSEData' + os.path.sep + 'forexCommodity.csv')
    except:
        existingfii = pd.DataFrame()

    for _, row in symExch.iterrows():
        df = tv.get_hist(symbol=row['symbols'], exchange=row['exch'], interval=Interval.in_daily, n_bars=5000)
        existingfii = existingfii.append(df)

    return


def generateData():
    eodData = getLast10YrsAdjustedEODData(DATA_FLRD)
    oiData = getFeaturesOIDataForLast6Months(DATA_FLRD)
    fiiData = getFIIInvestmentData(DATA_FLRD)

    dataForAmiBrokder = pd.merge(left=eodData.reset_index(),
                                 right=oiData,
                                 left_on=['Symbol', 'Date'],
                                 right_on=['Symbol', 'Date'],
                                 how='left')
    dataForAmiBrokder['YMD'] = dataForAmiBrokder.Date.apply(lambda x: dt.datetime.strftime(x.date(), '%Y%m%d'))

    start = dt.date.today().replace(year=dt.date.today().year - 10)
    end = dt.date.today()
    dateRange = pd.Series(pd.date_range(start, end, freq='B').values)
    holiday_list = pd.read_excel(DATA_FLRD + os.path.sep + 'NSEData' + os.path.sep + 'nse_holidays.xlsx')
    dateRange = dateRange.loc[~dateRange.isin(holiday_list.holidays)]

    dataForAmiBrokderGrpd = dataForAmiBrokder.groupby('Symbol')

    for name, grpd in dataForAmiBrokderGrpd:
        max_date = grpd.Date.max()
        min_date = grpd.Date.min()
        if max_date != dateRange[dateRange.shape[0] - 1]:
            logging.error('Max date not equal to current date')

    dataForAmiBrokder[['Symbol', 'YMD', 'Open',
                       'High', 'Low', 'Close',
                       'Volume', 'Trades', 'Deliverable Volume',
                       'cummOI']].to_csv(DATA_FLRD + os.path.sep + 'NSEData' + os.path.sep + 'dataForAmiBrokder.csv',
                                         index=False)
    dataForAmiBrokder['Symbol_VWAP'] = dataForAmiBrokder.Symbol.apply(lambda x: x + '_VWAP')
    dataForAmiBrokder[['Symbol_VWAP', 'YMD', 'correctedOpen',
                       'correctedHigh', 'correctedLow', 'correctedClose',
                       'VWAP']].to_csv(DATA_FLRD + os.path.sep + 'NSEData' + os.path.sep + 'dataForAmiBrokderVWAP.csv',
                                       index=False)


def whereAreFiiInvesting(myPath):
    def masscenter(arr):
        return (arr[1] - arr[0]) * 100 / arr[0]

    fiiData = pd.read_csv(myPath + os.path.sep + 'NSEData' + os.path.sep + 'fii.csv')
    fiiData.set_index('Sector', inplace=True)

    changFii = fiiData.rolling(window=2, axis=1).apply(masscenter, raw=True)

    changFiiDesc = changFii.iloc[0:changFii.shape[0] - 1, :].sort_values(changFii.columns[len(changFii.columns) - 1],
                                                                         ascending=False)
    fiiData.iloc[0:changFii.shape[0] - 1, :].to_csv(myPath + os.path.sep + 'NSEData' + os.path.sep + 'fiiPlot.csv')
    changFiiDesc.to_csv(myPath + os.path.sep + 'NSEData' + os.path.sep + 'fiiChangePlot.csv')


def analzeStcocsBasedOIDelevery(myPath):
    eodData = pd.read_csv(myPath + os.path.sep + 'NSEData' + os.path.sep + 'NSEBhavCopy.csv', low_memory=False)
    oiData = pd.read_csv(myPath + os.path.sep + 'NSEData' + os.path.sep + 'oiData.csv', low_memory=False)
    stockDesc = pd.read_csv(myPath + os.path.sep + 'NSEData' + os.path.sep + 'index.csv', low_memory=False)

    for script in eodData.Symbol.unique():
        eodDScript = eodData.script



