import io
import os
import glob
import json
import zipfile
import datetime
import requests
import pandas as pd
from pynse import *

from shutil import copyfile
from distutils.dir_util import copy_tree
import logging

logging.basicConfig(format='Date-Time : %(asctime)s : Line No. : %(lineno)d - %(message)s', level=logging.INFO)

Nifty_index_list = ['nifty50', 'niftynext50', 'nifty100', 'nifty200', 'nifty500', 'niftymidcap50', 'niftymidcap100',
                    'niftymidcap150', 'niftysmallcap50', 'niftysmallcap100', 'niftysmallcap250',
                    'niftylargemidcap250', 'niftymidsmallcap400', ]

DATA_FLRD = r'C:\Users\ksdee\Documents\PersonalFinance\Trading\Trading_Data'


def BhavCopyDownload(mypath, fetchday):
    Year = fetchday.strftime("%Y")
    Month = fetchday.strftime("%b").upper()
    day = fetchday.strftime("%d").upper()
    formatDate = day + Month + Year

    # url = 'https://www1.nseindia.com/content/historical/EQUITIES/' + Year + '/' + Month + '/cm' + formatDate + 'bhav.csv.zip'
    url = 'https://archives.nseindia.com/content/historical/EQUITIES/' + Year + '/' + Month + '/cm' + formatDate + 'bhav.csv.zip'
    # logging(url)
    successFlag = False
    try:
        response = requests.get(url, timeout=10)
        if response.ok:
            z = zipfile.ZipFile(io.BytesIO(response.content))
            z.extractall(mypath)
            successFlag = True
            logging.info(f'Bhavcopy Downloaded for Date %s' % (formatDate))
        else:
            logging.info('NSE Bhavcopy not available on %s' % (formatDate))
    except:
        logging.info('NSE Bhavcopy not available on %s' % (formatDate))

    return successFlag


def downloadBhwaCopy(dates_list):
    savepath = os.path.join(
        "C:",
        os.sep,
        r"C:\Users\ksdee\Documents\PersonalFinance\Trading\Trading_Data\NSEData\Bhavcopy",
    )
    logging.info(
        "Downloding files from NSE Bhawcopy %d"
        % (len(dates_list))
    )
    # return
    sucs_cnt = 0
    un_sucs_cnt = 0
    for x in dates_list:
        if BhavCopyDownload(savepath, x):
            sucs_cnt = sucs_cnt + 1
        else:
            un_sucs_cnt = un_sucs_cnt + 1
    logging.info(
        "Dates to download = %d Sucessfully downloaded = %d Could not download = %d"
        % (len(dates_list), sucs_cnt, un_sucs_cnt)
    )


def downloadAllBhavCopyLight():
    global DATA_FLRD
    start_date = datetime.date(2012, 1, 1)
    end_date = datetime.date.today()

    dateRange = pd.Series(pd.date_range(start_date, end_date, freq='B').values)
    holiday_list = pd.read_excel(DATA_FLRD + os.path.sep + 'NSEData' + os.path.sep + 'nse_holidays.xlsx')
    dateRange = dateRange.loc[~dateRange.isin(holiday_list.holidays)]

    flLst = pd.Series(os.listdir(r'C:\Users\ksdee\Documents\PersonalFinance\Trading\Trading_Data\NSEData\Bhavcopy'))
    flLst = flLst.apply(lambda x: datetime.datetime.strptime(x[2:11], '%d%b%Y').date())

    dateRange = dateRange.loc[~dateRange.isin(flLst)]

    downloadBhwaCopy(dateRange.apply(lambda x: x.date()))


def readLighBhavCopy():
    global DATA_FLRD
    fldr = r'C:\Users\ksdee\Documents\PersonalFinance\Trading\Trading_Data\NSEData\Bhavcopy'
    flLst = pd.Series(os.listdir(fldr))

    df = pd.concat([pd.read_csv(fldr + os.path.sep + fl) for fl in flLst])
    df = df.loc[df.SERIES == 'EQ', ['SYMBOL', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'TOTTRDQTY', 'TOTTRDVAL', 'TIMESTAMP',
                                    'TOTALTRADES']]
    df.rename(columns={'TOTTRDQTY': 'VOLUME', 'TIMESTAMP': 'DATE', 'TOTALTRADES': 'TRADES'}, inplace=True)
    df = df[['DATE', 'SYMBOL', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME', 'TRADES']]
    df.to_csv(r'C:\Users\ksdee\Documents\PersonalFinance\Trading\Trading_Data\NSEData\lightBhavCopy.csv')
    return df

def downlaodBhavCopy():
    global DATA_FLRD
    start_date = datetime.date(2020, 1, 1)
    end_date = datetime.date.today()
    bhavCopyPath = DATA_FLRD + os.path.sep + 'NSEData' + os.path.sep + 'bhavCopyPath.csv'

    if os.path.exists(bhavCopyPath):
        bhavDf = pd.read_csv(bhavCopyPath)
    else:
        bhavDf = pd.DataFrame()

    dateRange = pd.Series(pd.date_range(start_date, end_date, freq='B').values)
    holiday_list = pd.read_excel(DATA_FLRD + os.path.sep + 'NSEData' + os.path.sep + 'nse_holidays.xlsx')
    dateRange = dateRange.loc[~dateRange.isin(holiday_list.holidays)]
    dateRange = dateRange.loc[~dateRange.isin(np.unique(bhavDf.DATE1))]

    nse = Nse()
    for dd in dateRange:
        logging.info(
            "Dates to download = %s"
            % (dd.date())
        )
        try:
            df = nse.bhavcopy(datetime.date(dd.year, dd.month, dd.day))
        except:
            df = pd.DataFrame()
        if len(df)>0:
            logging.info(
                "Dates to download = %s size %d"
                % (dd.date(),df.shape[0])
            )
            bhavDf = bhavDf.append(df)

    bhavDf.to_csv(bhavCopyPath, index=False)
    return bhavDf.rename(columns={'DATE1': 'DATE'})
def getLast10YrsAdjustedEODData():
    global DATA_FLRD
    mypath = DATA_FLRD
    import datetime as dt
    import yfinance as yf
    import nsepy as npy
    import re
    from dateutil.relativedelta import relativedelta
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
                    if len(df)>0:
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


if __name__ == '__main__':
    downloadAllBhavCopyLight()
    lightBhavCopy = readLighBhavCopy()
    detlBhavCopy = downlaodBhavCopy()
    vvapBhavCopy = getLast10YrsAdjustedEODData()


