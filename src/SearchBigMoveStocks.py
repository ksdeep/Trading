import pandas as pd
import requests
import os
import sys
import datetime
import numpy as np
import re
import logging
from dateutil.relativedelta import relativedelta, TH
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import talib as ta

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
import nsepy as npy
logging.basicConfig(format='Date-Time : %(asctime)s : Line No. : %(lineno)d - %(message)s', level=logging.INFO)
freshRun = True
def oiAnalysis(prcChng, oiChg):
    if prcChng > 0 and oiChg > 0:
        return "LongBuildUp"
    elif prcChng > 0 and oiChg < 0:
        return "ShortCovering"
    elif prcChng < 0 and oiChg > 0:
        return "ShortBuildUp"
    elif prcChng < 0 and oiChg < 0:
        return "LongUnWinding"

if __name__ == '__main__':
    
    analysis = pd.DataFrame(columns=['Symbol','Signal'])
    print('numpy version %s'%np.__version__)
    start_date = datetime.date.today() - relativedelta(months=3)
    end_date = datetime.date.today()

    mypath = r"C:\Users\ksdee\Documents\PersonalFinance\Trading\Trading_Data"
    
    holidays = pd.read_excel(
        mypath + os.path.sep + "NSEData" + os.path.sep + "nse_holidays.xlsx"
    )
    holidays.holidays = holidays.holidays.apply(lambda x: x.date())
    symbols = pd.read_csv(mypath + os.path.sep + "NSEData" + os.path.sep + "FNO.csv")
    today = end_date
    from_date = start_date
    dtList = list(pd.date_range(from_date, today, freq="B").to_pydatetime())
    expriyDf = pd.DataFrame()
    oiData = pd.DataFrame()
    for dd in dtList:
        if holidays.holidays.isin([datetime.date(dd.year, dd.month, dd.day)]).sum() == 0:
            end_of_month = dd + relativedelta(day=31)
            expiry_date = end_of_month + relativedelta(weekday=TH(-1))
            while True:
                if (
                    holidays.holidays.isin(
                        [
                            datetime.date(
                                expiry_date.year, expiry_date.month, expiry_date.day
                            )
                        ]
                    ).sum()
                    == 0
                ):
                    break
                else:
                    expiry_date = expiry_date - relativedelta(day=1)
            end_of_next_month = dd + relativedelta(months=1, day=31)
            expiry_date_next_month = end_of_next_month + relativedelta(weekday=TH(-1))
            while True:
                if (
                    holidays.holidays.isin(
                        [
                            datetime.date(
                                expiry_date_next_month.year,
                                expiry_date_next_month.month,
                                expiry_date_next_month.day,
                            )
                        ]
                    ).sum()
                    == 0
                ):
                    break
                else:
                    expiry_date_next_month = expiry_date_next_month - relativedelta(day=1)

            end_of_month_after_next_month = dd + relativedelta(months=2, day=31)
            expiry_date_month_after_next_month = (
                end_of_month_after_next_month + relativedelta(weekday=TH(-1))
            )
            while True:
                if (
                    holidays.holidays.isin(
                        [
                            datetime.date(
                                expiry_date_month_after_next_month.year,
                                expiry_date_month_after_next_month.month,
                                expiry_date_month_after_next_month.day,
                            )
                        ]
                    ).sum()
                    == 0
                ):
                    break
                else:
                    expiry_date_month_after_next_month = (
                        expiry_date_month_after_next_month - relativedelta(day=1)
                    )
            if dd > expiry_date:
                expiry_date = expiry_date_next_month
                expiry_date_next_month = expiry_date_month_after_next_month
                end_of_month_after_next_month = dd + relativedelta(months=3, day=31)
                expiry_date_month_after_next_month = (
                    end_of_month_after_next_month + relativedelta(weekday=TH(-1))
                )
                while True:
                    if (
                        holidays.holidays.isin(
                            [
                                datetime.date(
                                    expiry_date_month_after_next_month.year,
                                    expiry_date_month_after_next_month.month,
                                    expiry_date_month_after_next_month.day,
                                )
                            ]
                        ).sum()
                        == 0
                    ):
                        break
                    else:
                        expiry_date_month_after_next_month = (
                            expiry_date_month_after_next_month - relativedelta(day=1)
                        )

            expriyDf = expriyDf.append(
                pd.DataFrame(
                    {
                        "date": [dd],
                        "current_exp_date": [expiry_date],
                        "next_month_exp_date": [expiry_date_next_month],
                        "month_after_next_month_exp_date": [
                            expiry_date_month_after_next_month
                        ],
                    }
                )
            )

    # print(expriyDf.shape)
    existingAnal = pd.read_csv(r'C:\Users\ksdee\Documents\PersonalFinance\Trading\Trading_Data\NSEData\bigMoveAnalysis.csv')
    
    for symbol in pd.read_csv(r'C:\Users\ksdee\Documents\PersonalFinance\Trading\Trading_Data\WatchList\fno_list.tls',header=None).values:
        # if symbol[0].strip()=='TCS' :
        if  freshRun or sum(existingAnal.Symbol==symbol[0])==0:            
            try:
                logging.info(
                        "fetching stock data for %s for date %s to %s "
                        % (
                            symbol[0],
                            datetime.datetime.strftime(start_date, "%Y-%m-%d"),
                            datetime.datetime.strftime(end_date, "%Y-%m-%d"),
                        )
                    )
                ohlc_data = npy.get_history(symbol=re.sub("&", "%26", symbol[0]), start=start_date, end=end_date).reset_index()
                if ohlc_data.shape[0]>30 :
                    ohlc_data["vol_per_trade"] = ohlc_data["Volume"] / ohlc_data["Trades"]
                    for _, row in expriyDf.iterrows():
                        logging.info(
                            "fetching features data for %s for date %s exp dates %s %s %s"
                            % (
                                symbol[0],
                                datetime.datetime.strftime(row["date"].date(), "%Y-%m-%d"),
                                datetime.datetime.strftime(row["current_exp_date"].date(), "%Y-%m-%d"),
                                datetime.datetime.strftime(row["next_month_exp_date"].date(), "%Y-%m-%d"),
                                datetime.datetime.strftime(
                                    row["month_after_next_month_exp_date"].date(), "%Y-%m-%d"
                                ),
                            )
                        )
                        current_month_features = npy.get_history(
                            symbol=re.sub("&", "%26", symbol[0]),
                            start=row["date"].date(),
                            end=row["date"].date(),
                            futures=True,
                            expiry_date=row["current_exp_date"].date(),
                        )
                        next_month_features = npy.get_history(
                            symbol=re.sub("&", "%26", symbol[0]),
                            start=row["date"].date(),
                            end=row["date"].date(),
                            futures=True,
                            expiry_date=row["next_month_exp_date"].date(),
                        )
                        month_after_next_month_features = npy.get_history(
                            symbol=re.sub("&", "%26", symbol[0]),
                            start=row["date"].date(),
                            end=row["date"].date(),
                            futures=True,
                            expiry_date=row["month_after_next_month_exp_date"].date(),
                        )
                        if (
                            (len(current_month_features["Open Interest"].values) == 1)
                            and (len(next_month_features["Open Interest"].values) > 0)
                            and (len(month_after_next_month_features["Open Interest"].values) > 0)
                        ):
                            oiData = oiData.append(
                                pd.DataFrame(
                                    {
                                        "Symbol": [symbol[0]],
                                        "Date": [row["date"].date()],
                                        "cummOI": [
                                            current_month_features["Open Interest"].values[0]
                                            + next_month_features["Open Interest"].values[0]
                                            + month_after_next_month_features["Open Interest"].values[0]
                                        ],
                                    }
                                )
                            )
                    ohlc_oi_date = pd.merge(
                        left=oiData,
                        right=ohlc_data,
                        left_on=["Date", "Symbol"],
                        right_on=["Date", "Symbol"],
                        how="inner",
                    )
                    ohlc_oi_date.set_index("Date", inplace=True)
                    ohlc_oi_date.sort_index(inplace=True)
                    ohlc_oi_date["5 Day avg Del Vol"] = ta.MA(
                        ohlc_oi_date["Deliverable Volume"], 5, ta.MA_Type.SMA
                        ).shift(1)
                    avgDelPct = ohlc_oi_date["%Deliverble"].mean()
                    avgVolPerTrade = ohlc_oi_date["vol_per_trade"].mean()
                    ohlc_oi_date["OI Chng"] = ohlc_oi_date["cummOI"] - ohlc_oi_date["cummOI"].shift(1)
                    ohlc_oi_date["% OI Change"] = (
                        ohlc_oi_date["cummOI"] - ohlc_oi_date["cummOI"].shift(1)
                    ) / ohlc_oi_date["cummOI"].shift(1)
                    ohlc_oi_date["% Price Change"] = (
                        ohlc_oi_date["Close"] - ohlc_oi_date["Close"].shift(1)
                    ) / ohlc_oi_date["Close"].shift(1)
                    ohlc_oi_date["oiAnalysis"] = ohlc_oi_date.apply(
                        lambda x: oiAnalysis(x["% Price Change"], x["% OI Change"]), axis=1
                        )
                    ohlc_oi_date.sort_index(ascending=False, inplace=True)
                    last5Days = 5
                    LBD_Days = sum(ohlc_oi_date.iloc[0:last5Days, :]["oiAnalysis"] == "LongBuildUp")
                    cls_higher_vwap = ohlc_oi_date.iloc[0, :]["Close"] > ohlc_oi_date.iloc[0, :]["VWAP"]
                    del_pct_higher_than_avg = ohlc_oi_date.iloc[0, :]["%Deliverble"] > avgDelPct
                    vol_per_trade_than_avg = ohlc_oi_date.iloc[0, :]["vol_per_trade"] > avgVolPerTrade
                    signal = "None"
                    if (
                        del_pct_higher_than_avg
                        and vol_per_trade_than_avg
                        and cls_higher_vwap
                        and LBD_Days > 1
                        and ohlc_oi_date.iloc[0, :]["oiAnalysis"] == "LongBuildUp"
                    ):
                        signal = "VERY_STRONG"
                    elif (
                        del_pct_higher_than_avg
                        and vol_per_trade_than_avg
                        and cls_higher_vwap 
                        and ohlc_oi_date.iloc[0, :]["oiAnalysis"] == "LongBuildUp"
                    ):
                        signal = "STRONG"
                    else:
                        signal = "None"
                    analysis = analysis.append(pd.DataFrame({'Symbol':[symbol[0]],
                                                'Signal':[signal]}))
                    ohlc_oi_date.sort_values(by="Date", ascending=False).to_csv(
                        r"C:\Users\ksdee\Documents\PersonalFinance\Trading\Trading_Data\NSEData\ohlc_oi_date_"+symbol[0]+'.csv')
                    analysis.to_csv(r"C:\Users\ksdee\Documents\PersonalFinance\Trading\Trading_Data\NSEData\bigMoveAnalysis.csv",   index=False)
                    logging.info('Symbol %s analysis %s'%(symbol[0],signal))
            except:
                logging.error('ERROR IN SYMBOL %S'%(symbol[0]))
                continue
