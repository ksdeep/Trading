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

logging.basicConfig(
    format="Date-Time : %(asctime)s : Line No. : %(lineno)d - %(message)s",
    level=logging.INFO,
)
monthsBack = 1


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
    analysis = pd.DataFrame(columns=["Symbol", "Signal"])
    print("numpy version %s" % np.__version__)
    start_date = datetime.date.today() - relativedelta(months=monthsBack)
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
    
