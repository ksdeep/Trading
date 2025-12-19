import pandas as pd
from datetime import datetime,timedelta
from kiteconnect import KiteConnect
import pandas as pd
import nsepython as nse
import yfinance as yf   
import os 
from pathlib import Path
import requests
from io import StringIO
from datetime import datetime, timedelta
import re
import numpy as np
import logging
import math 
import pickle as pkl
import time
from tqdm import tqdm 
import configparser
import zipfile
import io

logger = logging.getLogger()
logger.setLevel(logging.CRITICAL) 

HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) "
                   "Chrome/129.0.0.0 Safari/537.36"),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
}

SERIES_TO_CHOOSE = ['EQ','BZ','BE']

CONFIG_PATH = Path(__file__).parent.parent / "config.ini"
DATA_DIR = Path(__file__).parent.parent  / "Data"
STOCK_DIR = DATA_DIR / "stocks"
INDEX_DIR = DATA_DIR / "index"
INDEX_DIR_CONST = INDEX_DIR / "constituents"
INDEX_DIR_VAL = INDEX_DIR / "value"
TEMP = DATA_DIR / "temp"


for d in (DATA_DIR, STOCK_DIR):
        d.mkdir(parents=True, exist_ok=True)
for d in (DATA_DIR, INDEX_DIR):
    d.mkdir(parents=True, exist_ok=True)

for d in (INDEX_DIR,INDEX_DIR_CONST):
    d.mkdir(parents=True, exist_ok=True)

for d in (INDEX_DIR,INDEX_DIR_VAL):
    d.mkdir(parents=True, exist_ok=True)

for d in (DATA_DIR,TEMP):
    d.mkdir(parents=True, exist_ok=True)



index_data = {
    "NIFTY_500": ["https://www.niftyindices.com/IndexConstituent/ind_nifty500list.csv", "NIFTY 500","^CRSLDX"],
    "NIFTY_100": ["https://www.niftyindices.com/IndexConstituent/ind_nifty100list.csv", "NIFTY 100","^CNX100"],
    "NIFTY_MIDCAP": ["https://www.niftyindices.com/IndexConstituent/ind_niftyMidcap100list.csv", "NIFTY MIDCAP 100","NIFTY_MIDCAP_100.NS"],
    "NIFTY_SMALLCAP": ["https://www.niftyindices.com/IndexConstituent/ind_niftySmallcap100list.csv", "NIFTY SMLCAP 100","^CNXSC"],
    "NIFTY_AUTO": ["https://www.niftyindices.com/IndexConstituent/ind_niftyautolist.csv", "NIFTY AUTO","^CNXAUTO"],
    "NIFTY_BANK": ["https://www.niftyindices.com/IndexConstituent/ind_niftybanklist.csv", "NIFTY BANK","^NSEBANK"], 
    "NIFTY_FMCG": ["https://www.niftyindices.com/IndexConstituent/ind_niftyfmcglist.csv", "NIFTY FMCG","^CNXFMCG"],
    "NIFTY_MEDIA": ["https://www.niftyindices.com/IndexConstituent/ind_niftymedialist.csv", "NIFTY MEDIA","^CNXMEDIA"],
    "NIFTY_METAL": ["https://www.niftyindices.com/IndexConstituent/ind_niftymetallist.csv", "NIFTY METAL","^CNXMETAL"],
    "NIFTY_PHARMA": ["https://www.niftyindices.com/IndexConstituent/ind_niftypharmalist.csv", "NIFTY PHARMA","^CNXPHARMA"],
    "NIFTY_REALTY": ["https://www.niftyindices.com/IndexConstituent/ind_niftyrealtylist.csv", "NIFTY REALTY","^CNXREALTY"],
    "NIFTY_IT": ["https://www.niftyindices.com/IndexConstituent/ind_niftyitlist.csv", "NIFTY IT","^CNXIT"],
    "NIFTY_FINANCE_SER": ["https://www.niftyindices.com/IndexConstituent/ind_niftyfinancelist.csv", "NIFTY FIN SERVICE","NIFTY_FIN_SERVICE.NS"],
    "NIFTY_ENERGY": ["https://www.niftyindices.com/IndexConstituent/ind_niftyenergylist.csv", "NIFTY ENERGY","^CNXENERGY"],
    "NIFTY_PSUBANK": ["https://www.niftyindices.com/IndexConstituent/ind_niftypsubanklist.csv", "NIFTY PSU BANK","^CNXPSUBANK"],
    "NIFTY_CONSUMER_DURABLES": ["https://www.niftyindices.com/IndexConstituent/ind_niftyconsumerdurableslist.csv", "NIFTY CONSR DURBL","NIFTY_CONSR_DURBL.NS"],
    "NIFTY_OIL_GAS": ["https://www.niftyindices.com/IndexConstituent/ind_niftyoilgaslist.csv", "NIFTY OIL AND GAS","NIFTY_OIL_AND_GAS.NS"],
    "NIFTY_HEALTHCARE": ["https://www.niftyindices.com/IndexConstituent/ind_niftyhealthcarelist.csv", "NIFTY HEALTHCARE","NIFTY_HEALTHCARE.NS"],
    "NIFTY_50": ["https://www.niftyindices.com/IndexConstituent/ind_nifty50list.csv", "NIFTY 50","^NSEI"],
    "NIFTY_MIDCAP_150":["https://www.niftyindices.com/IndexConstituent/ind_niftymidcap150list.csv","NIFTY MIDCAP 150","NIFTYMIDCAP150.NS"],
    "NIFTY_SMALLCAP_250":["https://www.niftyindices.com/IndexConstituent/ind_niftysmallcap250list.csv","NIFTY SMLCAP 250","NIFTYSMLCAP250.NS"],
    "NIFTY_MICROCAP_250":["https://www.niftyindices.com/IndexConstituent/ind_niftymicrocap250_list.csv","NIFTY MICROCAP250","NIFTY_MICROCAP250.NS"]    
}
INDEX_TICKERS_DF = pd.DataFrame.from_dict(index_data, orient='index', columns=['CSV_URL', 'NSE_SYMBOL','YF_SYMBOL']).reset_index()
INDEX_TICKERS_DF.rename(columns={'index': 'INDEX_NAME'}, inplace=True)


def get_index_constituents():
     for _, row in INDEX_TICKERS_DF.iterrows():
        index_name = row['INDEX_NAME']           # Name of the index (e.g., NIFTY50)
        csv_url = row['CSV_URL']                 # URL to download the index constituents CSV
        outfile = INDEX_DIR_CONST / f"{index_name}.csv"  # Output file path for storing cleaned CSV

        # Download the CSV from NSE (or other source)
        response = requests.get(csv_url, headers=HEADERS, timeout=30)
        response.raise_for_status()              # Raise error if request failed

        # Read downloaded CSV into DataFrame
        df = pd.read_csv(StringIO(response.text))

        # Normalize the company-name column (different CSVs use different names)
        if 'Company Name' in df.columns:
            df = df.rename(columns={'Company Name': 'company_name'})
        elif 'COMPANY' in df.columns:
            df = df.rename(columns={'COMPANY': 'company_name'})

        # Convert all column names to lowercase + strip spaces for uniformity
        df.columns = [str(x).lower() for x in df.columns.str.strip()]
        df.to_csv(outfile, index=False)

def load_data_from_zerodha(api_key, api_secret):
    # kite = KiteConnect(api_key=api_key)
    # login_url = kite.login_url()
    # request_token = input(f"Ge here to get req token {login_url}: and then type it here: ") 
    # data  = kite.generate_session(request_token, api_secret)
    # kite.set_access_token(data["access_token"])
    # instruments = pd.DataFrame(kite.instruments())
    # instruments.to_csv(TEMP/'instruments.csv')
    # print("Instruments data saved to instruments.csv")

     
    end_date = datetime.now() - timedelta(days=1)
    start_date = end_date - timedelta(days=365*20)  # Last 20 years 

    bhavcopy_file = url = f"https://nsearchives.nseindia.com/content/cm/BhavCopy_NSE_CM_0_0_0_{end_date.strftime('%Y%m%d')}_F_0000.csv.zip"
    print(f"Downloading bhavcopy from NSE...{bhavcopy_file}")

    print(f"Downloading: {url}")
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()  # raise error if not 200
    # Open ZIP from memory
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        # Assume only one CSV in the zip
        csv_name = [n for n in zf.namelist() if n.lower().endswith(".csv")][0]
        with zf.open(csv_name) as csv_file:
            bhavcopy_df = pd.read_csv(csv_file)

    bhavcopy_df.to_csv(TEMP/'bhavcopy.csv', index=False)
    print("Bhavcopy data saved to bhavcopy.csv")    



if __name__ == "__main__":
    
    # Load API credentials from config file
    config = configparser.ConfigParser()
    config.read(CONFIG_PATH)
    API_KEY = config['ZERODHA']['API_KEY']
    API_SECRET = config['ZERODHA']['API_SECRET']


    get_index_constituents()
    print("Index constituents updated.")

    load_data_from_zerodha(API_KEY, API_SECRET)

