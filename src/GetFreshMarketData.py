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
from nselib import capital_market
from jugaad_data.nse import bhavcopy_save




HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) "
                   "Chrome/129.0.0.0 Safari/537.36"),
    
    "Accept": "application/json,text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Referer": "https://www.nseindia.com/",
}

SERIES_TO_CHOOSE = ['EQ','BZ','BE']

CONFIG_PATH = Path(__file__).parent.parent / "config.ini"
DATA_DIR = Path(__file__).parent.parent  / "Data"
STOCK_DIR = DATA_DIR / "stocks"
INDEX_DIR = DATA_DIR / "index"
INDEX_DIR_CONST = INDEX_DIR / "constituents"
INDEX_DIR_VAL = INDEX_DIR / "value"
BHAV_COPY_DIR = DATA_DIR / "bhavcopy"
TEMP = DATA_DIR / "temp"

for d in (DATA_DIR, BHAV_COPY_DIR):
        d.mkdir(parents=True, exist_ok=True)

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


logger = logging.getLogger("MomentumTrader")
logging.getLogger("charset_normalizer").setLevel(logging.DEBUG)
file_handler = logging.FileHandler(TEMP/"MomentumTrader.log", mode="w")     
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

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

def download_bhavcopy_get_symbol_list(date):
    end_date = date
    start_date = end_date - timedelta(days=365*20)  # Last 20 years 

    bhavcopy_file = url = f"https://nsearchives.nseindia.com/content/cm/BhavCopy_NSE_CM_0_0_0_{end_date.strftime('%Y%m%d')}_F_0000.csv.zip"
    logger.debug(f"Downloading bhavcopy from NSE...{bhavcopy_file}")

    logger.debug(f"Downloading: {url}")
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()  # raise error if not 200
    # Open ZIP from memory
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        # Assume only one CSV in the zip
        csv_name = [n for n in zf.namelist() if n.lower().endswith(".csv")][0]
        with zf.open(csv_name) as csv_file:
            bhavcopy_df = pd.read_csv(csv_file)
        bhavcopy_df.to_csv(TEMP/'bhavcopy.csv', index=False)

    bhavcopy_df = pd.read_csv(TEMP/'bhavcopy.csv')
    symbol_list = bhavcopy_df.loc[bhavcopy_df.SctySrs.isin(SERIES_TO_CHOOSE), 'TckrSymb'].apply(lambda x :str(x).strip().upper()).unique().tolist()
    index_symbol_list = INDEX_TICKERS_DF.NSE_SYMBOL.tolist()
    symbol_list = list(set(symbol_list).union(set(index_symbol_list)))
    logger.debug("Bhavcopy data saved to bhavcopy.csv")
    return symbol_list

def load_data_from_zerodha(api_key, api_secret,end_date, start_date):
    kite = KiteConnect(api_key=api_key)
    login_url = kite.login_url()
    request_token = input(f"Ge here to get req token {login_url}: and then type it here: ") 
    data  = kite.generate_session(request_token, api_secret)
    kite.set_access_token(data["access_token"])
    instruments = pd.DataFrame(kite.instruments())
    instruments.to_csv(TEMP/'instruments.csv')
    logger.debug("Instruments data saved to instruments.csv")

     
    

    symbol_list = download_bhavcopy_get_symbol_list(end_date) 
    progress_bar = tqdm(symbol_list, desc="Downloading historical data", total=len(symbol_list))

    for symbol in progress_bar:
        progress_bar.set_description(f"Downloading historical Data for {symbol}")
        try:
            start = start_date 
            instrument_row = instruments[(instruments.tradingsymbol == symbol) & (instruments.exchange == "NSE")]
            if instrument_row.empty:
                logger.debug(f"Symbol {symbol} not found in instruments list.")
                continue
            instrument_token = instrument_row.iloc[0]['instrument_token']
            instrument_name = instrument_row.iloc[0]['name']
            instrument_segment = instrument_row.iloc[0]['segment']
            if instrument_segment == 'INDICES':
                instrument_name = instrument_name + '_' + instrument_segment
            
            all_data_for_symbol = pd.DataFrame()
            while start < end_date:
                end = min(start + timedelta(days=365*5), end_date)
                if (end - start).days >= 1:                    
                    data =pd.DataFrame(kite.historical_data(instrument_token, start, end,  interval="day"))
                else :
                    data = pd.DataFrame()
                
                if not data.empty:
                    data.date = data.date.apply(lambda x : x.date())
                    data['symbol'] = symbol
                    data['name'] = instrument_name
                    all_data_for_symbol = pd.concat([all_data_for_symbol, data], ignore_index=True)
                start = end + timedelta(days=1)

            if not all_data_for_symbol.empty:
                hist_df = all_data_for_symbol[['symbol','name','date','open','high','low','close','volume']]
                hist_df.to_csv(STOCK_DIR / f"{symbol}.csv", index=False)
        except Exception as e:
            logger.error(f"Error downloading data for symbol {symbol}: {e}")            
        time.sleep(0.5)  # To avoid hitting rate limits 

def getAllTradingDates():
    url = ("https://www.nseindia.com/api/NextApi/apiClient/historicalGraph?functionName=getIndexChart&&index=NIFTY%2050&flag=30Y")

    resp = requests.get(url, headers=HEADERS, timeout=15)
    resp.raise_for_status()          # raise if HTTP error
    j = resp.json()
    # grapthData is: [[epoch_ms, close, "NM"], ...]
    graph_data = j["data"]["grapthData"]
    # Example: extract all dates
    dates = []
    for point in graph_data:
        epoch_ms = point[0]
        dt = datetime.fromtimestamp(epoch_ms / 1000.0)  # convert ms to seconds
        dates.append(dt.date())
    dates = pd.Series(dates)
    return dates

def downloadPastBhavCopies(trading_dates):
    pbar = tqdm(trading_dates,desc="Downloading Data", total=len(trading_dates))
    cuttoff_date = datetime(2024,7,5).date()
    for date in pbar:
        if date < cuttoff_date:
            try:
                bhavcopy_save(dt=date, dest=BHAV_COPY_DIR,skip_if_present=True)
            except Exception as e:
                continue
        else:
            flname= f"cm{date.strftime('%d%b%Y')}.csv"
            flname = BHAV_COPY_DIR / flname
            if not flname.exists():
                capital_market.bhav_copy_equities(trade_date=date.strftime("%d-%m-%Y")).to_csv(flname)
        pbar.set_description(f"Downloaded {date}") 
    
    list_of_files = list(BHAV_COPY_DIR.glob("*.csv"))
    pbar = tqdm(list_of_files,desc="Processing Data", total=len(list_of_files))

    all_traded_stock = pd.DataFrame()
    for flname in pbar :
        df = pd.read_csv(flname)
        df.columns = [x.lower() for x in df.columns] 
        if len(df.columns)<20:
            df = df.loc[:,['symbol','series','timestamp','tottrdqty','tottrdval']] 
            df = df.loc[df.series.isin(SERIES_TO_CHOOSE)]
            df =df.rename(columns={'timestamp':'date','tottrdqty':'total_qty','tottrdval':'total_value'})
            df['date'] = pd.to_datetime(df.date).apply(lambda x : x.date())
            df['total_trades'] = 0
        else:
            df = df.loc[:,['tckrsymb','sctysrs','traddt','ttltradgvol','ttltrfval','ttlnboftxsexctd']]
            df = df.loc[df.sctysrs.isin(SERIES_TO_CHOOSE)]        
            df =df.rename(columns={'tckrsymb':'symbol','sctysrs':'series','traddt':'date',
                                   'ttltradgvol':'total_qty','ttltrfval':'total_value',
                                   'ttlnboftxsexctd':'total_trades'})
            df['date'] = pd.to_datetime(df.date).apply(lambda x : x.date())
        all_traded_stock = pd.concat([all_traded_stock,df[['symbol','date','total_qty','total_value','total_trades']]])
    all_traded_stock.to_csv(INDEX_DIR_CONST/'all_traded_stock.csv')

if __name__ == "__main__":
    
    # Load API credentials from config file
    config = configparser.ConfigParser()
    config.read(CONFIG_PATH)
    API_KEY = config['ZERODHA']['API_KEY']
    API_SECRET = config['ZERODHA']['API_SECRET']

    end_date = datetime.now() - timedelta(days=1)
    start_date = end_date - timedelta(days=365*20)  # Last 20 years 


    get_index_constituents()
    print("Index constituents updated.")
    load_data_from_zerodha(API_KEY, API_SECRET, end_date, start_date)
    print("Stock constituents updated.")
    all_trading_dates = getAllTradingDates()
    trading_dates = all_trading_dates.loc[((all_trading_dates>=start_date.date()) & (all_trading_dates<=end_date.date()))]
    downloadPastBhavCopies(trading_dates)









