import FetchAndCleanData
import pandas as pd
def check():
    try:
        pd.read_csv('')
    except:
        print('error')
    print('out')

if __name__ == '__main__':
    DATA_FLRD = r'C:\Users\ksdee\Documents\PersonalFinance\Trading\Trading_Data'
    #eodData = getLast10YrsAdjustedEODData(DATA_FLRD)
    #oiData = FetchAndCleanData.getFeaturesOIDataForLast6Months()
    FetchAndCleanData.getFIIInvestmentData(DATA_FLRD)
    