import FetchAndCleanData



if __name__ == '__main__':
    eodData = getLast10YrsAdjustedEODData(DATA_FLRD)
    oiData = FetchAndCleanData.getFeaturesOIDataForLast6Months()
    