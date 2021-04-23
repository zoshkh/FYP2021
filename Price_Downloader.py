import pandas as pd
import yfinance as yf
import datetime


#Define data donwload Function
def get_data(tickers, start_date, end_date):
    tickers = tickers['Tickers'].tolist()
    '''
    #Use when donwloading FTSE share prices using the FTSE ticker file
    tickers = [s.replace('\n', '') for s in tickers]
    tickers = [s.replace('.', '') for s in tickers]
    mystring = ".L"        
    for n, i in enumerate(tickers):
                tickers[n] = i + mystring
    '''
    
    data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
    return data

#define download period
start = datetime.datetime(2010,1,1)
end = datetime.datetime(2021,1,1)

#Read tickers files
Germany = pd.read_csv('~path to Germany_Tickers.csv')
UK = pd.read_csv('~path to UK_Tickers.csv')
Euronext = pd.read_csv('~path to Euronext_Tickers.csv')
#Some manipulation to get exchange symbols at the end of tickers for YahooFinance
Euronext.loc[Euronext['Market'].str.contains('Paris'), 'Market'] = '.PA'
Euronext.loc[Euronext['Market'].str.contains('Oslo'), 'Market'] = '.OL'
Euronext.loc[Euronext['Market'].str.contains('Amsterdam'), 'Market'] = '.AS'
Euronext.loc[Euronext['Market'].str.contains('Brussels'), 'Market'] = '.BR'
Euronext.loc[Euronext['Market'].str.contains('Lisbon'), 'Market'] = '.LS'
Euronext.loc[Euronext['Market'].str.contains('Dublin'), 'Market'] = '.IR'
Euronext["Tickers"] = Euronext["Tickers"] + Euronext["Market"]
del Euronext['Market']


#Download Data - total data size is around 100MB
Germany_Prices = get_data(Germany, start, end)
Euronext_Prices = get_data(Euronext, start, end)
UK_Prices = get_data(UK, start, end)

#Write Data
Germany_Prices.to_csv('~path')
Euronext_Prices.to_csv('~path')
UK_Prices.to_csv('~path')
