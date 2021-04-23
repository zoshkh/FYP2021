import pandas as pd
import itertools
from collections import Counter, defaultdict
import numpy as np
import statsmodels.formula.api as smf


#This function removes digits and dots from tickers (colnames) and replaces them so no duplicate colnames exist
def modify_colnames(data): 
    tickers = data.columns.str.replace(r'\d+', '')
    counts = Counter(tickers)
    suffix_counter = defaultdict(lambda: itertools.count(1))
    tickers2 = [elem if counts[elem] == 1 else elem + f'_{next(suffix_counter[elem])}'
                for elem in tickers]
    data.colums = tickers2
    data.columns = data.columns.str.replace('.', '')
    return data

#Create one DataFrame using Euronext,UK,Germany, EAFE prices and substract risk-free for excess returns
def prepare_data(df, df2, df3, rf, index):
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date')
    
    df2['Date'] = pd.to_datetime(df2['Date'])
    df2 = df2.set_index('Date')
    del df2['^FTAS'] #UK file contains FTSE all share Index so it needs to be removed
    
    df3['Date'] = pd.to_datetime(df3['Date'])
    df3 = df3.set_index('Date')
    
    df = df.join(df2, how ='inner')
    df = df.join(df3, how ='inner')
    
    rf = rf[['Date','Yield']]
    rf['Date'] = pd.to_datetime(rf['Date'])
    rf.set_index('Date', inplace = True)
    rf = rf.sort_index(ascending = True)
    rf['Yield'] = np.log(1+rf['Yield']).fillna(0)
    
    index['Date'] = pd.to_datetime(index['Date'])
    index = index.set_index('Date')
    df = df.loc[:, pd.notnull(df).sum()>len(df)*.9] #filter tickers missing large data
    returns = df.pct_change()
    df = np.log(1+returns)
    df = df.iloc[1:]
    df = df.join(index, how ='inner')
    
    
    df = df.join(rf)
    df = df.subtract(df['Yield'], axis=0)
    del df['Yield']
    
    return df



#This method of beta calculation is much faster, but it is for one column, not-lagged, market returns
def compute_beta(column,proxy):
    covmat = np.cov(column, proxy)
    return covmat[0,1]/covmat[1,1]


def prepare_for_regression(raw_data, bquantile):
    #set up two empty dataframesL name columns -> B1 lowest Beta 25 highest beta
    names = []
    for i in range(1,bquantile+1):
        names.append(str('B'+str(i)))
    clean_data = pd.DataFrame(columns = names)
    clean_data_second = pd.DataFrame(columns = names)
    
    for i in range(2016,2021): #iterate through years to sort, e.g  2012-2016 to sort 2017, 2016-2019 to sort 2020
        data = raw_data[(raw_data.index.year < i) & (raw_data.index.year >= i-4)]
        data = data.dropna(how='all', axis=1)
        data = data.dropna(how='all', axis=0)
        returns = raw_data[(raw_data.index.year == i)]
        returns = returns.dropna(how='all', axis=1)
        returns = returns.dropna(how='all', axis=1)
        
        
        tickers = list(data.columns.values)
        tickers = tickers[:-3] #remove index returns from tickers
        beta_dict = {}
        
        #Run regression to estimate beta for each ticker: this runs ols for 1500 stocks so it may take some time
        for t in tickers:
            try:
                
                f = str(str(t)+' ~ T')
                est = smf.ols(formula=f, data=data).fit()
                B = est.params[1]
                beta_dict[t] = B
            except: 
                pass
        
        Beta = pd.DataFrame.from_dict(beta_dict, orient = 'index')
        Beta.rename(columns={0:'Beta'}, inplace=True)
        Beta['Quantile']= pd.qcut(Beta['Beta'],  q = bquantile, labels = False)#cut in 25 groups 
        Beta = Beta.reset_index(0).reset_index(drop=True)
        Beta = Beta.pivot(columns = 'Quantile', values = 'index')
        Beta = Beta.apply(lambda x: pd.Series(x.dropna().values))
        Beta = Beta.dropna(how='any', axis=0)
        Beta.columns = names
        #at this point Beta dataframe has B1 to b25 colnames and each column contains respective tickers
        Bdict = Beta.to_dict(orient = 'list')
        colnames = []
        
        for key,value in Bdict.items():
            colnames.append(key)
        port = pd.DataFrame(columns = colnames)
        #pull tickers returns for each column and calculate average return (equally weighted port per beta group)
        for key,value in Bdict.items():
            port[key] = returns[Bdict[key]].sum(axis=1)/(sum(map(len, Bdict.values()))/bquantile)
    
        
        if i < 2020: #for time series
            clean_data = clean_data.append(port)
        else: #for cross-section
            clean_data_second = clean_data_second.append(port)

    return clean_data, clean_data_second #clean_data, clean_data_second


#Define how many groups of beta, if 5 divides into quantiles, if 20 divides into 20 groups B1:B20
quantiles = 25

#Read index returns and price files
EAFE = pd.read_csv('~ path to EAFE.csv')
Germany = pd.read_csv('~ path to Germany_Prices.csv')
Euronext = pd.read_csv('~ path to Euronext_Prices.csv')
UK = pd.read_csv('~ path to UK_Prices.csv')
rf = pd.read_csv('~ path to Europe_RF_daily.csv')
Germany = modify_colnames(Germany)
Euronext = modify_colnames(Euronext)
UK = modify_colnames(UK)


df = prepare_data(Euronext,UK,Germany,rf, EAFE)


data_first_stage, data_second_stage = prepare_for_regression(df, quantiles)
data_first_stage.to_csv('~ path to write Data_1.csv')
data_second_stage.to_csv('~ path to write Data_2.csv')
