
import pandas as pd
import itertools
from collections import Counter, defaultdict
import numpy as np
import statsmodels.formula.api as smf
from sklearn.linear_model import LinearRegression
import scipy

import os

os.chdir('change working directory')

Returns = pd.read_csv('path to MOM_Returns.csv')
Returns.set_index('Date', inplace = True)

Risk_Free = pd.read_csv(r'C:/Users/zoshk/OneDrive - University of Limerick/FYP/Data/Q_Factor/Europe_RF_Daily.csv')
Risk_Free['Date'] = pd.to_datetime(Risk_Free['Date'])
Risk_Free.set_index('Date', inplace = True)

Returns = Returns.join(Risk_Free)
Returns = Returns.subtract(Returns['Yield'], axis=0)
del Returns['Yield']


Returns_2nd = pd.read_csv('path to MOM_Returns_2nd.csv')
Returns_2nd['Date'] = pd.to_datetime(Returns_2nd['Date'])
Returns_2nd.set_index('Date', inplace = True)

Returns_2nd = Returns_2nd.join(Risk_Free)
Returns_2nd = Returns_2nd.subtract(Returns_2nd['Yield'], axis=0)
del Returns_2nd['Yield']
Returns_2nd = Returns_2nd.dropna(how = 'any',axis = 0)

FF3 = pd.read_csv('path to FF3_Factors.csv')
FF3.set_index('Date', inplace = True)
MOM = pd.read_csv('path to MOM_Factor.csv')
MOM.set_index('Date', inplace = True)

Factors = FF3.join(MOM)

FF3_2nd = pd.read_csv('path to FF3_Factors_2nd_Stage.csv')
FF3_2nd.set_index('Date', inplace = True)
MOM_2nd = pd.read_csv(r'path to MOM_Factor_2nd.csv')
MOM_2nd.set_index('Date', inplace = True)

Factors_2nd = FF3_2nd.join(MOM_2nd)

#remove redundant factors for the CAPM
del Factors['SMB']
del Factors['HML']
del Factors['MOM']

del Factors_2nd['SMB']
del Factors_2nd['HML']
del Factors_2nd['MOM']

Data = Returns.join(Factors)
Data2 = Returns_2nd.join(Factors_2nd)
Portfolios = Returns.columns.to_list()

first_start = pd.to_datetime('1.1.2020')
first_end = pd.to_datetime('3.23.2020')

second_start = pd.to_datetime('3.24.2020')
second_end = pd.to_datetime('10.30.2020')

third_start = pd.to_datetime('10.31.2020')
third_end = pd.to_datetime('12.31.2020')

Data = Returns.join(Factors)

def time_series(data, names):
    data = data[(data.index.date <= pd.to_datetime('12.31.2019')) & (data.index.date >= pd.to_datetime('1.1.2016'))]

    B = np.array([])
    names = names
    for i in names:
        
        f = str(str(i) + ' ~ Mkt')
        est = smf.ols(formula=f, data=data).fit()
        #print(est.summary())
        #f = open('MOM_Sorted_CAPM.tex', 'a')
        B = np.r_[B,est.params[1]]
        #f.write(est.summary().as_latex())
        #f.close()     
    
    return B

B = time_series(Data, Portfolios)


Factor_Estimates = pd.DataFrame(B, columns = ['MKT'])

Returns_1st_Period = np.transpose(Returns_2nd[(Returns_2nd.index.date <= first_end) & (Returns_2nd.index.date >= first_start)]).reset_index(drop = True)
Returns_2nd_Period = np.transpose(Returns_2nd[(Returns_2nd.index.date <= second_end) & (Returns_2nd.index.date >= second_start)]).reset_index(drop = True)
Returns_3rd_Period = np.transpose(Returns_2nd[(Returns_2nd.index.date <= third_end) & (Returns_2nd.index.date >= third_start)]).reset_index(drop = True)

def prepare_data_second_stage(source,first_start,first_end,second_start,second_end,third_start,third_end): 
    #Clean data
    source = source.replace([np.inf, -np.inf], np.nan)
    source = source.dropna(axis = 0, how = 'any')
    source = source[(source.index.date < pd.to_datetime('1.1.2021')) & (source.index.date >= pd.to_datetime('1.1.2020'))]
    GRS_1 = source[(source.index.date <= first_end) & (source.index.date >= first_start)]
    GRS_2 = source[(source.index.date <= second_end) & (source.index.date >= second_start)]
    GRS_3 = source[(source.index.date <= third_end) & (source.index.date >= third_start)]
    GRS_1.to_csv('path to write MOM_Sorted_CAPM_1.csv')
    GRS_2.to_csv('path to write MOM_Sorted_CAPM_2.csv')
    GRS_3.to_csv('path to write MOM_Sorted_CAPM_3.csv')
 
#writecsv = prepare_data_second_stage(Data2, first_start, first_end, second_start, second_end, third_start, third_end)
def cross_section(matrix, factors):
   
    alpha = np.array([])
    gamma_mkt = np.array([])
    R = np.array([])
    for i in range(matrix.shape[1]):
        model = LinearRegression()
        model.fit(factors,matrix.iloc[:,i])
        alpha = np.r_[alpha, model.intercept_] #get alphas
        gamma_mkt = np.r_[gamma_mkt, model.coef_[0]]
        yhat = model.predict(factors)
        SS_Residual = sum((matrix.iloc[:,i]-yhat)**2)       
        SS_Total = sum((matrix.iloc[:,i]-np.mean(matrix.iloc[:,i]))**2)     
        r_squared = 1 - (float(SS_Residual))/SS_Total#get lambdas
        R = np.r_[R, r_squared]
    R = np.mean(R)
        
    return alpha, gamma_mkt, R






alphas_1, gamma_mkt_1, R_1 = cross_section(Returns_1st_Period, Factor_Estimates)
alphas_2, gamma_mkt_2, R_2 = cross_section(Returns_2nd_Period, Factor_Estimates)
alphas_3, gamma_mkt_3, R_3 = cross_section(Returns_3rd_Period, Factor_Estimates)


print(scipy.stats.ttest_1samp(alphas_1,popmean = 0, axis = 0))
print(scipy.stats.ttest_1samp(gamma_mkt_1,popmean = 0, axis = 0))

print(scipy.stats.ttest_1samp(alphas_2,popmean = 0, axis = 0))
print(scipy.stats.ttest_1samp(gamma_mkt_2,popmean = 0, axis = 0))

print(scipy.stats.ttest_1samp(alphas_3,popmean = 0, axis = 0))
print(scipy.stats.ttest_1samp(gamma_mkt_3,popmean = 0, axis = 0))



print(np.mean(alphas_1),np.mean(alphas_2), np.mean(alphas_3))
print(np.mean(gamma_mkt_1),np.mean(gamma_mkt_2), np.mean(gamma_mkt_3))

print(R_1, R_2, R_3)