
import pandas as pd
import itertools
from collections import Counter, defaultdict
import numpy as np
import statsmodels.formula.api as smf
from sklearn.linear_model import LinearRegression
import scipy

import os

''' follow the instructions #for FF3 to modify the code for the 3-factor model testing '''
os.chdir('set working directory')

Returns = pd.read_csv('path to FFC_Returns.csv')
Returns.set_index('Date', inplace = True)

Risk_Free = pd.read_csv('path to Europe_RF_Daily.csv')
Risk_Free['Date'] = pd.to_datetime(Risk_Free['Date'])
Risk_Free.set_index('Date', inplace = True)

Returns = Returns.join(Risk_Free)
Returns = Returns.subtract(Returns['Yield'], axis=0)
del Returns['Yield']


Returns_2nd = pd.read_csv('path to FFC_Returns_2nd.csv')
Returns_2nd['Date'] = pd.to_datetime(Returns_2nd['Date'])
Returns_2nd.set_index('Date', inplace = True)

Returns_2nd = Returns_2nd.join(Risk_Free)
Returns_2nd = Returns_2nd.subtract(Returns_2nd['Yield'], axis=0)
del Returns_2nd['Yield']
Returns_2nd = Returns_2nd.dropna(how = 'any',axis = 0)

FF3 = pd.read_csv('path to FF3_Factors.csv')
FF3.set_index('Date', inplace = True)
#For FF3 remove the below two lines 
MOM = pd.read_csv('path to MOM_Factor.csv')
MOM.set_index('Date', inplace = True) 

# For FF3 change the below to -> Factors = FF3
Factors = FF3.join(MOM) for FFC

FF3_2nd = pd.read_csv('path to FF3_Factors_2nd_Stage.csv')
FF3_2nd.set_index('Date', inplace = True)
#for ff3 remove the below two lines
MOM_2nd = pd.read_csv('path to MOM_Factor_2nd.csv')
MOM_2nd.set_index('Date', inplace = True)

#for ff3 change the below to > Factors_2nd = FF3
Factors_2nd = FF3_2nd.join(MOM_2nd)


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
    S = np.array([])
    H = np.array([])
    M = np.array([]) #remove this line for FF3
    Rsquared = np.array([])
    names = names
    for i in names:
        
        f = str(str(i) + ' ~ Mkt + SMB + HML + MOM')
        est = smf.ols(formula=f, data=data).fit()
        #print(est.summary())
        #f = open('C4_Time_Series.tex', 'a')
        A = np.r_[A,est.params[0]]
        B = np.r_[B,est.params[1]]
        S = np.r_[S,est.params[2]]
        H = np.r_[H,est.params[3]]
        M = np.r_[M,est.params[4]] #remove this line for FF3


        #f.write(est.summary().as_latex())
        #f.close()     
    
    return B,S,H,M #remove M for FF3

B,S,H,M = time_series(Data, Portfolios) #remove M for FF3

Factor_Estimates = np.vstack((B,S,H,M))
Factor_Estimates = pd.DataFrame(Factor_Estimates.T, columns = ['MKT','SMB','HML','MOM']) #remove 'MOM' for FF3

Returns_1st_Period = np.transpose(Returns_2nd[(Returns_2nd.index.date <= first_end) & (Returns_2nd.index.date >= first_start)]).reset_index(drop = True)
Returns_2nd_Period = np.transpose(Returns_2nd[(Returns_2nd.index.date <= second_end) & (Returns_2nd.index.date >= second_start)]).reset_index(drop = True)
Returns_3rd_Period = np.transpose(Returns_2nd[(Returns_2nd.index.date <= third_end) & (Returns_2nd.index.date >= third_start)]).reset_index(drop = True)
''' Write for GRS 
GRS_1 = Returns_2nd[(Returns_2nd.index.date <= first_end) & (Returns_2nd.index.date >= first_start)].join(Factors_2nd)
GRS_2 = Returns_2nd[(Returns_2nd.index.date <= second_end) & (Returns_2nd.index.date >= second_start)].join(Factors_2nd)
GRS_3 = Returns_2nd[(Returns_2nd.index.date <= third_end) & (Returns_2nd.index.date >= third_start)].join(Factors_2nd)

GRS_1.to_csv('path to write MOM_For_GRS_1.csv')
GRS_2.to_csv('path to write MOM_For_GRS_2.csv')
GRS_3.to_csv('path to write MOM_For_GRS_3.csv')
'''
def cross_section(matrix, factors):
   
    alpha = np.array([])
    gamma_mkt = np.array([])
    gamma_smb = np.array([])
    gamma_hml = np.array([])
    gamma_mom = np.array([]) #remove this line for FF3
    R = np.array([])
    for i in range(matrix.shape[1]):
        model = LinearRegression()
        model.fit(factors,matrix.iloc[:,i])
        alpha = np.r_[alpha, model.intercept_] #get alphas
        gamma_mkt = np.r_[gamma_mkt, model.coef_[0]] #get lambdas
        gamma_smb = np.r_[gamma_smb, model.coef_[1]]
        gamma_hml = np.r_[gamma_hml, model.coef_[2]]
        gamma_mom = np.r_[gamma_mom, model.coef_[3]] #remove this line for FF3
        yhat = model.predict(factors)
        SS_Residual = sum((matrix.iloc[:,i]-yhat)**2)       
        SS_Total = sum((matrix.iloc[:,i]-np.mean(matrix.iloc[:,i]))**2)     
        r_squared = 1 - (float(SS_Residual))/SS_Total#get lambdas
        R = np.r_[R, r_squared]
    R = np.mean(R)
    return alpha, gamma_mkt, gamma_smb, gamma_hml, gamma_mom, R #remove gamma_mom  for FF3




#remove gamma_mom 1, 2 and 3 for FF3
alphas_1, gamma_mkt_1, gamma_smb_1, gamma_hml_1, gamma_mom_1, R_1  = cross_section(Returns_1st_Period, Factor_Estimates)
alphas_2, gamma_mkt_2, gamma_smb_2, gamma_hml_2, gamma_mom_2, R_2 = cross_section(Returns_2nd_Period, Factor_Estimates)
alphas_3, gamma_mkt_3, gamma_smb_3, gamma_hml_3, gamma_mom_3, R_3 = cross_section(Returns_3rd_Period, Factor_Estimates)

print(scipy.stats.ttest_1samp(alphas_1,popmean = 0, axis = 0))
print(scipy.stats.ttest_1samp(gamma_mkt_1,popmean = 0, axis = 0))
print(scipy.stats.ttest_1samp(gamma_smb_1,popmean = 0, axis = 0))
print(scipy.stats.ttest_1samp(gamma_hml_1,popmean = 0, axis = 0))
print(scipy.stats.ttest_1samp(gamma_mom_1,popmean = 0, axis = 0)) #remove this line for FF3

print(scipy.stats.ttest_1samp(alphas_2,popmean = 0, axis = 0))
print(scipy.stats.ttest_1samp(gamma_mkt_2,popmean = 0, axis = 0))
print(scipy.stats.ttest_1samp(gamma_smb_2,popmean = 0, axis = 0))
print(scipy.stats.ttest_1samp(gamma_hml_2,popmean = 0, axis = 0))
print(scipy.stats.ttest_1samp(gamma_mom_2,popmean = 0, axis = 0)) #remove this line for FF3

print(scipy.stats.ttest_1samp(alphas_3,popmean = 0, axis = 0))
print(scipy.stats.ttest_1samp(gamma_mkt_3,popmean = 0, axis = 0))
print(scipy.stats.ttest_1samp(gamma_smb_3,popmean = 0, axis = 0))
print(scipy.stats.ttest_1samp(gamma_hml_3,popmean = 0, axis = 0))
print(scipy.stats.ttest_1samp(gamma_mom_2,popmean = 0, axis = 0)) #remove this line for FF3

print(np.mean(alphas_1),np.mean(alphas_2), np.mean(alphas_3))
print(np.mean(gamma_mkt_1),np.mean(gamma_mkt_2), np.mean(gamma_mkt_3))
print(np.mean(gamma_smb_1),np.mean(gamma_smb_2), np.mean(gamma_smb_3))
print(np.mean(gamma_hml_1),np.mean(gamma_hml_2), np.mean(gamma_hml_3))
print(np.mean(gamma_mom_1),np.mean(gamma_mom_2), np.mean(gamma_mom_3)) #remove this line for FF3

print(R_1,R_2,R_3)