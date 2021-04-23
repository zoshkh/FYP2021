
import pandas as pd
import itertools
from collections import Counter, defaultdict
import numpy as np
import statsmodels.formula.api as smf
from sklearn.linear_model import LinearRegression
import scipy

import os

''' Follow the instructions #For FFC to modify the code for 4-factor model testing '''
os.chdir('set working directory')
Returns = pd.read_csv('path to FF3_Returns.csv')
Returns.set_index('Date', inplace = True)

Risk_Free = pd.read_csv('path to Europe_RF_Daily.csv')
Risk_Free['Date'] = pd.to_datetime(Risk_Free['Date'])
Risk_Free.set_index('Date', inplace = True)
Risk_Free = Risk_Free

Returns = Returns.join(Risk_Free)
Returns = Returns.subtract(Returns['Yield'], axis=0)
del Returns['Yield']
Returns = Returns.dropna(how = 'any',axis = 0)

Returns_2nd = pd.read_csv('path to FF3_Returns_2nd_Stage.csv')
Returns_2nd['Date'] = pd.to_datetime(Returns_2nd['Date'])
Returns_2nd.set_index('Date', inplace = True)

Returns_2nd = Returns_2nd.join(Risk_Free)
Returns_2nd = Returns_2nd.subtract(Returns_2nd['Yield'], axis=0)
del Returns_2nd['Yield']
Returns_2nd = Returns_2nd.dropna(how = 'any',axis = 0)


FF3 = pd.read_csv('path to FF3_Factors.csv')
FF3.set_index('Date', inplace = True)
###### FOR FFC #####
#MOM = pd.read_csv('path to MOM_Factor.csv')
#MOM.set_index('Date', inplace = True)

Factors = FF3
#Factors = FF3.join(MOM)

FF3_2nd = pd.read_csv('path to FF3_Factors_2nd_Stage.csv')
FF3_2nd.set_index('Date', inplace = True)
###### FOR FFC #####
#MOM_2nd = pd.read_csv('path to MOM_Factor_2nd.csv')
#MOM_2nd.set_index('Date', inplace = True)

Factors_2nd = FF3_2nd
#Factors_2nd = FF3_2nd.join(MOM_2nd) for FFC


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
    #M = np.array([]) for FFC
    names = names
    for i in names:
        
        f = str(str(i) + ' ~ Mkt + SMB + HML') #add 'MOM' for FFC
        est = smf.ols(formula=f, data=data).fit()
        #print(est.summary())
        #f = open('BM_Sorted_C4.tex', 'a')
        B = np.r_[B,est.params[1]]
        S = np.r_[S,est.params[2]]
        H = np.r_[H,est.params[3]]
        #M = np.r_[M,est.params[4]] for ffc

        #f.write(est.summary().as_latex())
        #f.close()     
    
    return B,S,H  #add M for FFC

B,S,H = time_series(Data, Portfolios) #add M for FFC

Factor_Estimates = np.vstack((B,S,H)) #add M for FFC
Factor_Estimates = pd.DataFrame(Factor_Estimates.T, columns = ['MKT','SMB','HML']) #add 'MOM' for FFC

Returns_1st_Period = np.transpose(Returns_2nd[(Returns_2nd.index.date <= first_end) & (Returns_2nd.index.date >= first_start)]).reset_index(drop = True)
Returns_2nd_Period = np.transpose(Returns_2nd[(Returns_2nd.index.date <= second_end) & (Returns_2nd.index.date >= second_start)]).reset_index(drop = True)
Returns_3rd_Period = np.transpose(Returns_2nd[(Returns_2nd.index.date <= third_end) & (Returns_2nd.index.date >= third_start)]).reset_index(drop = True)
''' Write GRS
GRS_1 = Returns_2nd[(Returns_2nd.index.date <= first_end) & (Returns_2nd.index.date >= first_start)].join(Factors_2nd)
GRS_2 = Returns_2nd[(Returns_2nd.index.date <= second_end) & (Returns_2nd.index.date >= second_start)].join(Factors_2nd)
GRS_3 = Returns_2nd[(Returns_2nd.index.date <= third_end) & (Returns_2nd.index.date >= third_start)].join(Factors_2nd)

GRS_1.to_csv('path for BM_Sorted_C4_1.csv')
GRS_2.to_csv('path for BM_Sorted_C4_2.csv')
GRS_3.to_csv('path for BM_Sorted_C4_3.csv')
'''
def cross_section(matrix, factors):
   
    alpha = np.array([])
    gamma_mkt = np.array([])
    gamma_smb = np.array([])
    gamma_hml = np.array([])
    #gamma_mom = np.array([]) #for ffc
    R = np.array([])
    for i in range(matrix.shape[1]):
        model = LinearRegression()
        model.fit(factors,matrix.iloc[:,i])
        alpha = np.r_[alpha, model.intercept_] #get alphas
        gamma_mkt = np.r_[gamma_mkt, model.coef_[0]] #get lambdas
        gamma_smb = np.r_[gamma_smb, model.coef_[1]]
        gamma_hml = np.r_[gamma_hml, model.coef_[2]]
        #gamma_mom = np.r_[gamma_mom, model.coef_[3]] #for ffc
        yhat = model.predict(factors)
        SS_Residual = sum((matrix.iloc[:,i]-yhat)**2)       
        SS_Total = sum((matrix.iloc[:,i]-np.mean(matrix.iloc[:,i]))**2)     
        r_squared = 1 - (float(SS_Residual))/SS_Total#get lambdas
        R = np.r_[R, r_squared]
    R = np.mean(R)
    return alpha, gamma_mkt, gamma_smb, gamma_hml, R #gamma_mom for FFC




#For FFC add gamma_mom 
alphas_1, gamma_mkt_1, gamma_smb_1, gamma_hml_1, R_1  = cross_section(Returns_1st_Period, Factor_Estimates) #gamma_mom_1
alphas_2, gamma_mkt_2, gamma_smb_2, gamma_hml_2, R_2 = cross_section(Returns_2nd_Period, Factor_Estimates) #gamma_mom_2
alphas_3, gamma_mkt_3, gamma_smb_3, gamma_hml_3, R_3 = cross_section(Returns_3rd_Period, Factor_Estimates)#gamma_mom_3


print(scipy.stats.ttest_1samp(alphas_1,popmean = 0, axis = 0))
print(scipy.stats.ttest_1samp(gamma_mkt_1,popmean = 0, axis = 0))
print(scipy.stats.ttest_1samp(gamma_smb_1,popmean = 0, axis = 0))
print(scipy.stats.ttest_1samp(gamma_hml_1,popmean = 0, axis = 0))
#print(scipy.stats.ttest_1samp(gamma_mom_1,popmean = 0, axis = 0))

print(scipy.stats.ttest_1samp(alphas_2,popmean = 0, axis = 0))
print(scipy.stats.ttest_1samp(gamma_mkt_2,popmean = 0, axis = 0))
print(scipy.stats.ttest_1samp(gamma_smb_2,popmean = 0, axis = 0))
print(scipy.stats.ttest_1samp(gamma_hml_2,popmean = 0, axis = 0))
#print(scipy.stats.ttest_1samp(gamma_mom_2,popmean = 0, axis = 0))

print(scipy.stats.ttest_1samp(alphas_3,popmean = 0, axis = 0))
print(scipy.stats.ttest_1samp(gamma_mkt_3,popmean = 0, axis = 0))
print(scipy.stats.ttest_1samp(gamma_smb_3,popmean = 0, axis = 0))
print(scipy.stats.ttest_1samp(gamma_hml_3,popmean = 0, axis = 0))
#print(scipy.stats.ttest_1samp(gamma_mom_3,popmean = 0, axis = 0))

print(np.mean(alphas_1),np.mean(alphas_2), np.mean(alphas_3))
print(np.mean(gamma_mkt_1),np.mean(gamma_mkt_2), np.mean(gamma_mkt_3))
print(np.mean(gamma_smb_1),np.mean(gamma_smb_2), np.mean(gamma_smb_3))
print(np.mean(gamma_hml_1),np.mean(gamma_hml_2), np.mean(gamma_hml_3))
#print(np.mean(gamma_mom_1),np.mean(gamma_mom_2), np.mean(gamma_mom_3))

print(R_1,R_2,R_3)