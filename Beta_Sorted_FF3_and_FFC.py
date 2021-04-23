import pandas as pd
import itertools
from collections import Counter, defaultdict
import numpy as np
import math
import statistics 
import statsmodels.formula.api as smf
import scipy
from sklearn.linear_model import LinearRegression
import os


os.chdir('change directory')
data_first_stage = pd.read_csv('~Path to first stage file')
data_first_stage['Unnamed: 0'] = pd.to_datetime(data_first_stage['Unnamed: 0'])
data_first_stage = data_first_stage.set_index('Unnamed: 0')
data_second_stage = pd.read_csv('~Path to second stage file')
data_first_stage = data_first_stage*100
data_second_stage['Unnamed: 0'] = pd.to_datetime(data_second_stage['Unnamed: 0'])
data_second_stage = data_second_stage.set_index('Unnamed: 0')
data_second_stage = data_second_stage*100
data_first_stage = data_first_stage.iloc[:,:-3]
data_second_stage = data_second_stage.iloc[:,:-3]

Factors = pd.read_csv('path to FF3_Factors.csv')
Factors.set_index('Date', inplace = True)
Factors_2nd = pd.read_csv('path to FF3_Factors_2nd_Stage.csv')
Factors_2nd.set_index('Date', inplace = True)
''' for FFC remove: ' ' ' 
MOM = pd.read_csv('path to MOM_Factor.csv')
MOM.set_index('Date', inplace = True)
MOM_2nd = pd.read_csv('path to MOM_Factor_2nd.csv')
MOM_2nd.set_index('Date', inplace = True)


Factors = Factors.join(MOM)
Factors_2nd = Factors_2nd.join(MOM_2nd)

Data = data_first_stage.join(Factors)
Data2 = data_second_stage.join(Factors_2nd)
'''

Data = data_first_stage.join(Factors)
Data2 = data_second_stage.join(Factors_2nd)

Portfolios = data_first_stage.columns.to_list()

first_start = pd.to_datetime('1.1.2020')
first_end = pd.to_datetime('3.23.2020')

second_start = pd.to_datetime('3.24.2020')
second_end = pd.to_datetime('10.30.2020')

third_start = pd.to_datetime('10.31.2020')
third_end = pd.to_datetime('12.31.2020')

def time_series(data, names):
    data = data[(data.index.date <= pd.to_datetime('12.31.2019')) & (data.index.date >= pd.to_datetime('1.1.2016'))]
    A = np.array([])
    B = np.array([])
    S = np.array([])
    H = np.array([])
    #M = np.array([]) for FFC
    Rsquared = np.array([])
    names = names
    for i in names:
        
        f = str(str(i) + ' ~ Mkt + SMB + HML') #add MOM for FFC
        est = smf.ols(formula=f, data=data).fit()
        #print(est.summary())
        #f = open('Beta_Sorted_FF3_Time_Series.tex', 'a')
        A = np.r_[A,est.params[0]]
        B = np.r_[B,est.params[1]]
        S = np.r_[S,est.params[2]]
        H = np.r_[H,est.params[3]]
        #M = np.r_[H,est.params[3]] # for FFC
            #B = est.params[1] #no lag

        Rsquared = np.r_[Rsquared,est.rsquared]

        #f.write(est.summary().as_latex())
        #f.close()     
    
    return B,S,H #M for FFC

B,S,H = time_series(Data, Portfolios) #M for FFC

Factor_Estimates = np.vstack((B,S,H)) #M for FFC
Factor_Estimates = pd.DataFrame(Factor_Estimates.T, columns = ['MKT','SMB','HML']) #add 'MOM' for FFC

Returns_1st_Period = np.transpose(data_second_stage[(data_second_stage.index.date <= first_end) & (data_second_stage.index.date >= first_start)]).reset_index(drop = True)
Returns_2nd_Period = np.transpose(data_second_stage[(data_second_stage.index.date <= second_end) & (data_second_stage.index.date >= second_start)]).reset_index(drop = True)
Returns_3rd_Period = np.transpose(data_second_stage[(data_second_stage.index.date <= third_end) & (data_second_stage.index.date >= third_start)]).reset_index(drop = True)

def prepare_data_second_stage(source,first_start,first_end,second_start,second_end,third_start,third_end): 
    #Clean data
    source = source.replace([np.inf, -np.inf], np.nan)
    source = source.dropna(axis = 0, how = 'any')
    source = source[(source.index.date < pd.to_datetime('1.1.2021')) & (source.index.date >= pd.to_datetime('1.1.2020'))]
    GRS_1 = source[(source.index.date <= first_end) & (source.index.date >= first_start)]
    GRS_2 = source[(source.index.date <= second_end) & (source.index.date >= second_start)]
    GRS_3 = source[(source.index.date <= third_end) & (source.index.date >= third_start)]
    GRS_1.to_csv('path for Beta_Sorted_FF3_1.csv')
    GRS_2.to_csv('path for Beta_Sorted_FF3_2.csv')
    GRS_3.to_csv('path for Beta_Sorted_FF3_3.csv')


#Write for GRS
#writecsv = prepare_data_second_stage(Data2, first_start, first_end, second_start, second_end, third_start, third_end)


def remove_nas(data):
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.dropna(axis = 1, how = 'any')
    return data

Returns_1st_Period = remove_nas(Returns_1st_Period)

def cross_section(matrix, factors):
   
    alpha = np.array([])
    gamma_mkt = np.array([])
    gamma_smb = np.array([])
    gamma_hml = np.array([])
    #gamma_mom = np.array([])
    R = np.array([])
    for i in range(matrix.shape[1]):
        model = LinearRegression()
        model.fit(factors,matrix.iloc[:,i])
        alpha = np.r_[alpha, model.intercept_] #get alphas
        gamma_mkt = np.r_[gamma_mkt, model.coef_[0]] #get lambdas
        gamma_smb = np.r_[gamma_smb, model.coef_[1]]
        gamma_hml = np.r_[gamma_hml, model.coef_[2]]
        #gamma_hml = np.r_[gamma_hml, model.coef_[3]] #for ffc
        yhat = model.predict(factors)
        SS_Residual = sum((matrix.iloc[:,i]-yhat)**2)       
        SS_Total = sum((matrix.iloc[:,i]-np.mean(matrix.iloc[:,i]))**2)     
        r_squared = 1 - (float(SS_Residual))/SS_Total#get lambdas
        R = np.r_[R, r_squared]
    R = np.mean(R)
    return alpha, gamma_mkt, gamma_smb, gamma_hml, R #add gamma_mom for FFC





alphas_1, gamma_mkt_1, gamma_smb_1, gamma_hml_1, R_1  = cross_section(Returns_1st_Period, Factor_Estimates) #add gamma_mom for FFC
alphas_2, gamma_mkt_2, gamma_smb_2, gamma_hml_2, R_2  = cross_section(Returns_2nd_Period, Factor_Estimates)#add gamma_mom for FFC
alphas_3, gamma_mkt_3, gamma_smb_3, gamma_hml_3, R_3  = cross_section(Returns_3rd_Period, Factor_Estimates)#add gamma_mom for FFC


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
#print(scipy.stats.ttest_1samp(gamma_mom_2,popmean = 0, axis = 0))

print(np.mean(alphas_1),np.mean(alphas_2), np.mean(alphas_3))
print(np.mean(gamma_mkt_1),np.mean(gamma_mkt_2), np.mean(gamma_mkt_3))
print(np.mean(gamma_smb_1),np.mean(gamma_smb_2), np.mean(gamma_smb_3))
print(np.mean(gamma_hml_1),np.mean(gamma_hml_2), np.mean(gamma_hml_3))
#print(np.mean(gamma_mom_1),np.mean(gamma_mom_2), np.mean(gamma_mom_3))
print(R_1, R_2, R_3)
