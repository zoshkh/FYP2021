import pandas as pd

import numpy as np
import math
import statistics 
import statsmodels.formula.api as smf
import scipy
from sklearn.linear_model import LinearRegression
import os

os.chdir('path to working directory')
#estimate coeffiecient for the 2nd stage regression
def time_series(data, bquantile):
    names = []
    coefs = np.array([])
    C= np.array([])
    for i in range(1,bquantile+1):
        names.append(str('B'+str(i)))
    for i in names:
        try:
            f = str(str(i) + ' ~ T + T2+ T3') # lagged
            #f = str(str(i) + ' ~ T') # no lag
            est = smf.ols(formula=f, data=data).fit()
            #Write regression results in tex file
            f = open('CAPM_Lagged_Time_Series.tex', 'a')
            #C = np.r_[C,est.params[1:4]]
            B = est.params[1:4].sum() #lagged
            #B = est.params[1] #no lag
            coefs = np.r_[coefs,B]
            f.write(est.summary().as_latex())
            f.close()     
        except:
            pass
    
    return coefs #returns estimate betas

def prepare_data_second_stage(source,first_start,first_end,second_start,second_end,third_start,third_end): 
    #Clean data 
    source = source.replace([np.inf, -np.inf], np.nan)
    source = source.dropna(axis = 0, how = 'any')
    source = source[(source.index.date < pd.to_datetime('1.1.2021')) & (source.index.date >= pd.to_datetime('1.1.2020'))]
    #write files for GRS testing    
    GRS_1 = source[(source.index.date <= first_end) & (source.index.date >= first_start)]
    GRS_2 = source[(source.index.date <= second_end) & (source.index.date >= second_start)]
    GRS_3 = source[(source.index.date <= third_end) & (source.index.date >= third_start)]
    GRS_1.to_csv('~ path to write CAPM_GRS_1.csv')
    GRS_2.to_csv('~ path to write CAPM_GRS_2.csv')
    GRS_3.to_csv('~ path to write CAPM_GRS_3.csv')    
    #Writes data for GRS Tests, which is performed in R
    source.drop(columns=['T','T2','T3'], inplace=True) #remove market so we can run cross sectional regr
    #source.drop(columns='T', inplace=True) #no lag
    return source


def cross_section(matrix, start, end):
    matrix = matrix[(matrix.index.date <= end) & (matrix.index.date >= start)] #modifys window according to dates defined below
    alpha = np.array([])
    lambd = np.array([])
    R = np.array([])
    for i in range(matrix.shape[0]):
        model = LinearRegression()
        model.fit(coefs,matrix.iloc[i,:])
        alpha = np.r_[alpha, model.intercept_] #get alphas
        lambd = np.r_[lambd, model.coef_]
        #calculate R-squared
        yhat = model.predict(coefs)
        SS_Residual = sum((matrix.iloc[i,:]-yhat)**2)       
        SS_Total = sum((matrix.iloc[i,:]-np.mean(matrix.iloc[i,:]))**2)     
        r_squared = 1 - (float(SS_Residual))/SS_Total#get lambdas
        R = np.r_[R, r_squared]
    R = np.mean(R)
    return matrix, alpha, lambd, R
def get_cross_sectional_results(intercept,lamb): #this is essentialy redundant as we use different method below
    #Clean arrays
    intercept = [i for i in intercept if str(i) != 'nan']
    intercept = [i for i in intercept if math.isinf(i) == False]
    lamb = [i for i in lamb if str(i) != 'nan']
    lamb = [i for i in lamb if math.isinf(i) == False]
    
    #estimate means and perform t-test
    a = np.mean(intercept)
    l = np.mean(lamb)
    stda = statistics.stdev(intercept)
    stdg = statistics.stdev(lamb)
    ta = a/(stda/np.sqrt(len(intercept)))
    tg = l/(stdg/np.sqrt(len(lamb)))
    return ta, tg




quantiles = 25

data_first_stage = pd.read_csv('path to Data_1.csv')
data_second_stage = pd.read_csv('path to Data_2.csv')


#written data has a date column name Unnamed: 0, we have to modify it, if your file has a different name change the code to that name

data_first_stage['Unnamed: 0'] = pd.to_datetime(data_first_stage['Unnamed: 0'])
data_first_stage = data_first_stage.set_index('Unnamed: 0')
data_first_stage = data_first_stage*100 #to make market and portfoliso returns decimals uniform

data_second_stage['Unnamed: 0'] = pd.to_datetime(data_second_stage['Unnamed: 0'])
data_second_stage = data_second_stage.set_index('Unnamed: 0')
data_second_stage = data_second_stage*100


mkt_1 = pd.read_csv('path to Mkt_1.csv')
mkt_2 = pd.read_csv('path to Mkt_2.csv')

mkt_1.set_index('Date', inplace = True)
mkt_2.set_index('Date', inplace = True)

data_first_stage = data_first_stage.join(mkt_1)
data_second_stage = data_second_stage.join(mkt_2)

#get estimate betas and transpose
coefs = time_series(data_first_stage, quantiles)
coefs = coefs.reshape(-1,1)

#cross-section window, we modify it to 3 periods and run regression n times for each period wheere n is number of days
first_start = pd.to_datetime('1.1.2020')
first_end = pd.to_datetime('3.23.2020')

second_start = pd.to_datetime('3.24.2020')
second_end = pd.to_datetime('10.30.2020')

third_start = pd.to_datetime('10.31.2020')
third_end = pd.to_datetime('12.31.2020')



data_second_stage = prepare_data_second_stage(data_second_stage, first_start, first_end, second_start, second_end, third_start, third_end)

#should be named gammas instead of lambdas but doesn't really matter
first, alpha_1, lamb_1, R_1 = cross_section(data_second_stage, first_start, first_end)
second, alpha_2, lamb_2, R_2 = cross_section(data_second_stage, second_start, second_end)
third, alpha_3, lamb_3,R_3 = cross_section(data_second_stage, third_start, third_end)


#t-test using function
print(scipy.stats.ttest_1samp(alpha_1,popmean = 0, axis = 0))
print(scipy.stats.ttest_1samp(lamb_1,popmean = 0, axis = 0))
print(scipy.stats.ttest_1samp(alpha_2,popmean = 0, axis = 0))
print(scipy.stats.ttest_1samp(lamb_2,popmean = 0, axis = 0))
print(scipy.stats.ttest_1samp(alpha_3,popmean = 0, axis = 0))
print(scipy.stats.ttest_1samp(lamb_3,popmean = 0, axis = 0))

print(np.mean(alpha_1),np.mean(alpha_2), np.mean(alpha_3))
print(np.mean(lamb_1),np.mean(lamb_2), np.mean(lamb_3))

