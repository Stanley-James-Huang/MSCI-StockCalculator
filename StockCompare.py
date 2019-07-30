import pandas as pd
import numpy as np
from datetime import datetime
import pandas_datareader as pdr
import matplotlib.pyplot as plt
#include versions of the headers

#Prompt user
ticker1 = input("Please enter ticker 1: ")
ticker2 = input("Please enter ticker 2: ")

#Fetch and format data
marketData = pdr.get_data_yahoo('^GSPC',start=datetime(2018, 6, 1), end=datetime(2019, 6, 1),interval='m')
data1 = pdr.get_data_yahoo(ticker1,start=datetime(2018, 6, 1), end=datetime(2019, 6, 1),interval='m')
data2 = pdr.get_data_yahoo(ticker2,start=datetime(2018, 6, 1), end=datetime(2019, 6, 1),interval='m')

data1 = data1['Adj Close']
data2 = data2['Adj Close']

data = pd.merge(data1,data2, on='Date')
data.columns = [ticker1,ticker2]

#Calculate percentage returns for each stock
returns_monthly  = data.pct_change()
returns_annual = (1+returns_monthly.mean())**12-1

#Calculate covariance and variance for each stock
cov_monthly = returns_monthly.cov()
cov_annual = cov_monthly * 12

w = np.array([0,1])
print(w)
minVariance = 100000000
stdDev = 0
proportion = 0
expectedReturn = 0

returns = np.array(returns_annual)
print(returns)
covar = np.array(cov_annual)
print(covar)
for x in range(0,41):
	#calculate portfolio statistics for each increment of 2.5
	return_p = (w[0]+x*0.025) * returns[0] + (w[1]-x*0.025) * returns[1]
	var_p    = (w[0]+x*0.025)**2*covar[0,0] + (w[1]-x*0.025)**2*covar[1,1] + 2*(w[0]+x*0.025)*(w[1]-x*0.025)*covar[0,1]
	sd_p     = np.sqrt(var_p)
	
	print(return_p)
	print(var_p)
	print(sd_p)
	
	if minVariance > var_p:	
		minVariance = var_p
		stdDev = sd_p
		proportion = x
		expectedReturn = return_p

print("MVP Proportion", ticker1, ":", proportion*2.5, "%")
print("MVP Proportion", ticker2, ":", (100-proportion*2.5), "%")
print("MVP Standard Deviation:", stdDev*100, "%")
print("MVP Expected Portfolio Return:", expectedReturn*100, "%")
rf = 0.03
	

