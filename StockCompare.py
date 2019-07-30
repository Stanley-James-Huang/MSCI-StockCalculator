import pandas as pd
import numpy as np
from datetime import datetime
import pandas_datareader as pdr
import matplotlib.pyplot as plt
#include versions of the headers

#Prompt user
ticker1 = input("Please enter ticker 1: ")
ticker2 = input("Please enter ticker 2: ")

marketData = pdr.get_data_yahoo('SPY',start=datetime(2018, 6, 1), end=datetime(2019, 6, 1),interval='m')
data1 = pdr.get_data_yahoo(ticker1,start=datetime(2018, 6, 1), end=datetime(2019, 6, 1),interval='m')
data2 = pdr.get_data_yahoo(ticker2,start=datetime(2018, 6, 1), end=datetime(2019, 6, 1),interval='m')

data1 = data1['Adj Close']
data2 = data2['Adj Close']

returns_monthly1  = data1.pct_change()
returns_annual1 = (1+returns_monthly1.mean())**12-1

returns_monthly2  = data2.pct_change()
returns_annual2 = (1+returns_monthly2.mean())**12-1

#adjust weights of portfolio
w = np.array([0,1])
for x in range(0,10):
	#Calculate covariance
	cov_monthly1 = returns_monthly1.cov()
	cov_annual1 = cov_monthly1 * 12
	cov_monthly2 = returns_monthly2.cov()
	cov_annual2 = cov_monthly2 * 12

	#calculate portfolio statistics
	return_p = w[0] * returns[0] + w[1] * returns[1]
	var_p    = w[0]**2*covar[0,0] + w[1]**2*covar[1,1] + 2*w[0]*w[1]*covar[0,1]
	sd_p     = np.sqrt(var_p)
	rf       = 0.03 #

