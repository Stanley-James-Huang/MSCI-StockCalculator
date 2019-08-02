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
data1 = pdr.get_data_yahoo(ticker1,start=datetime(2018, 6, 1), end=datetime(2019, 6, 1),interval='m')
data2 = pdr.get_data_yahoo(ticker2,start=datetime(2018, 6, 1), end=datetime(2019, 6, 1),interval='m')

data1 = data1['Adj Close']
data2 = data2['Adj Close']

data = pd.merge(data1,data2, on='Date')
data.columns = [ticker1,ticker2]

#Calculate percentage returns for each stock
returns_monthly  = data.pct_change()
returns_annual = ((1+returns_monthly.mean())**12)-1

#Calculate covariance and variance for each stock
cov_monthly = returns_monthly.cov()
cov_annual = cov_monthly * 12

w = np.array([0,1])

#minVariance portfolio
minVariance = 100000000
MVPstdDev = 0
MVPproportion = 0
MVPexpectedReturn = 0

#Market portfolio (max sharpe ratio)
rf = 0.02
max_Sharpe = -1
maxSharpe_stdDev = 0
maxSharpe_proportion = 0
maxSharpe_expectedReturn = 0

#Set things up for the loop
returns = np.array(returns_annual)
covar = np.array(cov_annual)

for x in range(0,41):
	#calculate portfolio statistics for each increment of 2.5
	return_p = (w[0]+x*0.025) * returns[0] + (w[1]-x*0.025) * returns[1]
	var_p    = (w[0]+x*0.025)**2*covar[0,0] + (w[1]-x*0.025)**2*covar[1,1] + 2*(w[0]+x*0.025)*(w[1]-x*0.025)*covar[0,1]
	sd_p     = np.sqrt(var_p)

	#Find min variance portfolio
	if minVariance > var_p:	
		minVariance = var_p
		MVPstdDev = sd_p
		MVPproportion = x
		MVPexpectedReturn = return_p
		
	#Find market portfolio
	sharpe_ratio = (return_p - rf)/sd_p
	if max_Sharpe < sharpe_ratio:  
		max_Sharpe = sharpe_ratio
		maxSharpe_stdDev = sd_p
		maxSharpe_proportion = x
		maxSharpe_expectedReturn = return_p
print("PART 1: Minimum Variance Portfolio\n")
print("MVP Proportion", ticker1, ":", MVPproportion*2.5, "%")
print("MVP Proportion", ticker2, ":", (100-MVPproportion*2.5), "%")
print("MVP Standard Deviation:", MVPstdDev*100, "%")
print("MVP Expected Portfolio Return:", MVPexpectedReturn*100, "%\n")

#PART 2 BEGINS HERE
print("PART 2: CML, Market Portfolio and Risk-free Assets\n")
print("\tCML Case 1: Market Portfolio\n")
print("Maximum Sharpe Ratio:", max_Sharpe)
print("Market Portfolio Proportion", ticker1, ":", maxSharpe_proportion*2.5, "%")
print("Market Portfolio Proportion", ticker2, ":", (100-maxSharpe_proportion*2.5), "%")
print("Market Portfolio Standard Deviation:", maxSharpe_stdDev*100, "%")
print("Market Portfolio Expected Portfolio Return:", maxSharpe_expectedReturn*100, "%\n")

print("\tCML Case 2: 50% risk-free asset, 50% Market Portfolio\n")
#covariance and variance are 0 for a risk-free asset
return_2 = (0.5) * rf + (0.5) * maxSharpe_expectedReturn
var_2 = (0.5)**2*maxSharpe_stdDev**2
sd_2 = np.sqrt(var_2)
print("Portfolio Expected Return:", return_2*100, "%")
print("Portfolio Standard Deviation:", sd_2*100, "%\n")

print("\tCML Case 3: -50% risk-free asset, 150% Market Portfolio\n")
return_3 = (-0.5) * rf + (1.5) * maxSharpe_expectedReturn
var_3 = (1.5)**2*maxSharpe_stdDev**2
sd_3 = np.sqrt(var_3)
print("Portfolio Expected Return:", return_3*100, "%")
print("Portfolio Standard Deviation:", sd_3*100, "%\n")


weights_0 = np.array(list(range(0,41)))/(100/2.5)
weights_1 = 1 - weights_0 
weights   = np.array([weights_0,weights_1]).T

port_returns = [w[0] * returns[0] + w[1] * returns[1] for w in weights]
port_vars    = [w[0]**2*covar[0,0] + w[1]**2*covar[1,1] + 2*w[0]*w[1]*covar[0,1] for w in weights]
port_sds     = [np.sqrt(v) for v in port_vars]

def calc_SR(w,mu,Sigma,rf):
    return_p = np.matmul(w,mu.T)
    var_p    = np.matmul(np.matmul(w,Sigma),w.T)
    sd_p     = np.sqrt(var_p)
    return((return_p - rf)/sd_p)
port_SRs = [calc_SR(w,returns,covar,rf) for w in weights]

df = pd.DataFrame([port_returns,port_sds, port_SRs]).transpose()
df.columns=['Returns', 'Volatility', 'Sharpe Ratio']
print(df)

plt.style.use('seaborn-pastel')
df.plot.scatter(x='Volatility', y='Returns', c='Sharpe Ratio',
                cmap='RdYlGn', edgecolors='black', figsize=(10, 8), grid=True)
plt.xlabel('Volatility (Std. Deviation)')
plt.ylabel('Expected Returns')
plt.title('Efficient Frontier')
plt.show()