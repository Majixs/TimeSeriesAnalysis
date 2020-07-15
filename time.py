import pandas as pd
from matplotlib import pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

goog = pd.read_csv('GOOGL.csv', index_col='Date', parse_dates=['Date'])
msft = pd.read_csv('MSFT.csv')
hum = pd.read_csv('humidity.csv', index_col='datetime', parse_dates=['datetime'])
pres = pd.read_csv('pressure.csv')

#goog['2006':'2008'].plot(subplots=True, figsize=(10,12))
#plt.title('Hello world')
#plt.savefig('hello_world.png')

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

decompose = seasonal_decompose(goog.Close, freq=360)
decompose.plot()
def test_stationarity(data):
    #rolling statistics
    rolmean = data.rolling(12).mean()
    rolstd = data.rolling(12).std()
    plt.figure(figsize = (17, 7))
    plt.plot(data, color='blue',label='Original')
    plt.plot(rolmean, color='red', label='Mean')
    plt.plot(rolstd, color='black', label = 'Std')
    plt.legend(loc='best')
    plt.title('Mean & Standard Deviation')
    plt.show()
    #Dickey-Fuller test:
    dftest = adfuller(data['y'], autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)
    

    


