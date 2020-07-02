import lightgbm as lgb
import numpy as np
import pandas as pd
from matplotlib import pyplot

#from fbprophet import Prophet
from matplotlib import pyplot as plt
from pmdarima import auto_arima
from sklearn.metrics import mean_absolute_error, mean_squared_error
import datetime

data = pd.read_csv('data_for_regression.csv')
y = data.y
x = data.x

y_shift = y.shift()
y_stationary = y - y_shift
y_stationary = y_stationary.dropna()

dateFirst = datetime.datetime.today()
length = len(y_stationary)
dates = []
dates.append(dateFirst)
for i in range(length - 1):
    dateFirst += datetime.timedelta(days = 1)
    dates.append(dateFirst)

data = pd.DataFrame(y_stationary.values, index=dates, columns=['y'])

df_rolling3 = data['y'].rolling(window=3)
df_rolling7 = data['y'].rolling(window=7)
df_rolling30 = data['y'].rolling(window=30)

df_mean3 = df_rolling3.mean().shift().astype(np.float32)
df_mean7 = df_rolling7.mean().shift().astype(np.float32)
df_mean30 = df_rolling30.mean().shift().astype(np.float32)
df_std3 = df_rolling3.std().shift().astype(np.float32)
df_std7 = df_rolling7.std().shift().astype(np.float32)
df_std30 = df_rolling30.std().shift().astype(np.float32)

df_mean3.fillna(df_mean3.mean(), inplace=True)
df_mean7.fillna(df_mean7.mean(), inplace=True)
df_mean30.fillna(df_mean30.mean(), inplace=True)
df_std3.fillna(df_std3.mean(), inplace=True)
df_std7.fillna(df_std7.mean(), inplace=True)
df_std30.fillna(df_std30.mean(), inplace=True)

data['Date'] = data.index
data['month'] = data.Date.dt.month
data['week'] = data.Date.dt.week
data['day'] = data.Date.dt.day
data['df_mean3'] = df_mean3.values
data['df_mean7'] = df_mean7.values
data['df_mean30'] = df_mean30.values
data['df_std3'] = df_std3.values
data['df_std7'] = df_std7.values
data['df_std30'] = df_std30.values

data_train = data.iloc[:350, :]
data_test = data.iloc[350:, :]

features = ['df_mean3', 'df_mean7', 'df_mean30', 'df_std3', 'df_std7', 'df_std30', 'month', 'week', 'day']
dataFeatures =pd.DataFrame()
dataFeatures = data[features]
dataFeaturesTrain = dataFeatures.iloc[:350, :]
dataFeaturesTest = dataFeatures.iloc[350:, :]

model = auto_arima(data_train.y, trace=True, error_action='ignore', suppress_warnings=True)
model.fit(data_train.y)
forecast = model.predict(n_periods=len(data_test))
data_test['Forecast'] = forecast
true_pred = forecast + y_shift[351:]
pyplot.plot(true_pred)
pyplot.plot(y[351:])








































