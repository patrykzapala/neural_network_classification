# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import yfinance as yf
import pandas_datareader.data as web
import datetime
from datetime import date

from network import Network
from activationlayer import ActivationLayer
from fclayer import FCLayer
from activationfunctions import mse, d_mse, tanh, d_tanh, sigmoid, d_sigmoid, linear, d_linear

#Downloading company data
amazon = yf.Ticker('AMZN')
hist = amazon.history(period = 'max', interval = '1d')

#Downoading S&P500 index data
start = datetime.datetime(2010, 1, 1)
end = date.today()

SP500 = web.DataReader(['sp500'], 'fred', start, end)

#Setting the value of trend, 1 if share price go up, 0 otherwise
trend = np.empty(hist.shape[0])
trend.fill(1)
for i in range(hist.shape[0]-1):
    if(hist.iloc[i+1]['Close'] < hist.iloc[i]['Close']):
        trend[i] = 0
hist['PMov'] = trend

hist = hist[hist.index > "2018"]
SP500 = SP500[SP500.index > "2018"]
hist = hist[hist.index < "2021-03-28"]
SP500 = SP500[SP500.index < "2021-03-28"]

#Splitting data based on a freq argument
hist['sp500'] = SP500
df = hist.groupby(pd.Grouper(freq='W', label='right'))

#Array for the target values
y = np.empty(len(df))
#Array for average Close price
avClose = np.empty(len(df))
#Array for average Open price
avOpen = np.empty(len(df))
#Array for average Volume
avVolume = np.empty(len(df))
#Array for average trend
avTrend = np.empty(len(df))
#Array for standard deviation of Close proce
stdClose = np.empty(len(df))
#Array for standard deviation of Volume
stdVolume = np.empty(len(df))
#Array for average SP500 close price
avSP500 = np.empty(len(df))
#Array for standard deviation of SP500 proce
stdSP500 = np.empty(len(df))
y.fill(0)
i = 0
#Counting input values
for timestamp in df:
    values = timestamp[1].to_numpy()
    avClose[i] = np.mean(values[:-1,3])
    avOpen[i] = np.mean(values[:-1,0])
    avVolume[i] = np.mean(values[:-1,4])
    avTrend[i] = np.mean(values[:-1,7])
    stdClose[i] = np.std(values[:-1,3])
    stdVolume[i] = np.std(values[:-1,4])
    avSP500[i] = np.mean(values[:-1,8])
    stdSP500[i] = np.std(values[:-1,8])
    #Counting target value
    if(values[-1,3] > values[-2,3]):
        y[i] = 1
    i += 1

#Normalization    
avClose = (avClose - np.min(avClose))/ (np.max(avClose) - np.min(avClose))
avOpen = (avOpen - np.min(avOpen))/ (np.max(avOpen) - np.min(avOpen))
avVolume = (avVolume - np.min(avVolume))/ (np.max(avVolume) - np.min(avVolume))
stdClose = (stdClose - np.min(stdClose))/ (np.max(stdClose) - np.min(stdClose))
stdVolume = (stdVolume - np.min(stdVolume))/ (np.max(stdVolume) - np.min(stdVolume))
avSP500 = (avSP500 - np.min(avSP500))/ (np.max(avSP500) - np.min(avSP500))
stdSP500 = (stdSP500 - np.min(stdSP500))/ (np.max(stdSP500) - np.min(stdSP500))


#Making and fitting data
length = len(df)
rData = np.array([avClose, avOpen, avVolume, stdClose,  avTrend, stdVolume,
                  avSP500, stdSP500, y])
rData = np.rot90(rData, 3)
rData = rData.reshape((length,1,9))

#Setting data in random order
np.random.shuffle(rData)

#Splitting for train and test data
rDataTrain = rData[:int(0.8*len(rData))]
rDataTest = rData[int(0.8*len(rData)):]


#Making a network
network = Network()
#Adding layers
network.add(FCLayer(8, 4))
network.add(ActivationLayer(tanh, d_tanh))
network.add(FCLayer(4,2))
network.add(ActivationLayer(sigmoid, d_sigmoid))
network.add(FCLayer(2,1))
network.add(ActivationLayer(sigmoid, d_sigmoid))

#Setting loss functions
network.use(mse, d_mse)

#Training data
network.fit(rDataTrain[:,:,1:],rDataTrain[:,:,0], 3500, 0.3)

#Predicting output
out = network.predict(rDataTrain[:,:,1:])
res = np.empty(len(out))
for i in range(len(out)):
    if(out[i] > 0.5):
        res[i] = 1
    else:
        res[i] = 0
    
print("Accuracy of train data:")
print(np.mean(res==rDataTrain[:,:,0]))

out2 = network.predict(rDataTest[:,:,1:])
res2 = np.empty(len(out2))
for i in range(len(out2)):
    if(out2[i] > 0.5):
        res2[i] = 1
    else:
        res2[i] = 0
        
print("Accuracy of test data:")
print(np.mean(res2==rDataTest[:,:,0]))