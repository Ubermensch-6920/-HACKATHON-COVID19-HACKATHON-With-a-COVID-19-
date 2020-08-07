
# coding: utf-8

# In[ ]:


import warnings
import os
import math
import pandas as pd
import numpy as np
from pandas import read_csv
from pandas import datetime
from pandas import DataFrame
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

#series = pd.read_csv(r'F:\Hackathon_EXL\DL_data_confirmed.csv',index_col=0)
#data=series
# evaluate parameters
#"F:\Hackathon_EXL\Data_divided\AG\KA.csv"
#Update Path 


def detrend(data):
    X = range(1,len(data)+1)
    X = np.reshape(X, (len(X), 1))
    y = data.values
    pf = PolynomialFeatures(degree=3)
    Xp = pf.fit_transform(X)
    md2 = LinearRegression()
    md2.fit(Xp, y)
    global trendp
    trendp = md2.predict(Xp)
    data = data - trendp 
    return data


def mape(actual, pred): 
    actual, pred = np.array(actual), np.array(pred)
    return np.mean(np.abs((actual - pred) / actual)) * 100

path = "F:\\Hackathon_EXL\\TBR\\"
l = os.listdir (path)
warnings.filterwarnings("ignore")
tally = list()
fin_pred = list()
i = 0
for files in l: 
    fold = path + files 
    data = pd.read_csv(fold,index_col=0)
    X = data
    X = X.drop(columns = ['Status']) 
#Used -90 and -30
    X = X.iloc[-30:,]
    X = np.log(X + 1)
 #   X = detrend(X)
    # prepare training dataset
    train_size = int(len(X) * 0.75)
    train, test = X[0:train_size], X[train_size:]
#    train = train.iloc[29:,]
    trainthis = train
    predictions = list()
    p_values = (0,1,2,3,4,5,7)
    d_values = (0,1,2)
    q_values = (0,1,2)
    best_score = 50000

    for p in p_values:
        for d in d_values:
            for q in q_values:
                arima_order = (p,d,q)
                predictions = list()
                history = train
                history = history.astype('float32')
                # make predictions
                try:
                    for t in range(len(test)):
                        model = ARIMA(history, order=arima_order)
                        model_fit = model.fit(disp=0)
                        output = model_fit.forecast()
                        yhat = output[0]
                        predictions.append(yhat)
                        obs = test[0:t+1]
                        history = history.append(obs)
                    error = math.sqrt(mean_squared_error(test, predictions))
                    if error < best_score:
                        best_score, best_cfg = error, arima_order
                        #me = mape(test,predictions)
  #                 print('ARIMA%s RMSE=%.3f' % (arima_order,error))
                except:
                    continue
    print('Best ARIMA%s RMSE=%.3f' % (best_cfg, best_score))
    
    tally.append(str(files) + "-" +str(best_cfg) + "-" + str(best_score))
        
 #   X.iloc[-1,] = X.iloc[-1,] + trendp[-1] 
    model = ARIMA(X, order = best_cfg)
    model_fit = model.fit(disp=0)
    forecast = model_fit.forecast(steps = 21)
    fin_pred.append( str(files)+ "-"  + str(np.exp(forecast[0])-1))
 #       past = trendp[-1]
        #data.iloc[welen(data) -1,1].astype('float32')
  #      i = i + 1
 #       for y in forecast:
 #           fin_pred.append(str(files) + "-" + str(int(np.exp(y))))
 #           past = past + y 
#        print(str(files) + "-" + str(fin_pred))
        # calculate out of sample error
    #RMSE
    #rmse = dict()
    #rmse[files] = math.sqrt(mean_squared    _error(test, predictions))
     


print(fin_pred)

