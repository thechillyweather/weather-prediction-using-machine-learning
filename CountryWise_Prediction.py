#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from statsmodels.tsa.arima_model import ARIMA

def C_Prediction(C_Name, Date):
    
    df = pd.read_csv('BackEnd_Code/DataSet/GlobalLandTemperaturesByCountry.csv', delimiter=',')
    df_country = df.Country.unique()
    df_c = df.drop('AverageTemperatureUncertainty', axis=1)
    df_c = df_c[df_c.Country == C_Name]
    df_c = df_c.drop('Country',axis=1)
    df_c.index = pd.to_datetime(df_c.dt)
    df_c = df_c.drop('dt', axis=1)
    df_c = df_c.loc['1950-01-01':]
    df_c = df_c.sort_index()
    df_c.AverageTemperature.fillna(method='pad', inplace=True)
    df_c['Ticks'] = range(0,len(df_c.index.values))
    df_c['Roll_Mean'] = df_c.AverageTemperature.rolling(12).std()
    model = ARIMA(df_c.AverageTemperature, order=(1, 0, 2))  
    results_MA = model.fit()
    predictions = results_MA.predict('01/01/1950','04/01/2262')
    prediction = predictions[Date]
    
    return prediction

