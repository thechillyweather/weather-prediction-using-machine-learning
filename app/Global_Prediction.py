#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np 
import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression

def G_Prediction(value):
    globaltemp=pd.read_csv('BackEnd_Code/DataSet/GlobalTemperatures 2.csv',parse_dates=['dt','YEAR'])
    date=globaltemp["YEAR"]
    globaltemp=globaltemp.drop(columns=['YEAR','Unnamed: 9','Unnamed: 10','Unnamed: 11','Unnamed: 12','Unnamed: 14','Unnamed: 15','Unnamed: 16','Unnamed: 17'],axis=1)
    
    def wrangle(df):
        df["dt"]=pd.to_datetime(df["dt"])
        df["Month"]=df["dt"].dt.month
        df["Year"]=df["dt"].dt.year
        df=df.drop("dt",axis=1)
        df=df.drop("Month",axis=1)
        df=df[df.Year>=1850]
        df=df.set_index(['Year'])
        df=df.dropna()
        return df

    globaltemp=wrangle(globaltemp)
    target='LandAndOceanAverageTemperature'
    y=globaltemp[target]
    x=globaltemp[['LandAverageTemperature','LandMaxTemperature','LandMinTemperature']]
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=10)
    forestmodel=make_pipeline(
        SelectKBest(k="all"),
        StandardScaler(),
        RandomForestRegressor(
                                n_estimators=100,
                                max_depth=50,
                                random_state=77,
                                n_jobs=-1)
    )

    forestmodel.fit(x_train,y_train)
    year_temp=[]
    for i in range(len(date)):
        year_temp.append(date[i].year)
    year_temp = np.array(year_temp)
    year_temp = year_temp.reshape(-1,1)
    poly_reg = PolynomialFeatures(degree = 3)
    year_tempp=poly_reg.fit_transform(year_temp)
    poly_reg.fit(year_tempp,y)
    forestmodel.fit(year_tempp, y)
    year_tempp=poly_reg.fit_transform(year_temp)
    poly_reg.fit(year_tempp,y)
    lin_reg = LinearRegression()
    lin_reg.fit(year_tempp, y)
    value_int = int(value)
    if (value_int>=1850 and value_int<=2015):
        prediction=forestmodel.predict(poly_reg.fit_transform([[value_int]]))
    else:
        prediction=lin_reg.predict(poly_reg.fit_transform([[value_int]]))
    
    return prediction

