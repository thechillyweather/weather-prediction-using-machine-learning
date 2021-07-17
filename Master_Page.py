#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import re
from Global_Prediction import G_Prediction
from Gases_Prediction_CO2_O3 import CO2_Prediction, O3_Prediction
from CountryWise_Prediction import C_Prediction
from flask import Flask, render_template, request
import pyrebase
from Config_Info import get_config, Password
import smtplib ,ssl
from email.message import EmailMessage


# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


# In[3]:


app=Flask(__name__)


# In[4]:



@app.route('/')
def root():
    return render_template('Home.html')

@app.route('/Home')
def Home():
    return render_template('Home.html')

@app.route('/Prediction_Form')
def Prediction_Form():
    return render_template('Prediction_Form.html')

@app.route('/Global_Prediction')
def Global_Prediction():
    return render_template('Global_Prediction.html')

@app.route('/Carbon_Prediction')
def Carbon_Prediction():
    return render_template('Carbon_Prediction.html')

@app.route('/OzonePrediction')
def OzonePrediction():
    return render_template('OzonePrediction.html')

@app.route('/SO2_Prediction')
def SO2_Prediction():
    return render_template('SO2_Prediction.html')

@app.route('/NO2_Prediction')
def NO2_Prediction():
    return render_template('NO2_Prediction.html')

@app.route('/CO_Prediction')
def CO_Prediction():
    return render_template('CO_Prediction.html')

@app.route('/co2no2dynamicgraph')
def co2no2dynamicgraph():
    return render_template('co2no2dynamicgraph.html')

@app.route('/ch4dynamicgraph')
def ch4dynamicgraph():
    return render_template('ch4dynamicgraph.html')

@app.route('/Global_yearly_average_tempreature')
def Global_yearly_average_tempreature():
    return render_template('Global_yearly_average_tempreature.html')

@app.route('/Measurement')
def Measurement():
    return render_template('Measurement.html')

@app.route('/Per_month')
def Per_month():
    return render_template('Per_month.html')

@app.route('/Solar')
def Solar():
    return render_template('Solar.html')

@app.route('/CO_Analysis')
def CO_Analysis():
    return render_template('CO_Analysis.html')

@app.route('/CO2_Analysis')
def CO2_Analysis():
    return render_template('CO2_Analysis.html')

@app.route('/NO2_Analysis')
def NO2_Analysis():
    return render_template('NO2_Analysis.html')

@app.route('/O3_Analysis')
def O3_Analysis():
    return render_template('O3_Analysis.html')

@app.route('/SO2_Analysis')
def SO2_Analysis():
    return render_template('SO2_Analysis.html')

@app.route('/UserManual')
def UserManual():
    return render_template('UserManual.html')

@app.route('/InputSources')
def InputSources():
    return render_template('InputSources.html')

@app.route('/Feedback')
def Feedback():
    return render_template('Feedback.html')

@app.route('/Random_Forest')
def Random_Forest():
    return render_template('Random_Forest.html')

@app.route('/Arima')
def Arima():
    return render_template('Arima.html')

@app.route('/Linear_Regression')
def Linear_Regression():
    return render_template('Linear_Regression.html')

@app.route('/CO_Detail_Analysis')
def CO_Detail_Analysis():
    return render_template('CO_Detail_Analysis.html')

@app.route('/CO2_Detail_Analysis')
def CO2_Detail_Analysis():
    return render_template('CO2_Detail_Analysis.html')

@app.route('/NO2_Detail_Analysis')
def NO2_Detail_Analysis():
    return render_template('NO2_Detail_Analysis.html')

@app.route('/O3_Detail_Analysis')
def O3_Detail_Analysis():
    return render_template('O3_Detail_Analysis.html')

@app.route('/SO2_Detail_Analysis')
def SO2_Detail_Analysis():
    return render_template('SO2_Detail_Analysis.html')

@app.route('/PCarbonPrediction')
def PCarbonPrediction():
    return render_template('Carbon_Prediction.html')
                           
@app.route('/PCOPrediction')
def PCOPrediction():
    return render_template('CO_Prediction.html')

@app.route('/PCountryPrediction')
def PCountryPrediction():
    return render_template('Prediction_Form.html')

@app.route('/PGlobalPrediction')
def PGlobalPrediction():
    return render_template('Global_Prediction.html')

@app.route('/PNO2Prediction')
def PNO2Prediction():
    return render_template('NO2_Prediction.html')

@app.route('/POzonePrediction')
def POzonePrediction():
    return render_template('OzonePrediction.html')

@app.route('/PSO2Prediction')
def PSO2Prediction():
    return render_template('SO2_Prediction.html')


# In[5]:


@app.route('/G_Predict',methods=['post'])
def G_Predict():
    year = request.form['year']
    try:
        if re.search("[0-9]", year):
            predicted_value = G_Prediction(year) 
            value = str(np.round(predicted_value, 2))
            prediction = value[1:-1]
            return render_template('Global_Prediction_Output.html', year=year, prediction=prediction)

        elif re.search("[@_!#$%^&*()<>?/\|}{~:]", year):
            msg_s="Input contains Special Character."
            return render_template('Global_Prediction.html', msg_s = msg_s)

        elif re.search("[A-Z]", year):
            msg_s="Input contains Characters."
            return render_template('Global_Prediction.html', msg_s = msg_s)

        elif re.search("[a-z]", year):
            msg_s="Input contains Characters."
            return render_template('Global_Prediction.html', msg_s = msg_s)

    except:
        msg="Prediction Can\'t be done."
        return render_template('Global_Prediction.html', msg = msg)

    return None



@app.route('/C_Predict',methods=['post'])
def C_Predict():
    name = request.form['country_name']
    date = request.form['date']

    try:
        if re.search("[0-9]", name):
            msg_s="Input contains numbers."
            return render_template('Prediction_Form.html', msg_s = msg_s)
        elif re.search("[@_!#$%^&*()<>?/\|}{~:]", name):
            msg_s="Input contains Special Character."
            return render_template('Prediction_Form.html', msg_s = msg_s)
        else:
            predicted_value = C_Prediction(name, date) 
            prediction = str(np.round(predicted_value, 2))
            return render_template('Country_Prediction_Output.html', country_name=name, prediction=prediction)

    except:
        msg="Prediction Can\'t be done."
        return render_template('Prediction_Form.html', msg = msg)
    return None

@app.route('/CO2Prediction',methods=['post'])
def CO2Prediction():
    year = request.form['year']
    try:
        if re.search("[0-9]", year):
            predicted_value = CO2_Prediction(year) 
            value = str(np.round(predicted_value, 2))
            prediction = value[1:-1]
            return render_template('Carbon_Prediction_Output.html', prediction=prediction)

        elif re.search("[@_!#$%^&*()<>?/\|}{~:]", year):
            msg_s="Input contains Special Character."
            return render_template('Carbon_Prediction.html', msg_s = msg_s)

        elif re.search("[A-Z]", year):
            msg_s="Input contains Characters."
            return render_template('Carbon_Prediction.html', msg_s = msg_s)

        elif re.search("[a-z]", year):
            msg_s="Input contains Characters."
            return render_template('Carbon_Prediction.html', msg_s = msg_s)


    except:
        msg="Prediction Can\'t be done."
        return render_template('Carbon_Prediction.html', msg = msg)

    return None

@app.route('/O3Prediction',methods=['post'])
def O3Prediction():
    year = request.form['year']
    try:
        if re.search("[0-9]", year):
            predicted_value = O3_Prediction(year) 
            value = str(np.round(predicted_value, 2))
            prediction = value[1:-1]
            return render_template('OzonePrediction_Output.html', prediction=prediction)

        elif re.search("[@_!#$%^&*()<>?/\|}{~:]", year):
            msg_s="Input contains Special Character."
            return render_template('OzonePrediction.html', msg_s = msg_s)

        elif re.search("[A-Z]", year):
            msg_s="Input contains Characters."
            return render_template('OzonePrediction.html', msg_s = msg_s)

        elif re.search("[a-z]", year):
            msg_s="Input contains Characters."
            return render_template('OzonePrediction.html', msg_s = msg_s)

    except:
        msg="Prediction Can\'t be done."
        return render_template('OzonePrediction.html', msg = msg)

    return None


# In[6]:


@app.route('/COPrediction',methods=['post'])
def COPrediction():
    year = request.form['year']
    try:
        if re.search("[0-9]", year):
            dataset = pd.read_csv('BackEnd_Code/DataSet/co.csv', parse_dates=['Date Local'])
            date = dataset["Date Local"]
            means = dataset.iloc[:, 2].values
            def wrangle(df):
                df["Date Local"]=pd.to_datetime(df["Date Local"])
                df["Month"]=df["Date Local"].dt.month
                df["Year"]=df["Date Local"].dt.year
                df=df.drop("Date Local",axis=1)
                df=df.drop("Month",axis=1)
                df=df[df.Year>=1850]
                df=df.set_index(['Year'])
                df=df.dropna()
                return df
            dataset = wrangle(dataset)
            dataset = dataset.drop(columns=['CO Units'],axis=1)
            year_temp=[]
            for i in range(len(date)):
                year_temp.append(date[i].year) 
            year_temp = np.array(year_temp)
            year_temp = year_temp.reshape(-1, 1)
            X_train,X_test,y_train,y_test= train_test_split(year_temp,means,test_size=0.1,random_state=10)
            poly_reg = PolynomialFeatures(degree = 2)
            year_temp_poly = poly_reg.fit_transform(year_temp)
            poly_reg.fit(year_temp_poly, means)
            lin_reg_2 = LinearRegression()
            lin_reg_2.fit(year_temp_poly, means)
            yeari = int(year)
            value = str(np.round(lin_reg_2.predict(poly_reg.fit_transform([[yeari]])),2))
            prediction = value[1:-1]
            return render_template('CO_Prediction_Output.html', prediction=prediction)

        elif re.search("[@_!#$%^&*()<>?/\|}{~:]", year):
            msg_s="Input contains Special Character."
            return render_template('CO_Prediction.html', msg_s = msg_s)

        elif re.search("[A-Z]", year):
            msg_s="Input contains Characters."
            return render_template('CO_Prediction.html', msg_s = msg_s)

        elif re.search("[a-z]", year):
            msg_s="Input contains Characters."
            return render_template('CO_Prediction.html', msg_s = msg_s)

    except:
        msg="Prediction Can\'t be done."
        return render_template('CO_Prediction.html', msg = msg)
    return None


# In[7]:


@app.route('/NO2Prediction',methods=['post'])
def NO2Prediction():
    year = request.form['year']
    try:
        if re.search("[0-9]", year):
            dataset = pd.read_csv('BackEnd_Code/DataSet/no2.csv', parse_dates=['Date Local'])
            date = dataset["Date Local"]
            means = dataset.iloc[:, 2].values
            def wrangle(df):
                df["Date Local"]=pd.to_datetime(df["Date Local"])
                df["Month"]=df["Date Local"].dt.month
                df["Year"]=df["Date Local"].dt.year
                df=df.drop("Date Local",axis=1)
                df=df.drop("Month",axis=1)
                df=df[df.Year>=1850]
                df=df.set_index(['Year'])
                df=df.dropna()
                return df
            dataset = wrangle(dataset)
            dataset = dataset.drop(columns=['NO2 Units'],axis=1)
            year_temp=[]
            for i in range(len(date)):
                year_temp.append(date[i].year) 
            year_temp = np.array(year_temp)
            year_temp = year_temp.reshape(-1, 1)
            X_train,X_test,y_train,y_test= train_test_split(year_temp,means,test_size=0.1,random_state=10)
            poly_reg = PolynomialFeatures(degree = 2)
            year_temp_poly = poly_reg.fit_transform(year_temp)
            poly_reg.fit(year_temp_poly, means)
            lin_reg_2 = LinearRegression()
            lin_reg_2.fit(year_temp_poly, means)
            yeari = int(year)
            value = str(np.round(lin_reg_2.predict(poly_reg.fit_transform([[yeari]])),2))
            prediction = value[1:-1]
            return render_template('NO2_Prediction_Output.html', prediction=prediction)

        elif re.search("[@_!#$%^&*()<>?/\|}{~:]", year):
            msg_s="Input contains Special Character."
            return render_template('NO2_Prediction.html', msg_s = msg_s)

        elif re.search("[A-Z]", year):
            msg_s="Input contains Characters."
            return render_template('NO2_Prediction.html', msg_s = msg_s)

        elif re.search("[a-z]", year):
            msg_s="Input contains Characters."
            return render_template('NO2_Prediction.html', msg_s = msg_s)

    except:
        msg="Prediction Can\'t be done."
        return render_template('NO2_Prediction.html', msg = msg)

    return None


# In[8]:


@app.route('/SO2Prediction',methods=['post'])
def SO2Prediction():
    year = request.form['year']
    try:
        if re.search("[0-9]", year):
            dataset = pd.read_csv('BackEnd_Code/DataSet/so2.csv', parse_dates=['Date Local'])
            date = dataset["Date Local"]
            means = dataset.iloc[:, 2].values
            def wrangle(df):
                df["Date Local"]=pd.to_datetime(df["Date Local"])
                df["Month"]=df["Date Local"].dt.month
                df["Year"]=df["Date Local"].dt.year
                df=df.drop("Date Local",axis=1)
                df=df.drop("Month",axis=1)
                df=df[df.Year>=1850]
                df=df.set_index(['Year'])
                df=df.dropna()
                return df
            dataset = wrangle(dataset)
            dataset = dataset.drop(columns=['SO2 Units'],axis=1)
            year_temp=[]
            for i in range(len(date)):
                year_temp.append(date[i].year) 
            year_temp = np.array(year_temp)
            year_temp = year_temp.reshape(-1, 1)
            X_train,X_test,y_train,y_test= train_test_split(year_temp,means,test_size=0.1,random_state=10)
            poly_reg = PolynomialFeatures(degree = 2)
            year_temp_poly = poly_reg.fit_transform(year_temp)
            poly_reg.fit(year_temp_poly, means)
            lin_reg_2 = LinearRegression()
            lin_reg_2.fit(year_temp_poly, means)
            yeari = int(year)
            value = str(np.round(lin_reg_2.predict(poly_reg.fit_transform([[yeari]])),2))
            prediction = value[1:-1]
            return render_template('SO2_Prediction_Output.html', prediction=prediction)

        elif re.search("[@_!#$%^&*()<>?/\|}{~:]", year):
            msg_s="Input contains Special Character."
            return render_template('SO2_Prediction.html', msg_s = msg_s)

        elif re.search("[A-Z]", year):
            msg_s="Input contains Characters."
            return render_template('SO2_Prediction.html', msg_s = msg_s)

        elif re.search("[a-z]", year):
            msg_s="Input contains Characters."
            return render_template('SO2_Prediction.html', msg_s = msg_s)

    except:
        msg="Prediction Can\'t be done."
        return render_template('SO2_Prediction.html', msg = msg)
    return None


# In[9]:


config = get_config()
firebase = pyrebase.initialize_app(config)
db = firebase.database()

@app.route('/contactdata',methods=['GET','POST'])
def contactdata():
    try:
        name = request.form['name']
        email = request.form['email']
        comment = request.form['comment']
        if re.search("[0-9]", name):
            msg_s="Input contains numbers."
            return render_template('Feedback.html', msg_s = msg_s)
        elif re.search("[@_!#$%^&*()<>?/\|}{~:]", name):
            msg_s="Input contains Special Character."
            return render_template('Feedback.html', msg_s = msg_s)
        else:
            data = dict(Name = name, Email = email, Comment = comment)
            db.child("MyProject").child("ContactUs_Data").push(data)
            msgc = "Thank you for giving us your feedback!"
            msg = EmailMessage()
            msg.set_content("Thank You for visiting!\nEvery single feedback is important for us.\nWe\'ll get back to you as soon as we can.\n\nYou can also contact us on\nemail - thechillyweather@gmail.com\nGithub - https://github.com/thechillyweather")
            sender = 'thechillyweather@gmail.com'
            password = Password()
            msg['Subject'] = "We\'re glad you visited us!"
            msg['From'] = sender
            msg['To'] = email
            context=ssl.create_default_context()
            with smtplib.SMTP("smtp.gmail.com",587) as smtp:
                smtp.starttls(context=context)
                smtp.login(sender, password)
                smtp.send_message(msg)
                smtp.quit()
            return render_template('Feedback.html', msg = msgc)
    except:
        msgs = 'Feedback Cannot be sent.'
        return render_template('Feedback.html', msg = msgs)
    return None
        


# In[10]:


if __name__ == '__main__':
    app.run()

