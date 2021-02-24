# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 18:19:51 2021

@author: user
"""
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score

st.title("Profit Prediction using different ML Algorithms")
data = pd.read_csv("50_Startups.csv")

df1 = data.drop(['State'],axis = 1)

# Splitting the dataset into Train and Test set
X = df1.drop(['Profit'],axis=1)
y = df1['Profit']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state = 123)

# Algorithm Selection
st.subheader('Select Regressor')
algo = st.selectbox('Select Regressor',('Linear Regressor','Random Forest Regressor','KNN Regressor'))
def get_regressor(algo):
    if algo =='Linear Regressor' :
        reg = LinearRegression()
    elif algo == 'Random Forest Regressor':
        reg = RandomForestRegressor()
    else:
        reg = KNeighborsRegressor()
        
    st.write(algo)
    return reg

reg = get_regressor(algo)
    
reg.fit(X_train,y_train)    
y_pred = reg.predict(X_test)

  
r2 = r2_score(y_test,y_pred)  
st.write(f'r2_score for {algo} is {r2}')

# Predicting on user values
st.header('Predicting Profit')

def profitPrediction(rd_spent,admn_spent,market_spent):
    values = np.array([[rd_spent,admn_spent,market_spent]]).astype(np.float64)
    prediction = reg.predict(values)
    return prediction

def main():
    rd_spent = st.text_input('R&D Spent',"")
    admn_spent = st.text_input('Administration',"")
    market_spent = st.text_input('Marketing Spend','')
    
    result = ''
    
    if st.button("Predict"):
        result = profitPrediction(rd_spent,admn_spent,market_spent)
    st.write('The Profit of the company is:')
    st.success(result)
    
    
if __name__ == '__main__':
    main()
    
# Exploratory Data Analysis
st.header("EDA")
st.write('Shape of the data',data.shape)
st.write('Columns of the data',data.columns)
st.write('Description of the data',data.describe())
st.write('Checking Null values of the data',data.isnull().sum())

# Plots

st.subheader('Line Chart')
st.line_chart(df1)

st.subheader('Correlation')
cor = data.corr()
st.write(cor)
plt.matshow(data.corr()) 
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    