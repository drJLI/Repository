# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 16:23:42 2017

@author: jiaweili
"""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.svm import LinearSVR
from sklearn.preprocessing import MinMaxScaler
#from sklearn.model_selection import train_test_split
import os

min_max = MinMaxScaler()

pwd = os.getcwd()
dates = pd.bdate_range('2010-11-11','2017-11-15')
#df_TS = pd.DataFrame(index=dates).sort_index(axis = 0,ascending = False)
df_ts = pd.read_csv(os.path.join(pwd,'data.csv'), parse_dates=True,na_values=['nan'])



df_ts['Date'] = df_ts['Date'].str.slice(0,10)
df_ts = df_ts.set_index(['Date'])
df_ts = df_ts.drop_duplicates()



df_ts['ROC10'] = df_ts['Price Close']/ df_ts['Price Close'].shift(10)-1
df_ts['ROC5'] = df_ts['Price Close']/ df_ts['Price Close'].shift(5)-1
df_ts['ROC2'] = df_ts['Price Close']/ df_ts['Price Close'].shift(2)-1
df_ts['Willian_R'] = (df_ts['Price Close'].rolling(10).max()-df_ts['Price Close'])/(df_ts['Price Close'].rolling(14).max()-df_ts['Price Close'].rolling(14).min())*100

df_ts['SPX Momentum'] = df_ts['SPX'] - df_ts['SPX'].shift()
df_ts['SPX Momentum'] = df_ts['SPX Momentum'].apply(lambda x :1 if x >=0 else 0)

df_ts['SPLRCTHSP Momentum'] = df_ts['SPLRCTHSP'] - df_ts['SPLRCTHSP'].shift()
df_ts['SPLRCTHSP Momentum'] = df_ts['SPLRCTHSP Momentum'].apply(lambda x :1 if x >=0 else 0)

df_ts['Price Momentum'] = df_ts['Price Close'] - df_ts['Price Close'].shift()
df_ts['Price Momentum'] = df_ts['Price Momentum'].apply(lambda x :1 if x >=0 else 0)

        
df_ts['EWMA_10D']=df_ts['Price Close'].ewm(span=10,adjust = False).mean()

df_ts['Volatility_10D'] = df_ts['Price Close'].rolling(10).std()

df_ts = df_ts.drop(['Price Close','Instrument','Currency','Beta','WACC Cost of Equity, (%)','1 Week Total Return','Willian_R'],axis = 1)
df_ts = df_ts.dropna(axis = 0, how = 'any')

df = pd.DataFrame(min_max.fit_transform(df_ts),index = df_ts.index, columns = df_ts.columns)
num_feature = len(df.columns)-1
#Prepare datasets

length = df.shape[0]

train = df.iloc[0:int(length*0.7),:]
CV = df.iloc[int(length*0.7):int(length*0.8),:]
test = df.iloc[int(length*0.8):,:]

'''
delta = df_ts['Price Close'].diff()

up,down = delta.copy(),delta.copy()
up[up<0] = 0
down[down > 0] = 0
down = -down

RSI = up.mean()/(up.mean()+down.mean())*100
# Calculate the EWMA
roll_up1 = up.ewm(span=window_14,adjust = False).mean()
roll_down1 = down.ewm(span=window_14,adjust = False).mean()

# Calculate the RSI based on EWMA
RS1 = roll_up1 / roll_down1
RSI1 = 100.0 - (100.0 / (1.0 + RS1))

'''


X_train = train.drop('Price Momentum',axis = 1).as_matrix()
y_train = np.array(train['Price Momentum'].values.tolist())


X_cv = CV.drop('Price Momentum',axis = 1).as_matrix()
y_cv = np.array(CV['Price Momentum'].values.tolist())


X_test = test.drop('Price Momentum',axis = 1).as_matrix()
y_test = np.array(test['Price Momentum'].values.tolist())

#%%
#svr_lin = LinearSVR()
svr_rbf = SVR(kernel = 'rbf',C = 0.05, gamma =0.1)

#svr_lin.fit(X_train,y_train)
svr_rbf.fit(X_train,y_train)

print('Accuracy on training set: {:.2f}'.format(svr_rbf.score(X_train, y_train)))
print('Accuracy on CV set: {:.2f}'.format(svr_rbf.score(X_cv, y_cv)))
print('Accuracy on test set: {:.2f}'.format(svr_rbf.score(X_test, y_test)))

'''
print('Accuracy on training set: {:.2f}'.format(svr_lin.score(X_train, y_train)))
print('Accuracy on CV set: {:.2f}'.format(svr_lin.score(X_cv, y_cv)))

plt.scatter(X_test,y_test,color = 'black', label = 'Data')
plt.plot(dates,svr_rbf.predict(dates),color = 'red', label = 'RBF Model')
#plt.plot(dates,svr_lin.predict(dates),color = 'green', label = 'Linear Model')

#plt.plot(X_test,svr_rbf.predict(X_test),color = 'red', label = 'RBF Model')
#plt.plot(X_test,svr_lin.predict(X_test),color = 'green', label = 'Linear Model')
#plt.plot(X_test,svr_poly.predict(X_test),color = 'blue', label = 'Polynomial Model')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Support Vector Regression')
plt.legend()
plt.show()
'''
