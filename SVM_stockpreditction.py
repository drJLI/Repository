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
time_horizon = 1
dates = pd.bdate_range('2010-01-01','2017-11-15')
df_TS = pd.DataFrame(index=dates).sort_index(axis = 0,ascending = False)
df_ts = pd.read_csv(os.path.join(pwd,'AAPL.csv'), parse_dates=True,na_values=['nan'])
df_ts = df_ts.rename(columns = {'Adj Close':'Price',})


#df_ts['Date'] = df_ts['Date'].str.slice(0,10)
df_ts = df_ts.set_index(['Date'])
df_ts = df_ts.drop_duplicates()
df = df_TS.join(df_ts, how = 'left').dropna(axis = 0, how = 'all').bfill().sort_index(ascending = True)

df['SMA_10'] = df['Price'].rolling(10).mean()
df['SMA_3'] = df['Price'].rolling(3).mean()
df['SMA_50'] = df['Price'].rolling(50).mean()

df['EWMA_D_10'] = pd.DataFrame(df['Price']).ewm(span = 10,adjust = False).mean()

df['ROC10'] = (df['Price']/ df['Price'].shift(10)-1)*100
df['ROC5'] = (df['Price']/ df['Price'].shift(5)-1)*100
df['ROC3'] = (df['Price']/ df['Price'].shift(3)-1)*100
df['WilliamR_14'] = (df['Price'].rolling(14).max()-df['Price'])/(df['Price'].rolling(14).max()-df['Price'].rolling(14).min())*100

df['MOM3'] = df['Price']- df['Price'].shift(3)
df['MOM1'] = df['Price']- df['Price'].shift()
# Calculate the Relative Strength Index based on EWMA        
delta = df['Price'].diff()

up,down = delta.copy(),delta.copy()
up[up<0] = 0
down[down >= 0] = 0
down = abs(down)
df['RS_up'] = up
df['RS_down'] = down
avg_up = up.ewm(span=14,adjust = False).mean()
avg_down = down.ewm(span=14,adjust = False).mean()

RS = avg_up / avg_down
df['RSI'] = 100.0 - (100.0 / (1.0 + RS))

#Calculate Money Flow Index
df['TP'] = (df['High']+df['Low']+df['Close'])/3

delta_tp = df['TP'].diff()
delta_tp[delta_tp<0] = -1
delta_tp[delta_tp>0] = 1
delta_tp[delta_tp==0] = 0

df['TP_delta'] = delta_tp

MF = df['TP']*df['Volume']*delta_tp
MF_up,MF_down = MF.copy(),MF.copy()
MF_up[MF_up<0] = 0
MF_down[MF_down >= 0] = 0
MF_down = abs(MF_down)
MFR = MF_up.rolling(14).mean()/MF_down.rolling(14).mean()
df['MFI_14'] = 100-100/(1+MFR)

#Calculate Average True Range
TR = pd.DataFrame({'a':abs(df['High']-df['Low']),'b':abs(df['High']-df['Close'].shift()),'c':abs(df['Low']-df['Close'].shift())})
df['ATR_14'] = TR.max(1).ewm(span=14,adjust = False).mean()

df['OBV'] = 0

delta_obv = delta.apply(lambda x : -1 if x<0 else(1 if x>0 else 0))
df.at[df.index[0],'OBV'] = df['Volume'][0]
for i in range(1,len(delta_obv)):
    if delta_obv[i] ==0:
        df.at[df.index[i],'OBV'] = df.at[df.index[i-1],'OBV']
    elif delta_obv[i] > 0:
        df.at[df.index[i],'OBV'] = df.at[df.index[i-1],'OBV']+ df.at[df.index[i],'Volume']
    else:
        df.at[df.index[i],'OBV'] = df.at[df.index[i-1],'OBV']- df.at[df.index[i],'Volume']

if time_horizon ==1:
    label1 =  df['Price'] -df['Price'].shift()
    label = df['Price'] - df['Price'].shift(-1)
else:
    label = df['SMA_'+str(time_horizon)].shift(-1) - df['SMA_'+str(time_horizon)]
    label1 = df['SMA_'+str(time_horizon)].shift() - df['SMA_'+str(time_horizon)]
#label = label.dropna()
df['Label'] = label.apply(lambda x : 1 if x>0 else 0)
df['Label1'] = label1.apply(lambda x : 1 if x>0 else 0)
df['Label2'] = 0


for i in range(0,len(df.index)-1):
    
    if time_horizon ==1:
        if df.at[df.index[i],'Price']<df.at[df.index[i+1],'Price']:
            df.at[df.index[i],'Label2'] = 1
        else:
            df.at[df.index[i],'Label2'] = 0
    else:
        if df.at[df.index[i],'SMA_'+str(time_horizon)]<df.at[df.index[i+1],'SMA_'+str(time_horizon)]:
            df.at[df.index[i],'Label2'] = 1
        else:
            df.at[df.index[i],'Label2'] = 0


df_svm = df.dropna(axis = 0, how = 'any').drop(['Price','Open','High','Low','Close','Volume'],axis = 1)



df_svm = pd.DataFrame(min_max.fit_transform(df_svm),index = df_svm.index, columns = df_svm.columns)
num_feature = len(df.columns)-1
#Prepare datasets

length = df_svm.shape[0]

train = df_svm.iloc[0:int(length*0.7),:]
CV = df_svm.iloc[int(length*0.7):int(length*0.8),:]
test = df_svm.iloc[int(length*0.8):,:]






X_train = train.drop(['Label','Label1','Label2'],axis = 1).as_matrix()
y_train = np.array(train['Label2'].values.tolist())


X_cv = CV.drop(['Label','Label1','Label2'],axis = 1).as_matrix()
y_cv = np.array(CV['Label2'].values.tolist())


X_test = test.drop(['Label','Label1','Label2'],axis = 1).as_matrix()
y_test = np.array(test['Label2'].values.tolist())

#%%
svr_lin = LinearSVR(C = 100)
svr_rbf = SVR(kernel = 'rbf',C = 1 , gamma =1)

svr_lin.fit(X_train,y_train)
svr_rbf.fit(X_train,y_train)

y_predict = svr_rbf.predict(X_train)


print('Accuracy on training set: {:.2f}'.format(svr_rbf.score(X_train, y_train)))
print('Accuracy on CV set: {:.2f}'.format(svr_rbf.score(X_cv, y_cv)))
print('Accuracy on test set: {:.2f}'.format(svr_rbf.score(X_test, y_test)))

'''
print('Accuracy on training set (lin): {:.2f}'.format(svr_lin.score(X_train, y_train)))
print('Accuracy on CV set: {:.2f}'.format(svr_lin.score(X_cv, y_cv)))
print('Accuracy on test set: {:.2f}'.format(svr_lin.score(X_test, y_test)))

'''