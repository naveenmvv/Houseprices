# -*- coding: utf-8 -*-
"""
Created on Wed May  8 20:59:06 2019

@author: NAVEEN
"""

import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np
import xgboost
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import GridSearchCV


def rmsle(y_pred, y_test) : 
    assert len(y_test) == len(y_pred)
    return np.sqrt(np.mean((np.log(1+y_pred) - np.log(1+y_test))**2))


df = pd.read_csv("F:/house-prices-advanced-regression-techniques/train.csv")
dtest = pd.read_csv("F:/house-prices-advanced-regression-techniques/test.csv")
corr = df.corr()

#ax = sns.heatmap(corr)

li = []
for i in range(0,len(df.columns)):
    if  (str(df.iloc[0][i])).isalpha():
            li.append(df.iloc[:,i].tolist())

dictlist=[]

for l in li:
    mydict={}
    i = 0
    for item in l:
        if(i>0 and item in mydict):
            continue
        else:    
            i = i+1
            mydict[item] = i
    dictlist.append(mydict)        

nlist=[]

for (a, d) in zip(li, dictlist): 
     k =[]
     for item in a:
         k.append(d[item])
     nlist.append(k)

sale = df['SalePrice']    
nlist.append(sale.tolist()) 
dn = (pd.DataFrame(nlist)).transpose()    
corrn = dn.corr() 

row = corr['SalePrice']

rlist = []
for i in range(0,len(row)):
    if row[i] > 0.6:
            rlist.append(row.index[i] )

dt = df[rlist]
X = df[['OverallQual','GrLivArea','TotalBsmtSF','1stFlrSF','GarageCars','GarageArea']]
Y = df['SalePrice']
X_train, X_val, y_train, y_val = train_test_split(
    X, Y , train_size = 0.8 ,random_state=10)


#reg = LinearRegression().fit(X, Y)
#print(reg.score(X, Y))
#y_pred = reg.predict(X_val)
#rms =  rmsle(y_val, y_pred)
#print(rms)
# A parameter grid for XGBoost
parameters = {'nthread':[4], #when use hyperthread, xgboost may become slower
              'objective':['reg:linear'],
              'learning_rate': [.03, 0.05, .07, 0.08], #so called `eta` value
              'max_depth': [5, 6, 7,8,9],
              'min_child_weight': [4,5,6],
              'subsample': [0.7,0.9,1],
              'colsample_bytree': [0.7, 1],
              'n_estimators': [85],
              'gamma':[1,5,10]}


xgb = xgboost.XGBRegressor(booster = 'gbtree' ,n_estimators=85, learning_rate=0.07, gamma=1, subsample=0.8, random_state=10,
                           colsample_bytree =1, max_depth=5, min_child_weight=3)
#xgb = xgboost.XGBRegressor()
#xgb_grid = GridSearchCV(xgb,
#                        parameters,
#                        cv = 2,
#                        n_jobs = 5,
#                        verbose=True)
#
#xgb_grid.fit(X_train,
#         y_train)
#
#print(xgb_grid.best_score_)
#print(xgb_grid.best_params_)

xgb = xgb.fit(X_train,y_train)
y_pred = xgb.predict(X_val)
rms =  rmsle(y_val, y_pred)
print("XGB here")
print(xgb.score(X, Y))
print(rms)

test = dtest[['OverallQual','GrLivArea','TotalBsmtSF','1stFlrSF','GarageCars','GarageArea']]
Result = xgb.predict(test)

