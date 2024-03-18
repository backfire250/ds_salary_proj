# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 10:39:07 2023

@author: eredfield
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('eda_data.csv')

## choose relevant columns
df.columns

df_model = df[['avg_salary', 'Rating', 'Size', 'Type of ownership', 'Industry', 'Sector', 'Revenue', 'num_comp', 'hourly', 'employer_provided',
             'job_state', 'same_state', 'age', 'python_yn', 'spark', 'aws', 'excel', 'job_simp', 'seniority', 'desc_len']]

## get dummy data
df_dum = pd.get_dummies(df_model)

## train test splits
from sklearn.model_selection import train_test_split

X = df_dum.drop('avg_salary', axis = 1)
y = df_dum.avg_salary.values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

## multiple linear regression
import statsmodels.api as sm

X_sm = X = sm.add_constant(X)
model = sm.OLS(y, X_sm)
model.fit().summary()

from sklearn.linear_model import LinearRegression, Lasso
from sklearn.model_selection import cross_val_score

lm = LinearRegression()
lm.fit(X_train, y_train)

# finding the error value using multiple linear regression
np.mean(cross_val_score(lm, X_train, y_train, scoring= 'neg_mean_absolute_error', cv=3))
# this gives us an error value of -20.766, meaning that using this model, our estimates can be up to $20k off when predicting salaries

## lasso regression
lm_l = Lasso(alpha=.13)
lm_l.fit(X_train,y_train)
np.mean(cross_val_score(lm_l, X_train, y_train, scoring= 'neg_mean_absolute_error', cv=3))

alpha = []
error = []

for i in range(1, 100):
    alpha.append(i/100)
    lml = Lasso(alpha=(i/100))
    error.append(np.mean(cross_val_score(lml, X_train, y_train, scoring= 'neg_mean_absolute_error', cv=3)))
    
plt.plot(alpha, error)

# finding the alpha value that gives us the best error term
err = tuple(zip(alpha, error))
df_err = pd.DataFrame(err, columns = ['alpha', 'error'])
df_err[df_err.error == max(df_err.error)]
# an alpha value of 0.13 gives our lowest error rate of -19.257798. this means that using lasso regression gives us a
# slightly smaller error than using multiple linear regression

## random forest
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor()
np.mean(cross_val_score(rf, X_train, y_train, scoring = 'neg_mean_absolute_error', cv=3))
# this gives us an error value of -15.073, meaning we've improved our error rate by about $4k by using a random forest model

## tune models using GridsearchCV
# with Gridsearch, you put in all the parameters you want and it spits out the model that produces the best results
from sklearn.model_selection import GridSearchCV

parameters = {'n_estimators':range(10, 300, 10), 'criterion': ('squared_error', 'absolute_error'), 'max_features':('sqrt', 'log2')}

gs = GridSearchCV(rf, parameters, scoring = 'neg_mean_absolute_error', cv=3)
gs.fit(X_train, y_train)

gs.best_score_
# the best error score we can achieve is -15.153 which is about the same as our random forest model
gs.best_estimator_
# this tell us the parameters of the best model found: # of estimators at 200, max_features was sqrt

## test all your different ensembles
tpred_lm = lm.predict(X_test)
tpred_lml = lm_l.predict(X_test)
tpred_rf = gs.best_estimator_.predict(X_test)

from sklearn.metrics import mean_absolute_error
mean_absolute_error(y_test, tpred_lm)
mean_absolute_error(y_test, tpred_lml)
mean_absolute_error(y_test, tpred_rf)
# our random forest model gives the best error on our testing data with an error score of 12.12

# combining our multiple regression and random forest models to see if we can get a better number
mean_absolute_error(y_test,(tpred_lm+tpred_rf)/2)
# gives us an error of 14.41

# In the end, our random forest model performed the best and is able to predict salaries within $12k of their actual value

## pickling the model allows it to be used by other programs without having to re-create the model
import pickle
pickl = {'model': gs.best_estimator_}
pickle.dump( pickl, open( 'model_file' + ".p", "wb" ))

file_name = "model_file.p"
with open(file_name, 'rb') as pickled:
          data = pickle.load(pickled)
          model = data['model']

model.predict(X_test.iloc[1, :].values.reshape(1, -1))