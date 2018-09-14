# -*- coding: utf-8 -*-
"""
Created on Sun Jan 21 16:57:02 2018

@author: SKT
"""

import numpy as np
from numpy.linalg import inv
import pandas as pd
from sklearn import linear_model

class GradientDescent():
    theta = None
    m = None
    n = None
    
    def learn(self, X, y, lr):
        #finding h(x)
        hx = np.dot(X, self.theta) 
        #finding J
        J = np.dot(X.T,hx - y)/self.m    
        #computing new value of theta with a learning rate
        self.theta = self.theta - np.dot(lr, J)    
        return self.theta
    
    def fit(self, X, y, itr, lr):
        
        #no. of samples, no. of features
        self.m, self.n = X.shape
        
        ones = np.ones((self.m, 1))
        
        #append ones
        X = np.concatenate((ones, X), axis=1)
        
        #weights to set
        self.theta = np.zeros((self.n+1,1))
    
        for i in range(itr):
            self.theta = self.learn(X, y, lr)
    
    def predict(self, X):
        
        X = np.array(X) 
        m,n = X.shape
        ones = ones = np.ones((m, 1))
        X = np.concatenate((ones, X), axis=1)
        
        return np.dot(X, self.theta)

    def _fit(self, X, y):
        X = np.array(X) 
        m,n = X.shape
        ones = ones = np.ones((m, 1))
        X = np.concatenate((ones, X), axis=1)
        
        #using normal equation
        self.theta = np.dot(np.dot(inv(np.dot(X.T, X)),X.T), y)


###############training###############

data = pd.read_csv('house_data.csv')
data = data[:100]

features = data.filter(['bedrooms','bathrooms' ,'sqft_living', 'yr_built'])
features = features.values

target = data.filter(['price'])
target = target.values

features[:,3] = 2017 - features[:,3]

X_train = features[:-20]
y_train = target[:-20]

X_test = features[-20:]
y_test = target[-20:]


#############comparision##############

#my model
gd = GradientDescent()
gd._fit(X_train, y_train)

#sklearn model
regr = linear_model.LinearRegression()
regr.fit(X_train, y_train)


print('my prediction')
print(gd.predict(X_test))
print()

print('sklearn prediction')
print(regr.predict(X_test))
print()

print('my weights')
print(gd.theta)
print()

print('sklearn weights')
print(regr.intercept_, regr.coef_)



