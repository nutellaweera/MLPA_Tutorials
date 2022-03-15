import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from yellowbrick.regressor import ResidualsPlot

def load_ext_boston():
    boston = datasets.load_boston()
    x = boston.data

    x = MinMaxScaler().fit_transform(boston.data)
    x = PolynomialFeatures(degree=1, include_bias=False).fit_transform(x)
    return x, boston.target

x,y = load_ext_boston()

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0)
lr = LinearRegression().fit(x_train, y_train)

ridge = Ridge(alpha=1.0).fit(x_train, y_train)

print(f'Training set score: {lr.score(x_train, y_train)}') #0.77
print(f'Test set score: {lr.score(x_test, y_test)}') #0.64
print(f'Ridge training set score: {ridge.score(x_train, y_train)}') #0.77
print(f'Ridge test set score: {ridge.score(x_test, y_test)}') #0.62

# vis_lr = ResidualsPlot(lr)
# vis_lr.fit(x_train, y_train)
# vis_lr.score(x_test, y_test)
# vis_lr.show()

vis_ridge = ResidualsPlot(ridge)
vis_ridge.fit(x_train, y_train)
vis_ridge.score(x_test, y_test)
vis_ridge.show()

