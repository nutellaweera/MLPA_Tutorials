import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge

from yellowbrick.datasets import load_concrete
from yellowbrick.regressor import ResidualsPlot

boston_ds = datasets.load_boston()
x = boston_ds.data

x = MinMaxScaler().fit_transform(boston_ds.data)
x = PolynomialFeatures(degree=2, include_bias=False).fit_transform(x)

y = boston_ds.target

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0)
lr = LinearRegression().fit(x_train, y_train)

ridge = Ridge(alpha=1.0).fit(x_train, y_train)

print(f'Training set score: {lr.score(x_train, y_train)}') #.9520519609032729
print(f'Test set score: {lr.score(x_test, y_test)}') #0.6074721959665736
print (f'Ridge training test score: {ridge.score(x_train, y_train)}') #0.8857966585170938
print(f'Ridge test set score: {ridge.score(x_test,y_test)}') #0.7527683481744756

visualizer = ResidualsPlot(ridge)
visualizer.score(x_test, y_test)
visualizer.show()