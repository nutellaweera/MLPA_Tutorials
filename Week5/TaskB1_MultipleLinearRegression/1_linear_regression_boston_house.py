import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sqlalchemy import false

boston_ds = datasets.load_boston()
x = boston_ds.data

x = MinMaxScaler().fit_transform(boston_ds.data)
x = PolynomialFeatures(degree=2, include_bias=False).fit_transform(x)

y = boston_ds.target

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0)
lr = LinearRegression().fit(x_train, y_train)

print(f'Training set score: {lr.score(x_train, y_train)}') #0.9520519609032729
print(f'Test set score: {lr.score(x_test, y_test)}') #0.6074721959665736