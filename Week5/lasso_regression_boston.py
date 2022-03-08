import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge

boston_ds = datasets.load_boston()
x = boston_ds.data

x = MinMaxScaler().fit_transform(boston_ds.data)
x = PolynomialFeatures(degree=2, include_bias=False).fit_transform(x)

y = boston_ds.target

x_train, x_test, y_train, y_test = train_test_split(x,y,random_state=0)
lr = LinearRegression().fit(x_train, y_train)

ridge = Ridge(alpha=1.0).fit(x_train, y_train)
ridge10 = Ridge(alpha=10).fit(x_train, y_train)
ridge01 = Ridge(alpha=0.1).fit(x_train, y_train)

plt.plot(ridge.coef_, 's', label='Ridge alpha=1')
plt.plot(ridge10.coef_, '^', label='Ridge alpha=10')
plt.plot(ridge01.coef_, 'v', label='Ridge alpha=0.1')

plt.plot(lr.coef_, 'o', label='Linear regression')

plt.xlabel('Coefficient index')
plt.ylabel('Coefficient magnitude')
xlims = plt.xlim()
plt.hlines(0, xlims[0], xlims[1])
plt.xlim(xlims)
plt.ylim(-25,25)
plt.legend()
plt.show()