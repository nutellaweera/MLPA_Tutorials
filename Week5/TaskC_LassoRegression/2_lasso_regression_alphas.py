import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso

def load_ext_boston():
    boston = datasets.load_boston()
    x = boston.data
    x = MinMaxScaler().fit_transform(boston.data)
    x = PolynomialFeatures(degree=2, include_bias=False).fit_transform(x)
    return x, boston.target

x, y = load_ext_boston()

x_train, x_test, y_train, y_test = train_test_split(x,y, random_state = 0)
lr = LinearRegression().fit(x_train, y_train)

l1 = Lasso(alpha=1).fit(x_train, y_train)
l10 = Lasso(alpha=10).fit(x_train, y_train)
l01 = Lasso(alpha=0.1).fit(x_train, y_train)

plt.plot(l1.coef_, 's', label="Lasso alpha=1")
plt.plot(l10.coef_, '^', label="Lasso alpha=0.01")
plt.plot(l01.coef_, 'v', label="Lasso alpha=0.0001")

plt.plot(lr.coef_, 'o', label="Linear Regression")
plt.xlabel("Coefficient index")
plt.ylabel("Coefficient magnitude")
xlims = plt.xlim()

plt.hlines(0, xlims[0], xlims[1])
plt.xlim(xlims)
plt.ylim(-25,25)
plt.legend()
plt.show()
