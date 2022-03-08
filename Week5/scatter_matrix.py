import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures

boston_ds = datasets.load_boston()
x = boston_ds.data
#print(type(x))

x = MinMaxScaler().fit_transform(boston_ds.data)
x = PolynomialFeatures(degree=2, include_bias=False).fit_transform(x)

x_pd = pd.DataFrame(x)
pd.plotting.scatter_matrix(x_pd)
plt.show()