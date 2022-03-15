import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pandas.plotting import scatter_matrix
from sklearn import datasets


# based on https://www.kaggle.com/mattcarter865/boston-house-prices to avoid timing out
boston = datasets.load_boston()
cols = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT','MEDV']
dataset = pd.DataFrame(data=np.c_[boston['data'], boston['target']], columns=cols)

scatter_matrix(dataset, figsize=(12,12))
plt.show()