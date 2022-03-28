import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score

diabetes = datasets.load_diabetes()
x = diabetes.data[:, np.newaxis, 2] # np.newaxis makes it a 2d array (?)
y = np.array(diabetes.target)

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42)

tree = DecisionTreeRegressor(random_state=0)
tree.fit(x_train, y_train)

print(f'Accuracy on training set: {round(tree.score(x_train, y_train), 3)}') # 0.617
print(f'Accuracy on test set: {round(tree.score(x_test, y_test), 3)}') # 0.125

y_pred = tree.predict(x_test)
print(f'Coefficient of determination r squared: {r2_score(y_test, y_pred)}')
# 0.125, compared to 0.317 for LR
# R^2 lower than LR, prolly not a better model