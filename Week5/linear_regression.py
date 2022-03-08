import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

diabetes_ds = datasets.load_diabetes()
#print(type(diabetes_ds))
print(diabetes_ds.keys())

data = pd.DataFrame(diabetes_ds.data, columns=[diabetes_ds.feature_names])
target = pd.DataFrame(diabetes_ds.target)
x = diabetes_ds.data[:, np.newaxis, 2] # np.newaxis makes it a 2d array (?)
y = np.array(target)


x_train, x_test, y_train, y_test = train_test_split(x,y,random_state=42)
lr = LinearRegression().fit(x_train, y_train)


print(f'lr coefficient: {lr.coef_}')
print(f'lr intercept: {lr.intercept_}')

# make predictions using test set
y_pred = lr.predict(x_test)

print(f'Coefficient of determination r squared: {r2_score(y_test, y_pred)}')


plt.scatter(x_test, y_test, color='black')
plt.plot(x_test, y_pred, color='blue', linewidth=3)
plt.xticks()
plt.yticks()
plt.show()