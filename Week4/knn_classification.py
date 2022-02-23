import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

df = pd.read_csv('Week4/breast_cancer_data.txt')
print(df.head(5))

df.replace('?', -99999, inplace=True)
df.drop(['id'], 1, inplace=True)

x = np.array(df.drop(['class'], 1))
y = np.array(df['class'])

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0)

clf = KNeighborsClassifier(n_neighbors=1)
clf.fit(x_train, y_train)

print(clf.score(x_train, y_train))
print(clf.score(x_test, y_test))


for i in range(1,20):
    clf = KNeighborsClassifier(i)
    clf.fit(x_train, y_train)
    print('K=', i, ' Training=', clf.score(x_train, y_train), ' Testing=', clf.score(x_test, y_test))


# Test accuracy of training and testing for different Ks; most accurate when K=5
# Evaluate with id included; test and training scores are worse because the parameter isn't correlated
# Other parameters; default distance (metric) is Minkowski, changing to Euclidean or Manhattan affects K