import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer

def plot(train_scores, test_scores):
    plt.plot(range(1,11), train_scores, label="training accuracy")
    plt.plot(range(1,11), test_scores, label="testing accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("N_Neighbors")
    plt.legend()
    plt.show()

def load_from_txt():
    df = pd.read_csv('Week4/breast_cancer_data.txt')
    #print(df.head(5))

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

    data = np.array([4,3,3,2,1,2,1,1,2])
    prediction = clf.predict(data.reshape(1, -1))
    print(prediction)

    train_acc, test_acc = [],[]

    for i in range(1,11):
        clf = KNeighborsClassifier(i)
        clf.fit(x_train, y_train)
        train_acc.append(clf.score(x_train, y_train))
        test_acc.append(clf.score(x_test, y_test))

    plot(train_acc, test_acc)
    # Intepretation of plot; there are multiple high/low points (possible K), visually k=5 still seems like a good fit 


def load_from_sklearn():
    cancer_ds = load_breast_cancer()
    # random state - shuffling with reproducibility, stratify - class labels
    x_train, x_test, y_train, y_test = train_test_split(cancer_ds.data, cancer_ds.target, stratify=cancer_ds.target, random_state=66)
    
    train, test = [], []
    for i in range(1,12):
        clf = KNeighborsClassifier(n_neighbors=i)
        clf.fit(x_train, y_train)
        train.append(clf.score(x_train, y_train))
        test.append(clf.score(x_test, y_test))
    
    plot(train, test)


#load_from_txt()
#load_from_sklearn()

