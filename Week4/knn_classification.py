import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import f1_score, plot_confusion_matrix

def gen_plot(train_scores, test_scores):
    plt.plot(range(1,11), train_scores, label="training accuracy")
    plt.plot(range(1,11), test_scores, label="testing accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("N_Neighbors")
    plt.legend()
    plt.show()

def gen_confusion_matrix(clf, x_test, y_test):
    plot_confusion_matrix(clf, x_test, y_test, cmap=plt.cm.Blues)
    plt.show()

# f1 score = 2 * (precision * recall) / (precision + recall)
def gen_f1_score(y_true, y_pred):
    return f1_score(y_true, y_pred)

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
    train_acc, test_acc = [],[]

    for i in range(1,11):
        clf = KNeighborsClassifier(i)
        clf.fit(x_train, y_train)
        train_acc.append(clf.score(x_train, y_train))
        test_acc.append(clf.score(x_test, y_test))

    gen_plot(train_acc, test_acc)
    # Intepretation of plot; there are multiple high/low points (possible K), visually k=5 still seems like a good fit 


def load_from_sklearn():
    cancer_ds = load_breast_cancer()
    # print(type(cancer_ds)) sklearn bunch
    df = pd.DataFrame(cancer_ds.data, columns=cancer_ds.feature_names)
    df['target'] = cancer_ds.target
    # print(df.info) 569 rows, 31 cols (inc target), no visible missing values
    # random state - shuffling with reproducibility, stratify - class labels
    x_train, x_test, y_train, y_test = train_test_split(cancer_ds.data, cancer_ds.target, stratify=cancer_ds.target, random_state=66)
    
    train, test = [], []
    for i in range(1,11):
        clf = KNeighborsClassifier(n_neighbors=i)
        clf.fit(x_train, y_train)
        train.append(clf.score(x_train, y_train))
        test.append(clf.score(x_test, y_test))
    
    gen_plot(train, test)
    # best fit when k=6

    clf = KNeighborsClassifier(6)
    clf.fit(x_train, y_train)
    gen_confusion_matrix(clf, x_test, y_test)
    # TP 48 | FP 5
    # FN 4  | TN 86
    # high accuracy because true classifications are much larger values than FN and FP

#load_from_txt()
load_from_sklearn()

