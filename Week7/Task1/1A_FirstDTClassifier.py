import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import plot_confusion_matrix

cancer = load_breast_cancer()

# EDA commented
# print(f'cancer.keys(): {cancer.keys()}')
# print(f'shape: {cancer.data.shape}')

# sample_counts = {n: v for n, v in zip(cancer.target_names, np.bincount(cancer.target))}
# print(f'Sample counts per class:\n{sample_counts}')

# print(f'Feature names:\n{cancer.feature_names}')

x_train, x_test, y_train, y_test = train_test_split(cancer.data, cancer.target, stratify=cancer.target, random_state=42)

tree = DecisionTreeClassifier(random_state=0)
tree.fit(x_train, y_train)

print(f'Accuracy on training set: {round(tree.score(x_train, y_train), 3)}')
print(f'Accuracy on test set: {round(tree.score(x_test, y_test), 3)}')


# The importance of a feature is computed as the (normalized) total reduction of the criterion brought by that feature.
# Gini index
print(f'Feature importances: {tree.feature_importances_}')

# confusion matrix
plot_confusion_matrix(tree, x_test, y_test, cmap=plt.cm.Blues)
plt.show()
# TP 84 | FP  4
# FN  5 | TN 86

# For knn (week 4) cf was
#  48 |  5
#   4 | 86