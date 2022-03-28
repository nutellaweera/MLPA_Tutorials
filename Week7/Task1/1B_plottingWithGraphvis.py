import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import export_graphviz
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer

from IPython.display import display

DOT_FILE = 'Week7/Task1/tree.dot'

cancer = load_breast_cancer()

x_train, x_test, y_train, y_test = train_test_split(cancer.data, cancer.target, stratify=cancer.target, random_state=42)

tree = DecisionTreeClassifier(random_state=0)
tree.fit(x_train, y_train)

export_graphviz(tree, out_file=DOT_FILE, class_names=['malignant', 'benign'], feature_names=cancer.feature_names, impurity=False, filled=True)
import graphviz

with open(DOT_FILE) as f:
    dot_graph = f.read()

# TODO: Figure out how to display graph
#display(graphviz.Source(dot_graph))
#dot_graph.render(graphviz.Source(dot_graph))

def plot_feature_importances_cancer(model):
    n_features = cancer.data.shape[1]
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), cancer.feature_names)
    plt.xlabel('Feature importance')
    plt.ylabel('Feature')
    plt.ylim(-1, n_features)
    plt.show()

# plot_feature_importances_cancer(tree)
