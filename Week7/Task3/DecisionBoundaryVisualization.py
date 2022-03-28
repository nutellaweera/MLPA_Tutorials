import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

x, y = make_blobs(n_samples=300, centers=4, random_state=0, cluster_std=1.0)
plt.scatter(x[:,0], x[:,1], c=y, s=50, cmap='rainbow')
#plt.show()

tree = DecisionTreeClassifier().fit(x,y)

def visualize_classifier(model, x,y,ax=None, cmap='rainbow'):
    ax = ax or plt.gca()
    ax.scatter(x[:,0], x[:,1], c=y, s=30, cmap=cmap, clim=(y.min(), y.max()), zorder=3)
    ax.axis('tight')
    ax.axis('off')
    xlim=ax.get_xlim()
    ylim=ax.get_ylim()

    model.fit(x,y)
    xx, yy = np.meshgrid(np.linspace(*xlim, num=200), np.linspace(*ylim, num=200))
    z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    n_classes = len(np.unique(y))
    ax.contourf(xx, yy, z, alpha=0.3, levels=np.arange(n_classes+1)-0.5, cmap=cmap, clim=(y.min(), y.max()), zorder=1)

    ax.set(xlim=xlim, ylim=ylim)
    plt.show()

visualize_classifier(tree, x, y)

