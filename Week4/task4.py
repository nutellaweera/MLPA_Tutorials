import numpy as np
import pandas as pd

import mglearn

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

ds = mglearn.datasets.make_forge()
print(type(ds))
print(ds)
x,y = ds

# df = pd.DataFrame(ds)
# print(df.head())

#x_train, x_test, y_train, y_test = train_test_split(x,y, random_state = 0)

