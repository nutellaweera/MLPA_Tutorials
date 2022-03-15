import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split

# based on https://www.kaggle.com/patrickparsa/boston-housing-linear-regression/notebook 

boston = datasets.load_boston()
x = boston.data
cols = boston.feature_names

vif = pd.DataFrame()
vif['feature'] = cols
vif['VIF'] = [variance_inflation_factor(x, i) for i in range(len(cols))]

print(vif)

# feature        VIF
# 0      CRIM   2.100373
# 1        ZN   2.844013
# 2     INDUS  14.485758
# 3      CHAS   1.152952
# 4       NOX  73.894947
# 5        RM  77.948283
# 6       AGE  21.386850
# 7       DIS  14.699652
# 8       RAD  15.167725
# 9       TAX  61.227274
# 10  PTRATIO  85.029547
# 11        B  20.104943
# 12    LSTAT  11.102025