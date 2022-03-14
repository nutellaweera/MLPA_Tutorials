import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets

diabetes_ds = datasets.load_diabetes(as_frame=True)
#print(type(diabetes_ds.data))
df = diabetes_ds.data
y = diabetes_ds.target
x = df['bmi']

plt.scatter(x,y)
plt.xlabel('BMI')
plt.ylabel('Dependent var (y)')
plt.show()