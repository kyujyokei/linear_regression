import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
# matplotlib.use('Agg')
# %matplotlib inline
import random as random
import numpy as np
import csv
from sklearn import linear_model
import statsmodels.api as sm
import matplotlib.pyplot as plt

D = pd.read_csv("PA1_train.csv", index_col=False) # /// if index_column set true, will exclude the index column and not put into consideration
# print(D.head())

colnames = D.columns.values.tolist() # colnames is an array
x_data = D[colnames[2:-1]] # from date to the end, excluding id
y_data = D[colnames[-1:]] # the prices only

b = -120 #initial b

w = [ -4 for _ in range(0,len(colnames-3))] # -3 is to exclude dummy, id and price columns

# print(w)


# print(X)
# print(y_data.head())
# print(x_data.head())