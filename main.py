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

print(x_data)

# b = -120 #initial b
# w = [ -4 for _ in range(0,len(x_data))] # an array of all the Ws for each feature
#
# lr = 0.000001 #learning rate
# iteration = 100000
#
# b_history = [b]
# w_history = [w]
#
# for i in range (iteration):
#     b_grad = 0.0
#     w_grad = [0.0 for _ in range(0,len(x_data))]
#     for j in range(len(x_data)):
#         b_grad = b_grad - 2.0*(y_data[j] - b - w*x_data[j])*1.0
#         w_grad = w_grad - 2.0*(y_data[j] - b - w*x_data[j])*x_data[j]
#     b = b - lr * b_grad
#     w = w - lr * w_grad

# print(w)
# print(X)
# print(y_data.head())
# print(x_data.head())