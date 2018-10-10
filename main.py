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

features = ['date','bedrooms','bathrooms','sqft_living','sqft_lot','floors','waterfront','view','condition','grade','sqft_above','sqft_basement','yr_built','yr_renovated','zipcode','lat','long','sqft_living15','sqft_lot15']

w_dict = {'date':0,'bedrooms':0,'bathrooms':0,'sqft_living':0,'sqft_lot':0,'floors':0,'waterfront':0,'view':0,'condition':0,'grade':0,'sqft_above':0,'sqft_basement':0,'yr_built':0,'yr_renovated':0,'zipcode':0,'lat':0,'long':0,'sqft_living15':0,'sqft_lot15':0}

D = pd.read_csv("PA1_train.csv") # /// if index_column set true, will exclude the index column and not put into consideration
# print(D)

colnames = D.columns.values.tolist() # colnames is an array
x_data = D[colnames[2:-1]] # from date to the end, excluding id
y_data = D[colnames[-1:]] # the prices only

# for i in features:
    # print(w_dict[i])

# stores initial w and b
b = -120 #initial b
for i in features: w_dict[i] = -4

lr = 0.0001 #learning rate
iteration = 100000

# b_history = [b]
# w_history = [w_dict]

# print(w_dict)

# for i in range (iteration):
#     b_grad = 0.0
#     w_grad = [0.0 for _ in range(0,len(x_data))]
#     for j in range(len(x_data)):
#         b_grad = b_grad - 2.0 * (y_data[j] - b - w * x_data[j]) * 1.0
#         for n in features:
#             w_grad = w_grad - 2.0 * (y_data[j] - b - w * x_data[j]) * x_data[j]
#             w = w - lr * w_grad
#         b = b - lr * b_grad
