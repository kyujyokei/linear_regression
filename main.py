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

features = ['day','month','year','bedrooms','bathrooms','sqft_living','sqft_lot','floors','waterfront','view','condition','grade','sqft_above','sqft_basement','yr_built','yr_renovated','zipcode','lat','long','sqft_living15','sqft_lot15']

w_dict = {'day':0,'month':0,'year':0,'bedrooms':0,'bathrooms':0,'sqft_living':0,'sqft_lot':0,'floors':0,'waterfront':0,'view':0,'condition':0,'grade':0,'sqft_above':0,'sqft_basement':0,'yr_built':0,'yr_renovated':0,'zipcode':0,'lat':0,'long':0,'sqft_living15':0,'sqft_lot15':0}

dateparse = lambda x: pd.datetime.strptime(x, '%m/%d/%Y')
D = pd.read_csv("PA1_train.csv",parse_dates=['date'], date_parser=dateparse) # /// if index_column set true, will exclude the index column and not put into consideration
# print(D.head())

def splitDate(df):
    # df = df.copy()
    df['year'] = pd.DatetimeIndex(df['date']).year
    df['month'] = pd.DatetimeIndex(df['date']).month
    df['day'] = pd.DatetimeIndex(df['date']).day
    return df

colnames = D.columns.values.tolist() # colnames is an array
x_data = D[colnames[2:-1]] # from date to the end, excluding id
y_data = D[colnames[-1:]] # the prices only

splitDate(x_data)
# print(x_data.head())
# print(x_data['year'].head())

# for i in features:
    # print(w_dict[i])

# stores initial w and b
b = -120 #initial b
for i in features: w_dict[i] = -4.0

lr = 0.0001 #learning rate
iteration = 1

# b_history = [b]
# w_history = [w_dict]

# print(y_data['price'])

for i in range (iteration):
    b_grad = 0.0
    # for j in range(len(x_data)):
    for j in range(0,1):
        w_grad = [0.0 for _ in range(0, len(x_data))]
        for k, n in enumerate(features):

            b_grad = b_grad - 2.0 * (y_data['price'][j] - b - w_dict[n] * x_data[n][j]) * 1.0

            w_grad[k] = w_grad[k] - 2.0 * (y_data['price'][j] - b - w_dict[n] * x_data[n][j]) * x_data[n][j]

            w_dict[n] = w_dict[n] - lr * w_grad[k]


        b = b - lr * b_grad


print("B:", b)
print("W:", w_dict)