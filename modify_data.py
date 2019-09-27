import pandas as pd
import os
import csv
from sklearn import preprocessing

# get data
path = os.getcwd()+'/baseballdata.csv'
df = pd.read_csv(path)

# drop rows
df = df.drop(['PLAYER', 'Record_ID#', 'ROOKIE', 'POS'], axis=1)

#normalize
x = df.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
df = pd.DataFrame(x_scaled)

print(df.head())
df.to_csv(r'x_modified.csv')
